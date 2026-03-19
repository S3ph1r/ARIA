import time
import logging
import subprocess
import json
import os
import threading
from pathlib import Path
from typing import Optional, Type
from .queue_manager import AriaQueueManager
from .models import AriaTaskPayload, AriaTaskResult
from .rate_limiter import GeminiRateLimiter

logger = logging.getLogger("node.cloud")

class CloudManager:
    """
    Manages cloud-based LLM tasks for ARIA.
    Acts as a gateway to external providers (Gemini, Vertex, etc.) 
    avoiding the local GPU semaphore.
    """
    
    def __init__(self, queue_manager: AriaQueueManager, aria_root: Path, rate_limiter: Optional[GeminiRateLimiter] = None):
        self.qm = queue_manager
        self.aria_root = aria_root
        self.rate_limiter = rate_limiter or GeminiRateLimiter(redis_client=queue_manager.redis)
        self._stop_event = threading.Event()
        self._thread = None
        
        # Determine the isolated environment for cloud tasks
        self.cloud_env = None
        python_exe = "python.exe" if os.name == "nt" else "python"
        potential_env = aria_root / "envs" / "aria-cloud" / python_exe
        if potential_env.exists():
            self.cloud_env = str(potential_env)
        else:
            # Fallback to current sys.executable if specialized env not found
            import sys
            self.cloud_env = sys.executable

    def start(self):
        """Starts the background monitoring loop."""
        if self._thread and self._thread.is_alive():
            logger.warning("CloudManager is already running.")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._main_loop, daemon=True)
        self._thread.start()
        logger.info("CloudManager started (Sequential Mode).")

    def stop(self):
        """Stops the monitoring loop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("CloudManager stopped.")

    def _main_loop(self):
        """Main polling loop for cloud queues."""
        while not self._stop_event.is_set():
            try:
                # Dynamic discovery of cloud queues
                # Pattern: global:queue:cloud:{provider}:{model_id}:{client_id}
                cloud_queues = list(self.qm.redis.scan_iter(match="global:queue:cloud:*:*:*"))
                
                if not cloud_queues:
                    time.sleep(2)
                    continue

                for queue_key in cloud_queues:
                    if self._stop_event.is_set(): break
                    
                    # Fetch one task (non-blocking here, we poll)
                    raw_json, task = self.qm.fetch_task(queue_key, timeout=1)
                    if task:
                        logger.info(f"Processing cloud task {task.job_id} from {queue_key}")
                        self.process_cloud_task(task)
                        # Process one task per loop iteration to allow stop/check
                
                time.sleep(1) # Small rest between scans

            except Exception as e:
                logger.error(f"Error in CloudManager main loop: {e}", exc_info=True)
                time.sleep(5)

    def process_cloud_task(self, task: AriaTaskPayload):
        """
        Executes a cloud task using an isolated child process or direct API.
        """
        start_time = time.time()
        
        try:
            # 1. Wait for a pacing slot (centralized for Google)
            if task.provider == "google":
                logger.info(f"Task {task.job_id} requesting global pacing slot...")
                self.rate_limiter.wait_for_slot()

            # 2. Worker setup
            worker_script = self.aria_root / "aria_node_controller" / "backends" / "cloud" / "gemini_worker.py"
            if not worker_script.exists():
                 raise RuntimeError(f"Cloud worker script not found at {worker_script}")

            worker_python = self.cloud_env
            
            # Pass environment variables to the isolated subprocess
            worker_env = os.environ.copy()
            if "GOOGLE_API_KEY" not in worker_env:
                from dotenv import load_dotenv
                load_dotenv()
                worker_env["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
            
            # Prepare task JSON for the worker
            worker_payload = {
                **task.payload,
                "job_id": task.job_id,
                "client_id": task.client_id,
                "model_id": task.model_id
            }
            payload_json = json.dumps(worker_payload)
            
            # Execute worker sequentially (blocking)
            logger.info(f"Spawning worker process for task {task.job_id}...")
            result_process = subprocess.run(
                [worker_python, str(worker_script)],
                input=payload_json,
                capture_output=True,
                text=True,
                check=False,
                env=worker_env
            )
            
            if result_process.returncode != 0:
                logger.error(f"Worker process crashed: {result_process.stderr}")
                raise RuntimeError(f"Worker crash: {result_process.stderr}")

            # Parse result from stdout
            try:
                worker_result = json.loads(result_process.stdout)
            except json.JSONDecodeError:
                raise RuntimeError(f"Invalid JSON from worker: {result_process.stdout[:200]}...")

            if worker_result.get("status") == "success":
                result = AriaTaskResult(
                    job_id=task.job_id,
                    client_id=task.client_id,
                    model_type=task.model_type,
                    provider=task.provider,
                    model_id=task.model_id,
                    status="done",
                    processing_time_seconds=time.time() - start_time,
                    output=worker_result.get("output", {}),
                    usage=worker_result.get("usage", {})
                )
                if task.provider == "google":
                    self.rate_limiter.report_success()
            else:
                raise RuntimeError(f"Worker reported error: {worker_result.get('error')}")

        except Exception as e:
            logger.error(f"Cloud task {task.job_id} failed: {e}")
            
            error_msg = str(e)
            error_code = "CLOUD_ERROR"
            
            if "QUOTA_EXHAUSTED" in error_msg or "429" in error_msg:
                error_code = "QUOTA_EXHAUSTED"
                if task.provider == "google":
                    self.rate_limiter.report_429()
            
            result = AriaTaskResult(
                job_id=task.job_id,
                client_id=task.client_id,
                model_type=task.model_type,
                provider=task.provider,
                model_id=task.model_id,
                status="error",
                processing_time_seconds=time.time() - start_time,
                error=error_msg,
                error_code=error_code
            )

        # Post result
        self.qm.post_result(task, result)
        logger.info(f"Cloud task {task.job_id} completed with status: {result.status}")
