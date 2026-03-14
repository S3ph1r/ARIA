import time
import threading
import logging
import os
import json
from typing import Dict, Any, Type, Optional
from pathlib import Path

from .logger import get_logger
from .models import AriaTaskPayload, AriaTaskResult
from .queue_manager import AriaQueueManager

logger = get_logger("node.cloud")

class CloudManager:
    """
    Manages sequential execution of cloud-based AI tasks.
    Runs in a background thread to avoid blocking the GPU orchestrator.
    
    NOTE: Sequential mode ensures only one cloud task is processed at a time
    by this manager, even if multiple backends are registered.
    """
    
    def __init__(self, queue_manager: AriaQueueManager, aria_root: Path):
        self.qm = queue_manager
        self.aria_root = aria_root
        self.backends = {} # provider_id -> backend_class
        self._stop_event = threading.Event()
        self._thread = None
        
    def register_backend(self, provider: str, backend_class: Type):
        """Registers a backend class for a specific provider (e.g., 'google')."""
        self.backends[provider] = backend_class
        logger.info(f"Cloud backend registered for provider: {provider}")

    def start(self):
        """Starts the background monitoring loop."""
        if self._thread and self._thread.is_alive():
            logger.warning("CloudManager is already running.")
            return
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._main_loop, daemon=True, name="CloudManagerLoop")
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
                    time.sleep(1)
                    continue

                task_processed = False
                for queue_key in cloud_queues:
                    # Fetch one task from the queue
                    # fetch_task handles the 'global:processing' lock
                    raw_json, task = self.qm.fetch_task(queue_key, timeout=1)
                    
                    if task:
                        # SEQUENTIAL EXECUTION:
                        # We process the task directly in this loop thread.
                        # This blocks the loop from fetching the next task 
                        # until this one is finished.
                        self._run_task(task)
                        task_processed = True
                        break # Go back to start of loop to check priorities/new queues
                
                if not task_processed:
                    # Small sleep if no tasks were found in any queue
                    time.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error in CloudManager main loop: {e}")
                time.sleep(2)

    def _run_task(self, task: AriaTaskPayload):
        """Executes a single cloud task using an external process worker."""
        import subprocess
        import sys
        
        start_time = time.time()
        
        # Determine worker script path
        # In Step 4, we only have google/gemini implemented
        if task.provider != "google":
            logger.error(f"Provider {task.provider} not supported in CloudManager.")
            # ... (error result logic)
            return

        worker_script = self.aria_root / "aria_node_controller" / "backends" / "cloud" / "gemini_worker.py"
        
        # Select best python for cloud worker
        # On Windows (PC 139), we want the isolated aria-cloud env if it exists
        worker_python = sys.executable
        win_cloud_env = self.aria_root / "envs" / "aria-cloud" / "python.exe"
        if os.name == "nt" and win_cloud_env.exists():
            worker_python = str(win_cloud_env)
            logger.info(f"Using isolated cloud environment: {worker_python}")
        
        try:
            logger.info(f"Spawning worker process for task {task.job_id}...")
            
            # Formulate payload for the worker
            worker_payload = task.payload.copy()
            worker_payload["job_id"] = task.job_id
            worker_payload["client_id"] = task.client_id
            
            # Execute worker sequentially (blocking)
            result_process = subprocess.run(
                [worker_python, str(worker_script), json.dumps(worker_payload)],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result_process.returncode != 0:
                logger.error(f"Worker process crashed: {result_process.stderr}")
                raise RuntimeError(f"Worker crash: {result_process.stderr}")

            # Parse result from stdout
            try:
                worker_result = json.loads(result_process.stdout)
            except json.JSONDecodeError:
                raise RuntimeError(f"Invalid JSON from worker: {result_process.stdout[:100]}...")

            if worker_result.get("status") == "success":
                result = AriaTaskResult(
                    job_id=task.job_id,
                    client_id=task.client_id,
                    model_type=task.model_type,
                    provider=task.provider,
                    model_id=task.model_id,
                    status="done",
                    processing_time_seconds=time.time() - start_time,
                    output=worker_result.get("output")
                )
            else:
                raise RuntimeError(worker_result.get("error", "Unknown worker error"))

        except Exception as e:
            logger.error(f"Cloud task {task.job_id} failed: {e}")
            
            error_msg = str(e)
            error_code = "CLOUD_ERROR"
            
            if "QUOTA_EXHAUSTED" in error_msg or "429" in error_msg:
                error_code = "QUOTA_EXHAUSTED"
            
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

        # Post result and clear lock
        self.qm.post_result(task, result)
        logger.info(f"Cloud task {task.job_id} completed with status: {result.status}")
