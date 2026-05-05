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


def _is_daily_quota_error(error_msg: str) -> bool:
    """
    Distingue un 429 da quota giornaliera (RPD) da un 429 da rate-limit
    istantaneo (RPM/TPM). Controlla pattern nel messaggio di errore Google.
    """
    indicators = [
        "GenerateRequestsPerDayPerProjectPerModel",
        "PerDay",
        "daily",
        "PREVENTIVE_QUOTA_EXHAUSTED",
    ]
    return any(k in error_msg for k in indicators)


class CloudManager:
    """
    Manages cloud-based LLM tasks for ARIA.
    Acts as a gateway to external providers (Gemini, Vertex, etc.)
    avoiding the local GPU semaphore.
    """

    def __init__(
        self,
        queue_manager: AriaQueueManager,
        aria_root: Path,
        rate_limiter: Optional[GeminiRateLimiter] = None,
    ):
        self.qm           = queue_manager
        self.aria_root    = aria_root
        self.rate_limiter = rate_limiter or GeminiRateLimiter(
            redis_client=queue_manager.redis
        )
        self._stop_event  = threading.Event()
        self._thread      = None

        # Isolated Python env for cloud tasks
        self.cloud_env = None
        python_exe     = "python.exe" if os.name == "nt" else "python"
        potential_env  = aria_root / "envs" / "aria-cloud" / python_exe
        if potential_env.exists():
            self.cloud_env = str(potential_env)
        else:
            import sys
            self.cloud_env = sys.executable

    def start(self):
        if self._thread and self._thread.is_alive():
            logger.warning("CloudManager is already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._main_loop, daemon=True)
        self._thread.start()
        logger.info("CloudManager started (Sequential Mode).")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("CloudManager stopped.")

    def _main_loop(self):
        while not self._stop_event.is_set():
            try:
                cloud_queues = list(
                    self.qm.redis.scan_iter(match="aria:q:cloud:*:*:*")
                )
                if not cloud_queues:
                    time.sleep(2)
                    continue

                for queue_key in cloud_queues:
                    if self._stop_event.is_set():
                        break
                    raw_json, task = self.qm.fetch_task(queue_key, timeout=1)
                    if task:
                        logger.info(
                            f"Processing cloud task {task.job_id} from {queue_key}"
                        )
                        self.process_cloud_task(task)

                time.sleep(1)

            except Exception as e:
                logger.error(f"Error in CloudManager main loop: {e}", exc_info=True)
                time.sleep(5)

    def process_cloud_task(self, task: AriaTaskPayload):
        start_time = time.time()

        try:
            # 1. Pacing slot (Google only)
            if task.provider == "google":
                logger.info(f"Task {task.job_id} requesting global pacing slot...")
                self.rate_limiter.wait_for_slot()

            # 2. Worker subprocess
            worker_script = (
                self.aria_root
                / "aria_node_controller"
                / "backends"
                / "cloud"
                / "gemini_worker.py"
            )
            if not worker_script.exists():
                raise RuntimeError(f"Cloud worker script not found at {worker_script}")

            worker_env = os.environ.copy()
            if "GOOGLE_API_KEY" not in worker_env:
                from dotenv import load_dotenv
                load_dotenv()
                worker_env["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

            worker_payload = {
                **task.payload,
                "job_id":    task.job_id,
                "client_id": task.client_id,
                "model_id":  task.model_id,
            }
            payload_json = json.dumps(worker_payload)

            logger.info(f"Spawning worker process for task {task.job_id}...")
            result_process = subprocess.run(
                [self.cloud_env, str(worker_script)],
                input=payload_json,
                capture_output=True,
                text=True,
                check=False,
                env=worker_env,
            )

            if result_process.returncode != 0:
                logger.error(f"Worker process crashed: {result_process.stderr}")
                raise RuntimeError(f"Worker crash: {result_process.stderr}")

            try:
                worker_result = json.loads(result_process.stdout)
            except json.JSONDecodeError:
                raise RuntimeError(
                    f"Invalid JSON from worker: {result_process.stdout[:200]}..."
                )

            if worker_result.get("status") == "success":
                # Registra utilizzo RPM + TPM per tracking e dashboard
                usage = worker_result.get("usage", {})
                total_tokens = (
                    usage.get("total_tokens")
                    or usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                )
                if task.provider == "google":
                    self.rate_limiter.record_usage(tokens=total_tokens)

                result = AriaTaskResult(
                    job_id=task.job_id,
                    client_id=task.client_id,
                    model_type=task.model_type,
                    provider=task.provider,
                    model_id=task.model_id,
                    status="done",
                    processing_time_seconds=time.time() - start_time,
                    output=worker_result.get("output", {}),
                    usage=usage,
                )
            else:
                raise RuntimeError(
                    f"Worker reported error: {worker_result.get('error')}"
                )

        except Exception as e:
            logger.error(f"Cloud task {task.job_id} failed: {e}")

            error_msg  = str(e)
            error_code = "CLOUD_ERROR"

            if "QUOTA_EXHAUSTED" in error_msg or "429" in error_msg:
                error_code = "QUOTA_EXHAUSTED"
                if task.provider == "google":
                    if _is_daily_quota_error(error_msg):
                        # RPD esaurito: lockout lungo fino al reset PDT
                        self.rate_limiter.report_daily_quota_exhausted()
                    else:
                        # RPM / rate-limit istantaneo: lockout breve
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
                error_code=error_code,
            )

        # Post result verso il client (DIAS o altro)
        self.qm.post_result(task, result)

        logger.info(f"Cloud task {task.job_id} completed. Cooling down for 60s...")
        time.sleep(60)
        logger.info(f"Cloud task {task.job_id} completed with status: {result.status}")
