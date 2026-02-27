import requests
import time
from typing import Dict, Any

from aria_server.logger import get_logger
from .base import BaseAriaBackend
from ..models import AriaTaskPayload, AriaTaskResult
from ..config import settings

logger = get_logger("aria.backends.proxy")

class ExternalProxyBackend(BaseAriaBackend):
    """
    Backend implementation that forwards tasks to an external native server via HTTP.
    Used to bypass container limitations (e.g., WSL2 VRAM mmap bug for Llama.cpp).
    """
    
    def __init__(self, target_url: str):
        self.target_url = target_url

    def process_task(self, task: AriaTaskPayload) -> AriaTaskResult:
        logger.info(f"Avvio proxying per task {task.task_id} verso {self.target_url}")
        start_time = time.time()
        
        try:
            # Extract parameters tailored for llama.cpp compatible endpoints
            text_to_synthesize = task.parameters.get("text", "")
            
            payload = {
                "prompt": text_to_synthesize,
                # Add any other required parameters here mapped from task.parameters
            }
            
            response = requests.post(
                f"{self.target_url}/v1/completions", # Adjust path based on your exact Orpheus routing
                json=payload,
                timeout=120 # TTS can take time
            )
            response.raise_for_status() # Raise exception for 4XX/5XX errors
            
            # The exact response handling depends on what Orpheus returns (bytes, url, json string, etc.)
            # For now, assuming it returns JSON structure
            result_json = response.json()
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return AriaTaskResult(
                task_id=task.task_id,
                status="completed",
                result_data=result_json,
                execution_time_ms=execution_time
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Errore di rete durante il proxy del task {task.task_id}: {e}")
            return AriaTaskResult(
                task_id=task.task_id,
                status="failed",
                error_message=f"External server connection failed: {str(e)}",
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
        except Exception as e:
            logger.error(f"Errore inatteso nel proxy TTS per il task {task.task_id}: {e}")
            return AriaTaskResult(
                task_id=task.task_id,
                status="failed",
                error_message=f"Internal proxy error: {str(e)}",
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
