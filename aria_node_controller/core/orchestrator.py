import os
import threading
import time
import requests
import json
import base64

from .queue_manager import AriaQueueManager
from .batch_optimizer import BatchOptimizer
from .logger import get_logger
from .config_manager import SAMBA_PATH

from .models import AriaTaskResult

logger = get_logger("node.orchestrator")

FISH_TTS_HOST = "http://localhost:8080"
FISH_ENCODE_HOST = "http://localhost:8081"

class NodeOrchestrator:
    def __init__(self, redis_client):
        self.qm = AriaQueueManager(redis_client)
        self.optimizer = BatchOptimizer(redis_client)
        self.running = False
        self.thread = None
        
        # Semaforo locale copiato dalla Tray Icon
        self.semaphore_green = True
        
        # Cache RAM per token cloni
        self.token_cache = {}
        
    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("Orchestrator thread started")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)
        logger.info("Orchestrator thread stopped")

    def set_semaphore(self, state: bool):
        self.semaphore_green = state
        logger.info(f"Orchestrator semaphore set to {'GREEN' if state else 'RED'}")

    def _run_loop(self):
        current_model = None
        known_models = {
            "fish-s1-mini": BatchOptimizer.build_queue_key("tts", "fish-s1-mini")
        }
        
        while self.running:
            if not self.semaphore_green:
                time.sleep(2)
                continue

            try:
                # 1. Ask optimizer for next queue
                decision = self.optimizer.decide_next_queue(known_models, current_model)
                if not decision:
                    time.sleep(1)
                    continue

                next_model_id, queue_key = decision
                if current_model != next_model_id:
                    print(f"DEBUG: Switching batch focus to model: {next_model_id} (queue: {queue_key})")
                    logger.info(f"Switching batch focus to model: {next_model_id} (queue: {queue_key})")
                    current_model = next_model_id

                # 2. Consuma e Processa il Task
                raw_json, payload = self.qm.fetch_task(queue_key, timeout=2)
                if not payload:
                    continue  # Timeout o vuota

                print(f"DEBUG: Processando task {payload.job_id} for {payload.model_id}")

                logger.info(f"Processing task {payload.job_id} for {payload.model_id}")
                self._process_task(payload)

            except Exception as e:
                print(f"CRITICAL ERROR in orchestrator loop: {e}")
                logger.error("Error in orchestrator loop", exc_info=True)
                time.sleep(5)

    def _encode_audio_to_tokens(self, audio_path: str) -> bytes:
        logger.info(f"Encoding reference audio from {audio_path}...")
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            resp = requests.post(f"{FISH_ENCODE_HOST}/v1/vqgan/encode", files={"audio": ("ref.wav", audio_bytes)}, timeout=30)
            resp.raise_for_status()
            logger.info("Tokens encoded successfully.")
            return resp.content
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise

    def _process_task(self, task):
        start_t = time.time()
        # Questo è specifico per fish-s1-mini per ora. 
        # In base al tipo di task chiameremo processi diversi (es. Llama su p.5007)
        if task.model_id == "fish-s1-mini":
            try:
                # Prepara i Reference Tokens
                tokens = None
                voice_sample_path = task.payload.get("voice_sample_path")
                
                if voice_sample_path:
                    # Riscrivi il percorso Samba per renderlo compatibile con Z:\ (Windows)
                    win_path = voice_sample_path.replace("/mnt/aria-shared/", SAMBA_PATH).replace("/aria-shared/", SAMBA_PATH).replace("/", "\\")
                    
                    if win_path in self.token_cache:
                        tokens = self.token_cache[win_path]
                    else:
                        tokens = self._encode_audio_to_tokens(win_path)
                        self.token_cache[win_path] = tokens

                # Synthesize using ServeTTSRequest schema
                data = {
                    "text": task.payload.get("text", ""),
                    "format": task.payload.get("output_format", "wav"),
                    "streaming": False
                }
                
                # Aggiungiamo il token vocale decodificato come riferimento per il Voice Cloning
                if tokens:
                    import base64
                    b64_str = base64.b64encode(tokens).decode("utf-8")
                    data["references"] = [{
                        "audio": b64_str,
                        "text": task.payload.get("voice_ref_text", "Sample voice audio reference")
                    }]

                logger.info(f"Requesting TTS Synthesis to Fish Server at {FISH_TTS_HOST}/v1/tts")
                resp = requests.post(f"{FISH_TTS_HOST}/v1/tts", json=data, timeout=300)
                try:
                    resp.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    logger.error(f"HTTP Error {e.response.status_code} from Fish TTS Server. Response text: {e.response.text}")
                    raise
                audio_bytes = resp.content
                duration_s = time.time() - start_t
                
                # Save Output
                out_path = task.payload.get("output_path", "")
                if out_path:
                    from pathlib import Path
                    win_out_path = out_path.replace("/mnt/aria-shared/", SAMBA_PATH).replace("/aria-shared/", SAMBA_PATH).replace("/", "\\")
                    
                    try:
                        # Pathlib gestisce in modo eccellente i Mount Point Samba su Windows (Z:\) 
                        # ignorando i PermissionError passivi alla radice.
                        Path(win_out_path).parent.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        logger.warning(f"Pathlib could not silently bypass Samba root perms, falling back... ({e})")
                        # Fallback manuale finale per saltare il drive (se pathlib dovesse piangere)
                        drive, tail = os.path.splitdrive(os.path.dirname(win_out_path))
                        if drive:
                            curr_path = drive + os.sep
                            for folder in [f for f in tail.split(os.sep) if f]:
                                curr_path = os.path.join(curr_path, folder)
                                try:
                                    if not os.path.exists(curr_path):
                                        os.mkdir(curr_path)
                                except Exception:
                                    pass

                    with open(win_out_path, "wb") as f:
                        f.write(audio_bytes)
                    logger.info(f"Wrote generated WAV to {win_out_path}")

                # Rispondi a Redis
                result = AriaTaskResult(
                    job_id=task.job_id,
                    client_id=task.client_id,
                    model_type=task.model_type,
                    model_id=task.model_id,
                    status="done",
                    processing_time_seconds=duration_s,
                    output={"output_path": out_path, "duration_seconds": duration_s}
                )
                self.qm.post_result(task, result)
            
            except Exception as e:
                logger.error(f"Task Failed: {e}", exc_info=True)
                result = AriaTaskResult(
                    job_id=task.job_id,
                    client_id=task.client_id,
                    model_type=task.model_type,
                    model_id=task.model_id,
                    status="error",
                    processing_time_seconds=time.time() - start_t,
                    error=str(e)
                )
                self.qm.post_result(task, result)
        else:
            logger.warning(f"Unsupported model_id: {task.model_id}")
