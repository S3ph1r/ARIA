import threading
import time
import requests
import json
import base64

from .queue_manager import AriaQueueManager
from .batch_optimizer import BatchOptimizer
from .logger import get_logger
from .config_manager import SAMBA_PATH

logger = get_logger("node.orchestrator")

FISH_TTS_HOST = "http://localhost:8080"
FISH_ENCODE_HOST = "http://localhost:8081"

class NodeOrchestrator:
    def __init__(self):
        self.qm = AriaQueueManager()
        self.optimizer = BatchOptimizer()
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
        
        while self.running:
            if not self.semaphore_green:
                time.sleep(2)
                continue

            try:
                # 1. Recupera lunghezze code
                queue_lens = self.qm.queue_lengths()
                
                # 2. Decide if we need to switch model priority
                if not current_model or self.optimizer.should_switch(current_model, queue_lens):
                    next_model = self.optimizer.next_model(queue_lens, current_model)
                    if next_model:
                        logger.info(f"Switching batch focus to model: {next_model}")
                        current_model = next_model

                if not current_model:
                    time.sleep(1)
                    continue

                # 3. Consuma e Processa il Task
                payload = self.qm.next_task(model_key=current_model)
                if not payload:
                    continue  # Timeout o vuota

                logger.info(f"Processing task {payload.job_id} for {payload.model_id}")
                self._process_task(payload)

            except Exception as e:
                logger.error("Error in orchestrator loop", exc_info=True)
                time.sleep(5)

    def _encode_audio_to_tokens(self, audio_path: str) -> bytes:
        logger.info(f"Encoding reference audio from {audio_path}...")
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            resp = requests.post(f"{FISH_ENCODE_HOST}/encode_audio", files={"audio": ("ref.wav", audio_bytes)}, timeout=30)
            resp.raise_for_status()
            logger.info("Tokens encoded successfully.")
            return resp.content
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            raise

    def _process_task(self, task):
        # Questo è specifico per fish-s1-mini per ora. 
        # In base al tipo di task chiameremo processi diversi (es. Llama su p.5007)
        if task.model_id == "fish-s1-mini":
            try:
                # Prepara i Reference Tokens
                tokens = None
                voice_sample_path = task.payload.get("voice_sample_path")
                
                if voice_sample_path:
                    # Riscrivi il percorso Samba per renderlo compatibile con Z:\ (Windows)
                    # Es: /aria-shared/voices/narrator_it.wav -> Z:\voices\narrator_it.wav
                    win_path = voice_sample_path.replace("/aria-shared/", SAMBA_PATH).replace("/", "\\")
                    
                    if win_path in self.token_cache:
                        tokens = self.token_cache[win_path]
                    else:
                        tokens = self._encode_audio_to_tokens(win_path)
                        self.token_cache[win_path] = tokens

                # Synthesize
                data = {"text": task.payload.get("text", "")}
                if tokens:
                    data["tokens"] = base64.b64encode(tokens).decode("utf-8")

                logger.info(f"Requesting TTS Synthesis to Fish Server at {FISH_TTS_HOST}/synthesize")
                start_t = time.time()
                resp = requests.post(f"{FISH_TTS_HOST}/synthesize", json=data, timeout=300)
                resp.raise_for_status()
                audio_bytes = resp.content
                duration_s = time.time() - start_t
                
                # Save Output
                out_path = task.payload.get("output_path", "")
                if out_path:
                    win_out_path = out_path.replace("/aria-shared/", SAMBA_PATH).replace("/", "\\")
                    os.makedirs(os.path.dirname(win_out_path), exist_ok=True)
                    with open(win_out_path, "wb") as f:
                        f.write(audio_bytes)
                    logger.info(f"Wrote generated WAV to {win_out_path}")

                # Rispondi a Redis
                self.qm.result_writer.write_result(
                    job_id=task.job_id,
                    client_id=task.client_id,
                    result_data={
                        "job_id": task.job_id,
                        "model_id": task.model_id,
                        "status": "done",
                        "output": {
                            "output_path": out_path,
                            "duration_seconds": duration_s
                        }
                    },
                    callback_key=task.callback_key
                )
            
            except Exception as e:
                logger.error(f"Task Failed: {e}", exc_info=True)
                self.qm.result_writer.write_result(
                    job_id=task.job_id,
                    client_id=task.client_id,
                    result_data={
                        "job_id": task.job_id,
                        "model_id": task.model_id,
                        "status": "error",
                        "error_message": str(e)
                    },
                    callback_key=task.callback_key
                )
        else:
            logger.warning(f"Unsupported model_id: {task.model_id}")
