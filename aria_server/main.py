import os
import sys
import time
import logging
import threading
import uvicorn
import redis
import requests
from dotenv import load_dotenv

from aria_server.queue_manager import AriaQueueManager
from aria_server.batch_optimizer import BatchOptimizer
from aria_server.semaphore import AriaSemaphore
from aria_server.vram_manager import VRAMManager
from aria_server.heartbeat import HeartbeatThread
from aria_server.backends.orpheus import OrpheusBackend
from aria_server.backends.fish_tts import FishTTSBackend
from aria_server.api.http_api import create_api

from aria_server.logger import setup_logging, get_logger

# Setup logging
setup_logging(log_level_name="INFO", console_only=False)
logger = get_logger("aria.main")

def main():
    load_dotenv()
    
    # 1. Connect to Redis
    redis_host = os.environ.get("REDIS_HOST", "192.168.1.120")
    redis_port = int(os.environ.get("REDIS_PORT", 6379))
    logger.info(f"Connecting to Redis at {redis_host}:{redis_port}...")
    
    try:
        redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
        redis_client.ping()
        logger.info("Redis connected beautifully.")
    except Exception as e:
        logger.critical(f"FATAL: Could not connect to Redis: {e}")
        sys.exit(1)

    # 2. Init Core Infrastructure Components
    queue_manager = AriaQueueManager(redis_client)
    semaphore = AriaSemaphore(redis_client)
    optimizer = BatchOptimizer(redis_client)
    vram_manager = VRAMManager()
    
    # 3. Start Heartbeat (Daemon Thread)
    heartbeat = HeartbeatThread(redis_client)
    heartbeat.start()

    # 4. Register Backends
    llama_host = os.environ.get("LLAMA_HOST", "http://host.docker.internal:5007")
    fish_encoder_host = os.environ.get("FISH_ENCODER_HOST", "http://192.168.1.139:8081")
    fish_tts_host = os.environ.get("FISH_TTS_HOST", "http://192.168.1.139:8080")
    
    backends = {
        # Schema: "model_type:model_id" -> Backend Instance
        "tts:orpheus-3b": OrpheusBackend(llama_url=llama_host),
        "tts:fish-s1-mini": FishTTSBackend(encoder_url=fish_encoder_host, tts_url=fish_tts_host)
    }
    
    # Build a lookup for the optimizer: logic_id -> actual queue key
    # e.g., {"tts:orpheus-3b": "gpu:queue:tts:orpheus-3b"}
    known_models = {
        key: BatchOptimizer.build_queue_key(*key.split(":")) for key in backends.keys()
    }
    
    logger.info(f"Registered Backends: {list(known_models.keys())}")

    # 5. Start FastAPI HTTP Server (Daemon Thread)
    api_port = int(os.environ.get("API_PORT", 7860))
    app_context = {
        "semaphore": semaphore,
        "vram_manager": vram_manager,
        "batch_optimizer": optimizer,
        "known_models": known_models
    }
    fastapi_app = create_api(app_context)
    
    def run_api():
        logger.info(f"Starting FastAPI server on port {api_port}...")
        uvicorn.run(fastapi_app, host="0.0.0.0", port=api_port, log_level="warning")
        
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    # 6. Main Orchestrator Loop
    logger.info("ARIA Orchestrator is running and waiting for tasks...")
    
    try:
        while True:
            # 6a. Check Semaphore
            if not semaphore.is_green():
                logger.debug("GPU Semaphore is RED. Paused. Waiting 10s...")
                time.sleep(10)
                continue
                
            # 6b. Optimize / Route
            decision = optimizer.decide_next_queue(known_models, vram_manager.current_model_id)
            if not decision:
                # All queues empty, take a breath
                time.sleep(2)
                continue
                
            logic_id, queue_key = decision
            selected_backend = backends[logic_id]
            
            # 6c. Load VRAM (if switched)
            if vram_manager.current_model_id != logic_id:
                # If a different model was loaded, unload it
                if vram_manager.current_model_id:
                    old_backend = backends[vram_manager.current_model_id]
                    vram_manager.unload(old_backend)
                
                # Load the new one
                vram_manager.load(selected_backend)
            
            # 6d. Fetch Task (Timeout blocks loop briefly)
            raw_json, task = queue_manager.fetch_task(queue_key, timeout=5)
            if not task:
                continue
                
            logger.info(f"Processing task {task.job_id} on {logic_id}...")
            
            # 6e. Execution Phase
            semaphore.set_busy()
            
            try:
                result = selected_backend.process_task(task)
            except Exception as e:
                logger.exception(f"Unhandled backend exception during {task.job_id}")
                # We need to construct a fallback error result if the backend crashed entirely
                from aria_server.models import AriaTaskResult
                result = AriaTaskResult(
                    job_id=task.job_id, client_id=task.client_id,
                    model_type=selected_backend.model_type, model_id=selected_backend.model_id,
                    status="error", processing_time_seconds=0.0,
                    error=str(e), error_code="FATAL_BACKEND_ERROR"
                )
            
            # 6f. Post Result & Cleanup
            queue_manager.post_result(task, result)
            semaphore.restore_green_if_busy()

    except KeyboardInterrupt:
        logger.info("Shutting down ARIA Orchestrator (KeyboardInterrupt)...")
    finally:
        heartbeat.stop()
        if vram_manager.current_model_id:
             vram_manager.unload(backends[vram_manager.current_model_id])
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()
