```python
from typing import Dict, Any, List
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aria_server.logger import get_logger

logger = get_logger("aria.http_api")

class SemaphoreState(BaseModel):
    state: str  # "green" or "red"

def create_api(context: Dict[str, Any]) -> FastAPI:
    """
    Factory function to create the FastAPI application, injecting dependencies
    from the main ARIA Orchestrator running context.
    
    context must include:
    - 'semaphore': AriaSemaphore instance
    - 'vram_manager': VRAMManager instance
    - 'batch_optimizer': BatchOptimizer instance
    - 'known_models': list of registered queue keys
    """
    app = FastAPI(title="ARIA Orchestrator API", version="2.0.0")

    @app.get("/health")
    def health_check():
        return {"status": "ok", "version": "2.0.0"}

    @app.get("/status")
    def get_status():
        semaphore = context.get('semaphore')
        vram = context.get('vram_manager')
        optimizer = context.get('batch_optimizer')
        known_models = context.get('known_models', {})
        
        queue_depths = optimizer.get_queue_depths(known_models) if optimizer else {}
        
        return {
            "semaphore": semaphore.get_state() if semaphore else "unknown",
            "model_loaded": vram.current_model_id if vram else None,
            "queue_depths": queue_depths
        }

    @app.post("/semaphore")
    def set_semaphore(payload: SemaphoreState):
        semaphore = context.get('semaphore')
        if not semaphore:
            raise HTTPException(status_code=500, detail="Semaphore module not loaded")
            
        state = payload.state.lower()
        if state == "green":
            semaphore.set_green()
        elif state == "red":
            semaphore.set_red()
        else:
            raise HTTPException(status_code=400, detail="Invalid state. Must be 'green' or 'red'.")
            
        logger.info(f"API Request: Semaphore state manually set to {state}")
        return {"semaphore": semaphore.get_state()}

    return app
