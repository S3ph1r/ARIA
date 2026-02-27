from typing import Optional
from aria_server.logger import get_logger
from aria_server.backends.base import BaseAriaBackend

logger = get_logger("aria.vram")

class VRAMManager:
    """
    Manages loading and unloading of models into VRAM to ensure only one
    large model is active at a time, preventing OOM errors.
    """
    def __init__(self):
        self._current_model_id: Optional[str] = None
        
    @property
    def current_model_id(self) -> Optional[str]:
        return self._current_model_id
        
    def load(self, backend: BaseAriaBackend) -> None:
        """Loads a backend's model into VRAM, unloading the previous one if necessary."""
        target_model = f"{backend.model_type}:{backend.model_id}"
        
        if self._current_model_id == target_model:
            return  # Already loaded
            
        logger.info(f"VRAMManager: Loading model {target_model} (est. {backend.estimated_vram_gb()} GB)")
        backend.load()
        self._current_model_id = target_model
        
    def unload(self, backend: BaseAriaBackend) -> None:
        """Unloads a backend's model from VRAM."""
        target_model = f"{backend.model_type}:{backend.model_id}"
        
        logger.info(f"VRAMManager: Unloading model {target_model}")
        backend.unload()
        
        if self._current_model_id == target_model:
            self._current_model_id = None
