import redis
from typing import Dict, Optional, Tuple

from aria_server.logger import get_logger

logger = get_logger("aria.batch")

class BatchOptimizer:
    """
    Decides which Redis queue to drain next based on VRAM state and queue depth.
    Goal: Minimize expensive VRAM load/unload cycles.
    """
    
    @staticmethod
    def build_queue_key(model_type: str, model_id: str) -> str:
        """
        Creates the standardized canonical Redis queue key name for a given model.
        Correct schema: gpu:queue:{model_type}:{model_id}
        """
        return f"gpu:queue:{model_type}:{model_id}"

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        
    def get_queue_depths(self, known_models: Dict[str, str]) -> Dict[str, int]:
        """
        Polls Redis for the depth of all known queues.
        known_models format: {"tts:orpheus-3b": "gpu:queue:tts:orpheus-3b", ...}
        """
        depths = {}
        for logic_id, queue_key in known_models.items():
            try:
                length = self.redis.llen(queue_key)
                if length > 0:
                    depths[logic_id] = length
            except Exception as e:
                logger.error(f"Error reading queue depth for {queue_key}: {e}")
        return depths

    def decide_next_queue(self, 
                         known_models: Dict[str, str], 
                         current_model_logic_id: Optional[str]) -> Optional[Tuple[str, str]]:
        """
        Core routing logic. Returns (logic_id, queue_key) of the next queue to drain.
        Returns None if all queues are empty.
        
        Rule 1: If the currently loaded model has tasks, STAY on it.
        Rule 2: Otherwise, pick the queue with the most tasks to process a batch.
        """
        depths = self.get_queue_depths(known_models)
        
        if not depths:
            return None
            
        # Rule 1: Drain current queue first
        if current_model_logic_id and current_model_logic_id in depths:
            logger.debug(f"BatchOptimizer: Staying on {current_model_logic_id} ({depths[current_model_logic_id]} tasks)")
            return current_model_logic_id, known_models[current_model_logic_id]
            
        # Rule 2: Pick the busiest queue
        busiest_logic_id = max(depths, key=depths.get)
        logger.info(f"BatchOptimizer: Switching to {busiest_logic_id} ({depths[busiest_logic_id]} tasks)")
        
        return busiest_logic_id, known_models[busiest_logic_id]
