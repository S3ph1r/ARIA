import json
import redis
from typing import Optional, Tuple
from pydantic import ValidationError
import time

from .logger import get_logger
from .models import AriaTaskPayload, AriaTaskResult
from .config_manager import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD, REDIS_TIMEOUT

logger = get_logger("node.queue")

class AriaQueueManager:
    """
    Handles fetching tasks from Redis queues and posting results,
    plus crash recovery and dead letter routing.
    """
    
    # Internal prefixes
    PREFIX_PROCESSING = "gpu:processing"
    PREFIX_DEAD = "gpu:dead"
    MAX_RETRIES = 3

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.recover_crashed_tasks()

    def recover_crashed_tasks(self):
        """
        Runs on startup. Finds any tasks stuck in processing 
        (meaning ARIA crashed during them) and either requeues them
        or sends them to dead letter if retries exceeded.
        """
        count_requeued = 0
        count_dead = 0
        
        for key in self.redis.scan_iter(match=f"{self.PREFIX_PROCESSING}:*"):
            task_json = self.redis.get(key)
            if not task_json:
                continue
                
            try:
                task_data = json.loads(task_json)
                task_data["retry_count"] = task_data.get("retry_count", 0) + 1
                
                job_id = task_data.get("job_id")
                client_id = task_data.get("client_id")
                model_type = task_data.get("model_type")
                model_id = task_data.get("model_id")
                queue_key = f"gpu:queue:{model_type}:{model_id}"
                
                # Circuit breaker check
                if task_data["retry_count"] >= self.MAX_RETRIES:
                    dead_key = f"{self.PREFIX_DEAD}:{client_id}:{job_id}"
                    self.redis.lpush(dead_key, json.dumps(task_data))
                    logger.warning(f"Task {job_id} failed {self.MAX_RETRIES} times. Moved to dead letter: {dead_key}")
                    count_dead += 1
                else:
                    logger.info(f"Re-queueing crashed task {job_id} to {queue_key} (retry {task_data['retry_count']})")
                    self.redis.lpush(queue_key, json.dumps(task_data))
                    count_requeued += 1
                    
            except Exception as e:
                logger.error(f"Failed to parse and recover processing key {key}: {e}")
            finally:
                self.redis.delete(key) # Clear the processing lock
                
        if count_requeued > 0 or count_dead > 0:
            logger.info(f"Crash recovery complete: {count_requeued} requeued, {count_dead} dead lettered.")

    def fetch_task(self, queue_key: str, timeout: int = 5) -> Tuple[Optional[str], Optional[AriaTaskPayload]]:
        """
        Pop a task from a specific queue and put it in processing state.
        Returns (raw_json_string, AriaTaskPayload) or (None, None).
        """
        result = self.redis.brpop(queue_key, timeout=timeout)
        if not result:
            return None, None
            
        _, raw_json_bytes = result
        raw_json_str = raw_json_bytes.decode('utf-8')
        
        try:
            task = AriaTaskPayload.model_validate_json(raw_json_str)
            # Create visibility lock BEFORE returning
            self._lock_processing(task.job_id, raw_json_str, task.timeout_seconds)
            return raw_json_str, task
            
        except ValidationError as e:
            logger.error(f"Invalid task payload received from {queue_key}: {e}")
            # Immediately send invalid tasks to dead letter as they'll never succeed
            try:
                task_data = json.loads(raw_json_str)
                client_id = task_data.get("client_id", "unknown")
                job_id = task_data.get("job_id", "unknown")
            except:
                client_id, job_id = "unknown", "unknown"
                
            dead_key = f"{self.PREFIX_DEAD}:{client_id}:{job_id}"
            self.redis.lpush(dead_key, raw_json_str)
            return None, None

    def _lock_processing(self, job_id: str, raw_json: str, ttl: int):
        proc_key = f"{self.PREFIX_PROCESSING}:{job_id}"
        self.redis.set(proc_key, raw_json, nx=True, ex=ttl + 30) # Add a 30s buffer

    def unlock_processing(self, job_id: str):
        proc_key = f"{self.PREFIX_PROCESSING}:{job_id}"
        self.redis.delete(proc_key)

    def post_result(self, task: AriaTaskPayload, result: AriaTaskResult) -> None:
        """
        Sends the result back to the callback_key specified in the task payload.
        Sets an expiration on the key so it doesn't linger forever if unread.
        """
        try:
            callback_key = task.callback_key
            result_json = result.model_dump_json()
            
            # Use a pipeline to ensure LPUSH and EXPIRE happen together
            pipeline = self.redis.pipeline()
            pipeline.lpush(callback_key, result_json)
            pipeline.expire(callback_key, 86400) # Keep result available for 24h max
            pipeline.execute()
            
            logger.info(f"Result for {task.job_id} posted to {callback_key}")
        except Exception as e:
            logger.error(f"Failed to post result for {task.job_id}: {e}")
        finally:
            self.unlock_processing(task.job_id)
