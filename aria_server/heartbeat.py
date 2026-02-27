import time
import threading
import json
import redis
from aria_server.config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD
from aria_server.logger import get_logger

logger = get_logger("aria.heartbeat")

class HeartbeatThread:
    """
    Background daemon thread that writes a presence signal to Redis every 10 seconds.
    This lets DIAS know that the ARIA Server is alive.
    If the server crashes, the EX 30 TTL will expire and the signal will disappear.
    """
    KEY = "gpu:server:heartbeat"
    INTERVAL_SECONDS = 10
    TTL_SECONDS = 30

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        logger.info("Heartbeat thread started.")

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        logger.info("Heartbeat thread stopped.")

    def _run(self):
        while not self._stop_event.is_set():
            try:
                now_ts = str(int(time.time()))
                # ONLY write the heartbeat timestamp. 
                # NEVER touch gpu:server:semaphore here.
                self.redis.set(self.KEY, now_ts, ex=self.TTL_SECONDS)
            except Exception as e:
                logger.error(f"Failed to write heartbeat: {e}")
            
            # Sleep in small chunks so we can exit quickly if stopped
            for _ in range(self.INTERVAL_SECONDS * 10):
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)
