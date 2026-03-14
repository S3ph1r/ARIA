import time
import logging
from datetime import datetime, timedelta
from typing import Optional
import redis

logger = logging.getLogger("node.cloud")

# Redis Keys (ARIA Standardized)
LAST_CALL_KEY = "aria:rate_limit:google:last_call"
LOCKOUT_KEY = "aria:rate_limit:google:lockout_until"
DAILY_COUNT_KEY_PREFIX = "aria:rate_limit:google:daily_count:"
INTERNAL_STATE_KEY = "aria:rate_limit:internal"

class GeminiRateLimiter:
    """
    Centralized rate limiter for ARIA Gateway to handle Google Gemini API limits.
    
    Features:
    - Global pacing (minimum delay between requests).
    - Global lockout on 429 (Quota Exhausted).
    - Daily request tracking.
    """
    
    def __init__(
        self, 
        redis_client: redis.Redis,
        min_delay_seconds: int = 30,
        max_daily_requests: int = 1500
    ):
        self.r = redis_client
        self.min_delay = min_delay_seconds
        self.max_daily = max_daily_requests
        
        # Load Lua script for atomic pacing
        self._pacing_script = self.r.register_script("""
            local last_call_key = KEYS[1]
            local lockout_until_key = KEYS[2]
            local min_delay = tonumber(ARGV[1])
            local now = tonumber(ARGV[2])
            
            -- Check for global lockout
            local lockout_until = tonumber(redis.call('GET', lockout_until_key) or 0)
            if now < lockout_until then
                return "LOCKOUT:" .. tostring(lockout_until - now)
            end
            
            -- Check for pacing
            local last_call = tonumber(redis.call('GET', last_call_key) or 0)
            local diff = now - last_call
            
            if diff < min_delay then
                return tostring(math.floor((min_delay - diff) * 1000)) -- Return ms to wait
            end
            
            -- Success: update last call
            redis.call('SET', last_call_key, now)
            return "OK"
        """)

    def wait_for_slot(self, timeout_seconds: int = 600) -> bool:
        """
        Blocks until a rate-limit slot is available or timeout.
        """
        start_time = time.time()
        total_wait_time = 0
        
        while (time.time() - start_time) < timeout_seconds:
            try:
                now = time.time()
                res = self._pacing_script(keys=[LAST_CALL_KEY, LOCKOUT_KEY], args=[self.min_delay, now])
                
                if res == b"OK":
                    if total_wait_time > 0:
                        logger.info(f"Slot acquired after {total_wait_time:.1f}s.")
                    return True
                
                res = res.decode('utf-8')
                if res.startswith("LOCKOUT:"):
                    wait = float(res.split(":")[1])
                    if wait > 0:
                        logger.warning(f"GLOBAL 429 Lockout active. Waiting {wait:.1f}s...")
                        time.sleep(min(wait, 5.0)) # Don't sleep too long to allow orchestration checks
                        total_wait_time += min(wait, 5.0)
                    else:
                        time.sleep(1)
                        total_wait_time += 1
                
                else:
                    # Specific delay requested by Lua (ms)
                    wait = float(res) / 1000.0
                    logger.debug(f"Pacing: waiting {wait:.2f}s for next slot.")
                    time.sleep(wait)
                    total_wait_time += wait
                    
            except Exception as e:
                logger.error(f"Error in wait_for_slot: {e}")
                time.sleep(1)
                total_wait_time += 1
                
        return False

    def report_success(self):
        """Called by worker to increment daily count if needed."""
        today = datetime.now().strftime("%Y-%m-%d")
        key = f"{DAILY_COUNT_KEY_PREFIX}{today}"
        self.r.incr(key)
        self.r.expire(key, 86400 * 2) # Keep for 2 days

    def report_429(self, retry_after_seconds: int = 60):
        """Handles Quota Exhausted by setting a global lockout."""
        until = time.time() + retry_after_seconds
        self.r.set(LOCKOUT_KEY, until)
        logger.error(f"Quota Exhausted (429). Setting Global Lockout for {retry_after_seconds}s.")
