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
        min_delay_seconds: int = 60, 
        lockout_minutes: int = 10, 
        daily_limit: int = 200
    ):
        self.redis = redis_client
        self.min_delay = timedelta(seconds=min_delay_seconds)
        self.lockout_duration = timedelta(minutes=lockout_minutes)
        self.daily_limit = daily_limit


    def _get_datetime_from_redis(self, key: str) -> Optional[datetime]:
        try:
            val = self.redis.get(key)
            if val:
                # Handle both bytes and str
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                return datetime.fromisoformat(val)
        except Exception as e:
            logger.error(f"Error reading {key} from Redis: {e}")
        return None

    def _get_daily_key(self) -> str:
        return f"{DAILY_COUNT_KEY_PREFIX}{datetime.now().strftime('%Y-%m-%d')}"

    def get_daily_count(self) -> int:
        key = self._get_daily_key()
        try:
            val = self.redis.get(key)
            return int(val) if val else 0
        except Exception as e:
            logger.error(f"Error reading daily count: {e}")
            return 0

    def increment_daily_count(self) -> int:
        # NOTE: This is now mostly handled by the Lua script in wait_for_slot for atomicity,
        # but kept here for manual use or auxiliary incrementing.
        key = self._get_daily_key()
        try:
            count = self.redis.incr(key)
            if count == 1:
                self.redis.expire(key, 90000) # ~25 hours
            return count
        except Exception as e:
            logger.error(f"Error incrementing daily count: {e}")
            return 0

    def report_429(self):
        """Signals a 429 error and activates a global lockout."""
        lockout_until = datetime.now() + self.lockout_duration
        try:
            self.redis.set(LOCKOUT_KEY, lockout_until.isoformat())
            self.redis.expire(LOCKOUT_KEY, int(self.lockout_duration.total_seconds()))
            logger.warning(f"⚠️ GLOBAL Lockout activated until {lockout_until.isoformat()}")
        except Exception as e:
            logger.error(f"Failed to set global lockout: {e}")

    def wait_for_slot(self) -> float:
        """
        Wait until a slot is available GLOBALLY.
        Returns total seconds waited.
        """
        total_wait_time = 0.0
        
        # IMPROVED Lua script: Atomic pacing + quota check
        lua_script = """
        local last_call_ms = tonumber(redis.call('get', ARGV[3]) or 0)
        local lockout_until = redis.call('get', KEYS[1])
        local daily_count = tonumber(redis.call('get', ARGV[5]) or 0)
        local daily_limit = tonumber(ARGV[6])
        
        local time_res = redis.call('time')
        local now_ms = (tonumber(time_res[1]) * 1000) + math.floor(tonumber(time_res[2]) / 1000)
        
        if lockout_until then return -1 end
        if daily_count >= daily_limit then return -2 end
        
        local min_delay = tonumber(ARGV[1])
        local diff = now_ms - last_call_ms
        
        if diff < min_delay then
            return min_delay - diff
        end
        
        -- Claim slot and increment quota ATOMICALLY
        redis.call('set', ARGV[3], now_ms)
        redis.call('set', ARGV[2], ARGV[4])
        redis.call('incr', ARGV[5])
        
        -- Success
        return 0
        """
        
        while True:
            now_iso = datetime.now().isoformat()
            min_delay_ms = int(self.min_delay.total_seconds() * 1000)
            daily_key = self._get_daily_key()
            
            try:
                # res will be: 0 (success), -1 (lockout), -2 (quota hit), or > 0 (wait ms)
                res = self.redis.eval(
                    lua_script, 1, 
                    LOCKOUT_KEY,             # KEYS[1]
                    min_delay_ms,            # ARGV[1]
                    INTERNAL_STATE_KEY,      # ARGV[2]
                    LAST_CALL_KEY,           # ARGV[3]
                    now_iso,                 # ARGV[4]
                    daily_key,               # ARGV[5]
                    self.daily_limit         # ARGV[6]
                )
                
                if res == 0:
                    # Success: slot obtained and quota incremented in Lua
                    if total_wait_time > 0:
                        logger.info(f"Slot secured after {total_wait_time:.2f}s delay.")
                    return total_wait_time
                
                elif res == -1:
                    lockout_until = self._get_datetime_from_redis(LOCKOUT_KEY)
                    if lockout_until:
                        wait = (lockout_until - datetime.now()).total_seconds()
                        if wait > 0:
                            logger.warning(f"🚫 GLOBAL Lockout active. Waiting {wait:.1f}s...")
                            time.sleep(wait)
                            total_wait_time += wait
                        else:
                            time.sleep(1)
                            total_wait_time += 1
                    else:
                        time.sleep(1)
                        total_wait_time += 1
                
                elif res == -2:
                    logger.error(f"🚨 PREVENTIVE QUOTA PROTECTION: Daily limit reached ({self.daily_limit}).")
                    # Activate lockout to prevent further attempts from other workers
                    self.report_429()
                    # Raising RuntimeError to stop CloudManager execution for this task
                    raise RuntimeError("PREVENTIVE_QUOTA_EXHAUSTED")
                
                else:
                    # Specific delay requested by Lua (ms)
                    wait = float(res) / 1000.0
                    logger.debug(f"Pacing: waiting {wait:.2f}s for next slot.")
                    time.sleep(wait)
                    total_wait_time += wait
                    
            except Exception as e:
                # Re-raise our specific error to propagate up to CloudManager
                if "PREVENTIVE_QUOTA_EXHAUSTED" in str(e):
                    raise
                logger.error(f"Error in wait_for_slot: {e}")
                time.sleep(1)
                total_wait_time += 1
