import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
import redis

logger = logging.getLogger("node.cloud")

# Redis Keys (ARIA Standardized)
LAST_CALL_KEY        = "aria:rate_limit:google:last_call"
LOCKOUT_KEY          = "aria:rate_limit:google:lockout_until"
DAILY_COUNT_KEY_PREFIX = "aria:rate_limit:google:daily_count:"
INTERNAL_STATE_KEY   = "aria:rate_limit:internal"
RPM_WINDOW_KEY       = "aria:rate_limit:google:rpm_window"
TPM_WINDOW_KEY       = "aria:rate_limit:google:tpm_window"

# Google AI Studio resetta le quote a mezzanotte America/Los_Angeles.
# In estate: PDT = UTC-7. In inverno: PST = UTC-8.
# Usiamo UTC-7 come stima conservativa (1h di margine in inverno).
_GOOGLE_RESET_TZ = timezone(timedelta(hours=-7))


def _seconds_to_google_quota_reset() -> float:
    """Secondi al prossimo reset quota Google (mezzanotte PDT/PST)."""
    now = datetime.now(_GOOGLE_RESET_TZ)
    reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (reset - now).total_seconds()


def _format_reset_eta() -> str:
    """Stringa human-readable del prossimo reset Google."""
    now = datetime.now(_GOOGLE_RESET_TZ)
    reset_pdt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    reset_it  = reset_pdt.astimezone(timezone(timedelta(hours=2)))
    reset_utc = reset_pdt.astimezone(timezone.utc)
    delta = reset_pdt - now
    h, rem = divmod(int(delta.total_seconds()), 3600)
    m = rem // 60
    return (
        f"fra {h}h {m:02d}m "
        f"({reset_it.strftime('%H:%M')} IT / {reset_utc.strftime('%H:%M')} UTC)"
    )


class GeminiRateLimiter:
    """
    Centralized rate limiter for ARIA Gateway — Google Gemini API.

    Limiti gestiti:
    - RPD  : max richieste/giorno  (reset mezzanotte PDT)
    - RPM  : max richieste/minuto  (finestra scorrevole 60s in Redis)
    - TPM  : max token/minuto      (finestra scorrevole 60s in Redis, tracking only)
    - Pacing: delay minimo tra chiamate consecutive

    Lockout intelligente:
    - 429 da quota giornaliera  → lockout fino al reset PDT (ore, non minuti)
    - 429 da rate-limit istantaneo → lockout breve (lockout_minutes)
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        min_delay_seconds: int = 60,
        lockout_minutes: int   = 10,
        daily_limit: int       = 500,
        rpm_limit: int         = 15,
        tpm_limit: int         = 250_000,
    ):
        self.redis            = redis_client
        self.min_delay        = timedelta(seconds=min_delay_seconds)
        self.lockout_duration = timedelta(minutes=lockout_minutes)
        self.daily_limit      = daily_limit
        self.rpm_limit        = rpm_limit
        self.tpm_limit        = tpm_limit

    # ── helpers ────────────────────────────────────────────────────────────

    def _get_datetime_from_redis(self, key: str) -> Optional[datetime]:
        try:
            val = self.redis.get(key)
            if val:
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                return datetime.fromisoformat(val)
        except Exception as e:
            logger.error(f"Error reading {key} from Redis: {e}")
        return None

    def _get_daily_key(self) -> str:
        return f"{DAILY_COUNT_KEY_PREFIX}{datetime.now().strftime('%Y-%m-%d')}"

    # ── lettura contatori ───────────────────────────────────────────────────

    def get_daily_count(self) -> int:
        try:
            val = self.redis.get(self._get_daily_key())
            return int(val) if val else 0
        except Exception as e:
            logger.error(f"Error reading daily count: {e}")
            return 0

    def get_rpm_current(self) -> int:
        """Richieste negli ultimi 60s (sliding window)."""
        try:
            now_ms = int(time.time() * 1000)
            self.redis.zremrangebyscore(RPM_WINDOW_KEY, "-inf", now_ms - 60_000)
            return self.redis.zcard(RPM_WINDOW_KEY)
        except Exception:
            return 0

    def get_tpm_current(self) -> int:
        """Token negli ultimi 60s (sliding window)."""
        try:
            now_ms = int(time.time() * 1000)
            self.redis.zremrangebyscore(TPM_WINDOW_KEY, "-inf", now_ms - 60_000)
            entries = self.redis.zrange(TPM_WINDOW_KEY, 0, -1)
            total = 0
            for e in entries:
                try:
                    s = e.decode("utf-8") if isinstance(e, bytes) else str(e)
                    total += int(s.split(":")[1])
                except Exception:
                    pass
            return total
        except Exception:
            return 0

    def get_lockout_info(self) -> dict:
        """Ritorna info sul lockout corrente per la dashboard."""
        lockout_until = self._get_datetime_from_redis(LOCKOUT_KEY)
        if lockout_until:
            remaining = (lockout_until - datetime.now()).total_seconds()
            if remaining > 0:
                return {
                    "active": True,
                    "until_iso": lockout_until.isoformat(),
                    "remaining_seconds": int(remaining),
                }
        return {"active": False}

    # ── registrazione utilizzo (chiamare dopo ogni task riuscito) ──────────

    def record_usage(self, tokens: int = 0):
        """Aggiorna le sliding window RPM e TPM in Redis."""
        try:
            now_ms = int(time.time() * 1000)
            self.redis.zadd(RPM_WINDOW_KEY, {str(now_ms): now_ms})
            self.redis.expire(RPM_WINDOW_KEY, 70)
            if tokens > 0:
                self.redis.zadd(TPM_WINDOW_KEY, {f"{now_ms}:{tokens}": now_ms})
                self.redis.expire(TPM_WINDOW_KEY, 70)
        except Exception as e:
            logger.error(f"Error recording usage: {e}")

    def increment_daily_count(self) -> int:
        key = self._get_daily_key()
        try:
            count = self.redis.incr(key)
            if count == 1:
                self.redis.expire(key, 90_000)  # ~25 ore
            return count
        except Exception as e:
            logger.error(f"Error incrementing daily count: {e}")
            return 0

    # ── report errori ───────────────────────────────────────────────────────

    def report_429(self):
        """429 temporaneo (RPM / rate-limit istantaneo) → lockout breve."""
        lockout_until = datetime.now() + self.lockout_duration
        try:
            self.redis.set(LOCKOUT_KEY, lockout_until.isoformat())
            self.redis.expire(LOCKOUT_KEY, int(self.lockout_duration.total_seconds()))
            logger.warning(
                f"⚠️ GLOBAL Lockout (rate-limit) attivo per "
                f"{int(self.lockout_duration.total_seconds() // 60)} min."
            )
        except Exception as e:
            logger.error(f"Failed to set lockout: {e}")

    def report_daily_quota_exhausted(self):
        """
        429 da quota giornaliera (RPD) → lockout fino al prossimo reset PDT.
        Sostituisce il lockout fisso di 10 min con la durata reale.
        """
        seconds = _seconds_to_google_quota_reset()
        lockout_until = datetime.now() + timedelta(seconds=seconds)
        eta = _format_reset_eta()
        try:
            self.redis.set(LOCKOUT_KEY, lockout_until.isoformat())
            self.redis.expire(LOCKOUT_KEY, int(seconds) + 120)
            logger.warning(
                f"🚨 RPD Google esaurito ({self.daily_limit}/giorno). "
                f"Ripresa automatica {eta}."
            )
        except Exception as e:
            logger.error(f"Failed to set RPD lockout: {e}")

    # ── wait_for_slot ────────────────────────────────────────────────────────

    def wait_for_slot(self) -> float:
        """
        Attende finché non è disponibile uno slot di chiamata.
        Controlla in ordine: lockout globale → RPD → pacing minimo.
        Ritorna i secondi totali attesi.
        """
        total_wait_time = 0.0

        lua_script = """
        local last_call_ms = tonumber(redis.call('get', ARGV[3]) or 0)
        local lockout_until = redis.call('get', KEYS[1])
        local daily_count   = tonumber(redis.call('get', ARGV[5]) or 0)
        local daily_limit   = tonumber(ARGV[6])

        local time_res = redis.call('time')
        local now_ms = (tonumber(time_res[1]) * 1000) + math.floor(tonumber(time_res[2]) / 1000)

        if lockout_until then return -1 end
        if daily_count >= daily_limit then return -2 end

        local min_delay = tonumber(ARGV[1])
        local diff = now_ms - last_call_ms

        if diff < min_delay then
            return min_delay - diff
        end

        -- Claim slot atomicamente
        redis.call('set', ARGV[3], now_ms)
        redis.call('set', ARGV[2], ARGV[4])
        redis.call('incr', ARGV[5])
        return 0
        """

        while True:
            now_iso      = datetime.now().isoformat()
            min_delay_ms = int(self.min_delay.total_seconds() * 1000)
            daily_key    = self._get_daily_key()

            try:
                res = self.redis.eval(
                    lua_script, 1,
                    LOCKOUT_KEY,         # KEYS[1]
                    min_delay_ms,        # ARGV[1]
                    INTERNAL_STATE_KEY,  # ARGV[2]
                    LAST_CALL_KEY,       # ARGV[3]
                    now_iso,             # ARGV[4]
                    daily_key,           # ARGV[5]
                    self.daily_limit,    # ARGV[6]
                )

                if res == 0:
                    if total_wait_time > 0:
                        logger.info(f"Slot secured after {total_wait_time:.1f}s wait.")
                    return total_wait_time

                elif res == -1:
                    lockout_until = self._get_datetime_from_redis(LOCKOUT_KEY)
                    if lockout_until:
                        remaining = (lockout_until - datetime.now()).total_seconds()
                        if remaining > 0:
                            eta = _format_reset_eta()
                            logger.warning(
                                f"🔒 Lockout attivo. Ripresa {eta}. "
                                f"Attendo {remaining:.0f}s..."
                            )
                            # Sleep max 60s per iterazione: permette stop esterno e log periodici
                            sleep_s = min(remaining, 60)
                            time.sleep(sleep_s)
                            total_wait_time += sleep_s
                            continue
                    time.sleep(1)
                    total_wait_time += 1

                elif res == -2:
                    logger.error(
                        f"🚨 PREVENTIVE QUOTA PROTECTION: "
                        f"Daily limit reached ({self.daily_limit})."
                    )
                    self.report_daily_quota_exhausted()
                    raise RuntimeError("PREVENTIVE_QUOTA_EXHAUSTED")

                else:
                    wait = float(res) / 1000.0
                    logger.debug(f"Pacing: waiting {wait:.2f}s for next slot.")
                    time.sleep(wait)
                    total_wait_time += wait

            except Exception as e:
                if "PREVENTIVE_QUOTA_EXHAUSTED" in str(e):
                    raise
                logger.error(f"Error in wait_for_slot: {e}")
                time.sleep(1)
                total_wait_time += 1
