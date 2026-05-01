import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import logging
logger = logging.getLogger("node.telemetry")

_DDL = """
CREATE TABLE IF NOT EXISTS task_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               TEXT    NOT NULL,
    job_id           TEXT    NOT NULL,
    client_id        TEXT,
    provider         TEXT,
    model_id         TEXT,
    model_type       TEXT,
    queued_at        TEXT,
    queue_wait_s     REAL,
    processing_s     REAL,
    status           TEXT,
    error_code       TEXT,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    audio_duration_s REAL,
    rtf              REAL,
    vram_peak_gb     REAL
);
CREATE INDEX IF NOT EXISTS idx_ts    ON task_log(ts);
CREATE INDEX IF NOT EXISTS idx_model ON task_log(model_id, status);
CREATE INDEX IF NOT EXISTS idx_err   ON task_log(error_code);
"""


class TelemetryDB:
    """
    Append-only SQLite telemetry store.
    One record per completed task (both local GPU and cloud).
    Thread-safe via WAL + write lock.
    """

    def __init__(self, db_path: Path):
        self._path = db_path
        self._lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"TelemetryDB ready at {db_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._connect()
            conn.executescript(_DDL)
            conn.commit()
            conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, task, result) -> None:
        """Write one telemetry record. Silently swallows all errors."""
        try:
            self._write(task, result)
        except Exception as e:
            logger.warning(f"Telemetry write skipped: {e}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(self, task, result):
        output = result.output or {}
        metrics = output.get("metrics", {}) if isinstance(output, dict) else {}

        # Audio duration: check top-level and nested
        audio_duration_s = (
            output.get("duration_seconds")
            if isinstance(output, dict) else None
        )

        # RTF: accept both output["rtf"] (old Qwen3 format) and output["metrics"]["rtf"]
        rtf = metrics.get("rtf") or (output.get("rtf") if isinstance(output, dict) else None)
        vram_peak_gb = metrics.get("vram_peak_gb")

        # Token usage (cloud only — Gemini)
        usage = getattr(result, "usage", None) or {}
        input_tokens  = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")

        # Queue wait = elapsed since queued minus processing time
        queue_wait_s = None
        try:
            queued_ts  = datetime.fromisoformat(task.queued_at)
            complete_ts = datetime.fromisoformat(result.completed_at)
            elapsed = (complete_ts - queued_ts).total_seconds()
            queue_wait_s = max(0.0, elapsed - result.processing_time_seconds)
        except Exception:
            pass

        row = (
            datetime.now(timezone.utc).isoformat(),
            result.job_id,
            result.client_id,
            getattr(result, "provider", None),
            result.model_id,
            result.model_type,
            getattr(task, "queued_at", None),
            queue_wait_s,
            result.processing_time_seconds,
            result.status,
            getattr(result, "error_code", None),
            input_tokens,
            output_tokens,
            audio_duration_s,
            rtf,
            vram_peak_gb,
        )

        with self._lock:
            conn = self._connect()
            conn.execute(
                """INSERT INTO task_log (
                    ts, job_id, client_id, provider, model_id, model_type,
                    queued_at, queue_wait_s, processing_s, status, error_code,
                    input_tokens, output_tokens, audio_duration_s, rtf, vram_peak_gb
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                row,
            )
            conn.commit()
            conn.close()
