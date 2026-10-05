"""
Lifelog Voices Backend wrapper for ARIA orchestrator (2026-10-05).
External HTTP Backend pattern (come lifelog_whisperx.py): il processo lo avvia JIT il ModelProcessManager
dal manifest; questa classe verifica l'health e inoltra il task.
Server: backends/lifelog_voices/server.py (DiariZen + WavLM anti-replay), porta 8094.
"""

import logging

import requests

logger = logging.getLogger("aria.backend.lifelog_voices")

PORT = 8094


class LifelogVoicesBackend:
    model_id = "lifelog-voices"
    model_type = "voices"

    def load(self, model_path=None, config=None) -> None:
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=5)
            r.raise_for_status()
            logger.info("Lifelog Voices server OK: %s", r.json())
        except Exception as exc:
            raise RuntimeError(f"Lifelog Voices server non raggiungibile sulla porta {PORT}: {exc}")

    def unload(self) -> None:
        logger.info("Lifelog Voices backend: unload (no-op, processo gestito dall'orchestratore)")

    def is_loaded(self) -> bool:
        try:
            return requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200
        except Exception:
            return False

    def estimated_vram_gb(self) -> float:
        return 11.0

    def run(self, payload: dict, **kwargs) -> dict:
        local_ip = kwargs.get("local_ip", "127.0.0.1")
        body = {"wav_url": payload["wav_url"], "segment_id": payload.get("segment_id", "")}
        for opt in ("replay", "min_turn_s"):
            if payload.get(opt) is not None:
                body[opt] = payload[opt]
        r = requests.post(f"http://{local_ip}:{PORT}/analyze", json=body, timeout=600)
        r.raise_for_status()
        return r.json()
