"""
Lifelog WhisperX Backend wrapper for ARIA orchestrator.
Follows the External HTTP Backend pattern (same as lifelog_asr.py).
The orchestrator's ModelProcessManager handles JIT process startup from the manifest.
This class only handles health verification and task dispatch.
"""

import logging

import requests

logger = logging.getLogger("aria.backend.lifelog_whisperx")

SERVER_URL = "http://127.0.0.1:8091"


class LifelogWhisperXBackend:
    model_id = "whisperx-large-v3"
    model_type = "stt"

    def load(self, model_path=None, config=None) -> None:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=5)
            r.raise_for_status()
            info = r.json()
            logger.info(
                "WhisperX server OK: device=%s VRAM=%.1f GB voiceprint=%s",
                info.get("device"),
                info.get("vram_gb", 0),
                info.get("voiceprint"),
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"WhisperX server non raggiungibile su {SERVER_URL}. "
                "Attendi l'avvio JIT da parte dell'orchestratore."
            )
        except Exception as exc:
            raise RuntimeError(f"Health check WhisperX fallito: {exc}")

    def unload(self) -> None:
        logger.info("WhisperX backend: unload (no-op, processo gestito dall'orchestratore)")

    def is_loaded(self) -> bool:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def estimated_vram_gb(self) -> float:
        return 12.0

    def run(self, payload: dict, **kwargs) -> dict:
        body = {
            "wav_url": payload["wav_url"],
            "segment_id": payload.get("segment_id", ""),
            "language": payload.get("language") or "it",
        }
        r = requests.post(f"{SERVER_URL}/transcribe", json=body, timeout=600)
        r.raise_for_status()
        return r.json()
