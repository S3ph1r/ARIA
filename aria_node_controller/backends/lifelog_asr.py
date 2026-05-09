"""
Lifelog ASR Backend wrapper for ARIA orchestrator.
Follows the External HTTP Backend pattern (like qwen3_tts.py).
The orchestrator's ModelProcessManager handles process startup from the manifest.
This class only handles health verification and task dispatch.
"""

import logging

import requests

logger = logging.getLogger("aria.backend.lifelog_asr")

SERVER_URL = "http://127.0.0.1:8087"


class LifelogASRBackend:
    model_id = "qwen3-asr-1.7b"
    model_type = "stt"

    def load(self, model_path=None, config=None) -> None:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=5)
            r.raise_for_status()
            info = r.json()
            logger.info(
                "Lifelog ASR server OK: device=%s VRAM=%.1f GB",
                info.get("device"),
                info.get("vram_gb", 0),
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Lifelog ASR server non raggiungibile su {SERVER_URL}. "
                "Attendi l'avvio JIT da parte dell'orchestratore."
            )
        except Exception as exc:
            raise RuntimeError(f"Health check Lifelog ASR fallito: {exc}")

    def unload(self) -> None:
        logger.info("Lifelog ASR backend: unload (no-op, processo gestito dall'orchestratore)")

    def is_loaded(self) -> bool:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def estimated_vram_gb(self) -> float:
        return 9.0

    def run(self, payload: dict, **kwargs) -> dict:
        body = {
            "wav_url": payload["wav_url"],
            "segment_id": payload.get("segment_id", ""),
            "language": payload.get("language"),
            "return_timestamps": payload.get("return_timestamps", True),
            "return_speaker_turns": payload.get("return_speaker_turns", True),
        }
        r = requests.post(f"{SERVER_URL}/transcribe", json=body, timeout=360)
        r.raise_for_status()
        return r.json()
