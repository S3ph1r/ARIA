"""
ARIA — Audiocraft Backend Connector
=====================================
Connettore per il wrapper AudioGen/MusicGen (porta 8086).
Interfaccia identica ad ACEStepBackend ma senza HTDemucs e senza relay.
"""

import logging
import uuid
import requests
from pathlib import Path

logger = logging.getLogger("node.backend.audiocraft")

AUDIOCRAFT_WRAPPER_URL = "http://127.0.0.1:8086"


class AudiocraftBackend:
    model_id   = "audiocraft-medium"
    model_type = "mus"

    def load(self, model_path: str = "", _config: dict = None) -> None:
        try:
            r = requests.get(f"{AUDIOCRAFT_WRAPPER_URL}/health", timeout=5)
            r.raise_for_status()
            logger.info("Audiocraft Wrapper OK")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Audiocraft wrapper non raggiungibile su {AUDIOCRAFT_WRAPPER_URL}. "
                "Avviare il processo JIT tramite l'Orchestratore."
            )
        except Exception as e:
            raise RuntimeError(f"Health check Audiocraft fallito: {e}")

    def unload(self) -> None:
        logger.info("AudiocraftBackend: unload (no-op, processo esterno)")

    def is_loaded(self) -> bool:
        try:
            r = requests.get(f"{AUDIOCRAFT_WRAPPER_URL}/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def run(self, payload: dict, aria_root: Path, local_ip: str) -> dict:
        """
        Invia un task al wrapper Audiocraft e aspetta il risultato.

        Parametri attesi in payload:
          - prompt        : descrizione semantica (obbligatorio)
          - duration      : durata in secondi (default 5.0)
          - seed          : seed (default 42, -1 = casuale)
          - output_style  : 'amb' | 'sfx' | 'sting' (default 'amb')
        """
        prompt = payload.get("prompt") or payload.get("text", "")
        if not prompt:
            raise ValueError("Campo 'prompt' obbligatorio per Audiocraft")

        job_id  = payload.get("job_id") or str(uuid.uuid4())
        style   = payload.get("output_style", "amb")
        timeout = payload.get("timeout_seconds", 300)

        request_body = {
            "job_id":       job_id,
            "prompt":       prompt,
            "duration":     float(payload.get("duration", 5.0)),
            "seed":         int(payload.get("seed", 42)),
            "output_style": style,
        }

        logger.info(
            f"[Audiocraft] Submit | job={job_id} | style={style} | "
            f"duration={request_body['duration']}s | seed={request_body['seed']}"
        )

        try:
            response = requests.post(
                f"{AUDIOCRAFT_WRAPPER_URL}/generate",
                json=request_body,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Timeout ({timeout}s) generazione Audiocraft — job={job_id}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Errore chiamata Audiocraft wrapper: {e}")

        if result.get("status") != "completed":
            err = result.get("error", "Errore sconosciuto")
            raise RuntimeError(f"Audiocraft generation failed: {err}")

        audio_path = result.get("audio_path", "")

        def _to_url(abs_path: str) -> str:
            try:
                rel = Path(abs_path).relative_to(aria_root / "data")
                return f"http://{local_ip}:8082/{rel.as_posix()}"
            except ValueError:
                return f"http://{local_ip}:8082/assets/sound_library/{style}/{job_id}/{Path(abs_path).name}"

        audio_url = _to_url(audio_path)

        logger.info(f"[Audiocraft] OK | job={job_id} | dur={result.get('duration_seconds', 0):.1f}s")

        return {
            "audio_url":        audio_url,
            "local_path":       audio_path,
            "output_style":     style,
            "duration_seconds": result.get("duration_seconds", request_body["duration"]),
            "status":           "success",
        }
