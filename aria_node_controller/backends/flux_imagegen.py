"""
ARIA — FLUX.2-klein-4B Image Generation Backend Client

HTTP adapter that calls the local FastAPI server on port 8092.
Used by the orchestrator to dispatch image generation tasks from Lifelog2.
"""

import logging
import requests
from pathlib import Path

logger = logging.getLogger("aria.backend.flux_imagegen")

FLUX_PORT = 8092


class FluxImageGenBackend:
    model_id   = "flux2-klein-4b"
    model_type = "imagegen"

    def is_loaded(self) -> bool:
        try:
            r = requests.get(f"http://127.0.0.1:{FLUX_PORT}/health", timeout=2)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            return False

    def run(self, payload: dict, aria_root: Path, local_ip: str) -> dict:
        prompt = payload.get("prompt", "").strip()
        if not prompt:
            raise ValueError("Campo 'prompt' obbligatorio per image gen")

        request_body = {
            "prompt":     prompt,
            "width":      payload.get("width",    512),
            "height":     payload.get("height",   512),
            "steps":      payload.get("steps",    20),
            "guidance":   payload.get("guidance", 3.5),
            "seed":       payload.get("seed",     -1),
            "output_key": payload.get("output_key"),
        }

        timeout = payload.get("timeout_seconds", 120)
        url = f"http://127.0.0.1:{FLUX_PORT}/generate"

        logger.info("FluxImageGen request — prompt='%.60s...' size=%dx%d",
                    prompt, request_body["width"], request_body["height"])

        try:
            resp = requests.post(url, json=request_body, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("output", {})
        except Exception as e:
            logger.error("FluxImageGen call failed: %s", e)
            raise RuntimeError(f"FluxImageGen call failed: {e}")
