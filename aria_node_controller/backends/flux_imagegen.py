"""
ARIA — FLUX.2-klein-4B Image Generation Backend Client

HTTP adapter that calls the local FastAPI server on port 8092.
Used by the orchestrator to dispatch image generation tasks from Lifelog2.

Output: JPEG saved locally by the server, served via asset server (port 8082).
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

        job_id          = payload.get("job_id", "output")
        output_filename = f"{job_id}.jpeg"

        request_body = {
            "prompt":          prompt,
            "output_filename": output_filename,
            "width":           payload.get("width",    512),
            "height":          payload.get("height",   512),
            "steps":           payload.get("steps",    20),
            "guidance":        payload.get("guidance", 3.5),
            "seed":            payload.get("seed",     -1),
        }

        timeout = payload.get("timeout_seconds", 300)
        url = f"http://127.0.0.1:{FLUX_PORT}/generate"

        logger.info(
            "FluxImageGen request — prompt='%.60s...' size=%dx%d out=%s",
            prompt, request_body["width"], request_body["height"], output_filename,
        )

        try:
            resp = requests.post(url, json=request_body, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("FluxImageGen call failed: %s", e)
            raise RuntimeError(f"FluxImageGen call failed: {e}")

        # Asset server (port 8082) serves files from ARIA_OUTPUT_DIR
        image_url = f"http://{local_ip}:8082/{output_filename}"

        return {
            "job_id":                  job_id,
            "image_url":              image_url,
            "local_path":             data.get("output_path"),
            "processing_time_seconds": data.get("processing_time"),
        }
