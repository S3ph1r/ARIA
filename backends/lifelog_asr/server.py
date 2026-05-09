"""
Lifelog ASR Server — FastAPI on port 8087
Wraps Qwen3-ASR-1.7B + ForcedAligner-0.6B + pyannote community-1.
Started JIT by ARIA orchestrator on first Lifelog2 task.
"""

import logging
import time
import tempfile
import os
from contextlib import asynccontextmanager

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from asr_pipeline import ASRPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

pipeline = ASRPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.load()
    yield
    pipeline.unload()


app = FastAPI(title="Lifelog ASR", version="1.0.0", lifespan=lifespan)


class TranscribeRequest(BaseModel):
    wav_url: str
    segment_id: str
    language: str | None = None
    return_timestamps: bool = True
    return_speaker_turns: bool = True


@app.get("/health")
def health():
    import torch
    vram_gb = round(torch.cuda.memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0.0
    return {
        "status": "ok",
        "model": "qwen3-asr-1.7b",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "vram_gb": vram_gb,
    }


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    t0 = time.perf_counter()

    # Download WAV from MinIO URL
    try:
        resp = requests.get(req.wav_url, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"WAV download failed: {exc}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(resp.content)
        wav_path = f.name

    try:
        output = pipeline.run(
            wav_path=wav_path,
            language=req.language,
            return_timestamps=req.return_timestamps,
            return_speaker_turns=req.return_speaker_turns,
        )
    except Exception as exc:
        logger.error("Pipeline error for %s: %s", req.segment_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(wav_path)

    processing_time = round(time.perf_counter() - t0, 2)
    logger.info(
        "Transcribed %s in %.1fs (%d chars, %d turns)",
        req.segment_id,
        processing_time,
        len(output.get("transcript", "")),
        len(output.get("speaker_turns", [])),
    )

    return {
        "job_id": None,
        "status": "done",
        "processing_time": processing_time,
        "output": output,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8087, log_level="info")
