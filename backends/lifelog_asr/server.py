"""
Lifelog ASR Server — FastAPI on port 8087
Wraps Qwen3-ASR-1.7B + ForcedAligner-0.6B + pyannote community-1.
Started JIT by ARIA orchestrator on first Lifelog2 task.
"""

import os
import sys
import logging

# CONFIGURAZIONE AMBIENTE HF - DEVE ESSERE PRIMA DI OGNI ALTRO IMPORT
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HOME"] = r"C:\Users\Roberto\aria\data\assets\models\huggingface"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import torchaudio
import torch
import soundfile as sf
from types import ModuleType
from dotenv import load_dotenv
from huggingface_hub import login

# Carica variabili d'ambiente da .env (per HF_TOKEN e credenziali MinIO)
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    try:
        login(token=hf_token)
    except Exception as e:
        logging.warning("HF Login failed (offline?): %s", e)

# MONKEYPATCH TORCHAUDIO PER COMPATIBILITÀ CON SPEECHBRAIN (usato internamente da pyannote.audio)
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]

if not hasattr(torchaudio, "io"):
    io_mock = ModuleType("torchaudio.io")
    io_mock.StreamReader = object
    torchaudio.io = io_mock
    sys.modules["torchaudio.io"] = io_mock

import time
import tempfile
import re
from urllib.parse import urlparse
from contextlib import asynccontextmanager

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from minio import Minio

try:
    from .asr_pipeline import ASRPipeline
except ImportError:
    from asr_pipeline import ASRPipeline

# Configurazione Logging (Console + File) - Forziamo la scrittura per bypassare uvicorn
LOG_FILE = r"C:\Users\Roberto\aria\logs\lifelog_asr.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Pulizia handler esistenti per evitare duplicati o blocchi
logging.root.handlers = []

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding='utf-8')
    ],
    force=True # Disponibile da Python 3.8+, forza questa config
)
logger = logging.getLogger(__name__)
logger.info("Logging inizializzato correttamente (File: %s)", LOG_FILE)

pipeline = ASRPipeline()

# MinIO Client Configuration from Environment (Injected by ARIA Orchestrator)
MINIO_ENDPOINT = os.getenv("ARIA_MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("ARIA_MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("ARIA_MINIO_SECRET_KEY", "minioadmin")

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False # NH-Mini uses HTTP for internal LAN
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up ASR Pipeline...")
    pipeline.load()
    yield
    logger.info("Shutting down ASR Pipeline...")
    pipeline.unload()

app = FastAPI(title="Lifelog ASR", version="1.0.0", lifespan=lifespan)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.error(f"Validation Error: {exc.errors()} | Body: {body.decode()}")
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors(), "body": body.decode()},
    )

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

def download_file(url: str, dest_path: str):
    """Downloads a file from URL, using MinIO client if it's a MinIO URL."""
    parsed = urlparse(url)
    
    # Check if it's a MinIO URL (internal IP)
    if parsed.netloc == MINIO_ENDPOINT or "9000" in parsed.netloc:
        logger.info(f"MinIO URL detected: {url}. Using authenticated client.")
        # Path format: /bucket/object/path...
        path_parts = parsed.path.lstrip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError(f"Invalid MinIO URL path: {parsed.path}")
        
        bucket_name = path_parts[0]
        object_name = "/".join(path_parts[1:])
        
        minio_client.fget_object(bucket_name, object_name, dest_path)
    else:
        # Standard HTTP download
        logger.info(f"Standard URL detected: {url}. Using requests.")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(dest_path, 'wb') as f:
            f.write(resp.content)

@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    t0 = time.perf_counter()
    logger.info(f"Received transcription request for segment: {req.segment_id}")

    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_path = tmp_wav.name
    tmp_wav.close()

    try:
        download_file(req.wav_url, wav_path)
        logger.info(f"WAV downloaded and saved at: {wav_path}")

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
        if os.path.exists(wav_path):
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
    uvicorn.run(app, host="0.0.0.0", port=8087, log_level="info")
