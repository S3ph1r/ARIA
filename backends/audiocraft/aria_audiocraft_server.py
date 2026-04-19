"""
ARIA — Audiocraft Wrapper Server v1.0
======================================
FastAPI server per AudioGen (amb/sfx) e MusicGen (sting).
Modelli caricati JIT per request, scaricati dopo ogni generazione per liberare VRAM.

Routing interno per output_style:
  amb, sfx  → AudioGen  (facebook/audiogen-medium)
  sting      → MusicGen  (facebook/musicgen-large)
"""

import os
import sys
import uuid
import asyncio
import argparse
from pathlib import Path

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ── Path resolution ───────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent          # backends/audiocraft/
ARIA_ROOT  = Path(os.environ.get("ARIA_ROOT", str(_THIS_DIR.parent.parent)))

# Punta la cache HuggingFace ai pesi locali (musicgen-large, musicgen-small già presenti)
_MODELS_DIR = ARIA_ROOT / "data" / "assets" / "models" / "audiocraft"
os.environ.setdefault("HF_HUB_CACHE", str(_MODELS_DIR))

# Override offline mode: AudioGen medium may need to download on first run
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from audiocraft.models import AudioGen, MusicGen  # noqa: E402 — dopo HF_HUB_CACHE

# ── Costanti modelli ──────────────────────────────────────────────────────────
AUDIOGEN_MODEL_ID = "facebook/audiogen-medium"
MUSICGEN_MODEL_ID = "facebook/musicgen-large"

# Sample rate di output normalizzato (Stage E lavora a 44100)
OUTPUT_SR = 44100

# ── Pydantic models ──────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    job_id:       str   = Field(default="")
    prompt:       str   = Field(..., description="Descrizione semantica del suono")
    duration:     float = Field(default=5.0,  description="Durata in secondi")
    seed:         int   = Field(default=42,   description="-1 = casuale")
    output_style: str   = Field(default="amb", description="amb | sfx | sting")


class GenerateResponse(BaseModel):
    status:           str
    job_id:           str
    audio_path:       str   = ""
    duration_seconds: float = 0.0
    error:            str   = ""


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ARIA Audiocraft Wrapper",
    version="1.0.0",
    description="AudioGen + MusicGen JIT wrapper per DIAS Sound Factory.",
)

_task_lock = asyncio.Lock()


@app.get("/health")
def health():
    return {
        "status":    "ok",
        "model":     "audiocraft-wrapper-v1",
        "models_dir": str(_MODELS_DIR),
        "ready":     True,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    async with _task_lock:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run_task, req)


# ── Core generation ───────────────────────────────────────────────────────────

def _run_task(req: GenerateRequest) -> GenerateResponse:
    job_id = req.job_id or str(uuid.uuid4())
    style  = req.output_style.lower()

    try:
        if req.seed >= 0:
            torch.manual_seed(req.seed)

        wav, native_sr = _generate(req.prompt, req.duration, style)

        # Normalizza a stereo 44100 Hz
        wav = _normalize_audio(wav, native_sr)

        # Salva WAV
        out_dir  = ARIA_ROOT / "data" / "assets" / "sound_library" / style / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{job_id}.wav"
        torchaudio.save(str(out_path), wav, OUTPUT_SR)

        duration_s = wav.shape[-1] / OUTPUT_SR

        return GenerateResponse(
            status="completed",
            job_id=job_id,
            audio_path=str(out_path),
            duration_seconds=duration_s,
        )

    except Exception as e:
        import traceback
        return GenerateResponse(status="failed", job_id=job_id, error=f"{e}\n{traceback.format_exc()}")


def _generate(prompt: str, duration: float, style: str) -> tuple:
    """Carica il modello JIT, genera, lo rimuove dalla VRAM. Ritorna (tensor [C, T], sample_rate)."""
    if style in ("amb", "sfx"):
        model = AudioGen.get_pretrained(AUDIOGEN_MODEL_ID)
    else:  # sting
        model = MusicGen.get_pretrained(MUSICGEN_MODEL_ID)

    model.set_generation_params(duration=duration)

    with torch.no_grad():
        wav = model.generate([prompt])  # [1, channels, T]

    native_sr = model.sample_rate
    audio     = wav[0].cpu()           # [channels, T]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return audio, native_sr


def _normalize_audio(audio: torch.Tensor, src_sr: int) -> torch.Tensor:
    """Porta l'audio a stereo 44100 Hz."""
    # Resample se necessario
    if src_sr != OUTPUT_SR:
        audio = torchaudio.functional.resample(audio, src_sr, OUTPUT_SR)

    # Mono → stereo
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)

    return audio


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARIA Audiocraft Wrapper Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8086)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
