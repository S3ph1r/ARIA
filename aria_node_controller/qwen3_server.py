"""
ARIA — Qwen3-TTS Server Standalone (QW-1)
==========================================
FastAPI server che espone Qwen3-TTS-12Hz-1.7B-Base su porta 8083.
Gira nell'ambiente conda `qwen3-tts` (Python 3.12).

Avvio:
    conda activate qwen3-tts
    cd C:\\Users\\Roberto\\aria
    python qwen3_server.py

Variabili ambiente:
    QWEN3_MODEL_PATH   : path modello (default: C:/models/qwen3-tts-1.7b)
    ARIA_OUTPUT_DIR    : directory output WAV (default: C:/Users/Roberto/aria/data/outputs)
    QWEN3_HOST         : bind address (default: 0.0.0.0)
    QWEN3_PORT         : porta (default: 8083)
"""

import os
import re
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModel

# ──────────────────────────────────────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("qwen3-tts-server")

MODEL_PATH   = os.getenv("QWEN3_MODEL_PATH",  r"C:/Users/Roberto/aria/data/models/qwen3-tts-1.7b")
OUTPUT_DIR   = Path(os.getenv("ARIA_OUTPUT_DIR", r"C:/Users/Roberto/aria/data/outputs"))

HOST         = os.getenv("QWEN3_HOST",  "0.0.0.0")
PORT         = int(os.getenv("QWEN3_PORT", "8083"))
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Stato globale
# ──────────────────────────────────────────────────────────────────────────────
model = None
app   = FastAPI(title="Qwen3-TTS Server", version="1.0.0",
                description="ARIA Qwen3-TTS 1.7B — porta 8083")


# ──────────────────────────────────────────────────────────────────────────────
# Caricamento modello
# ──────────────────────────────────────────────────────────────────────────────
def load_model():
    global model
    logger.info(f"Caricamento Qwen3-TTS 1.7B su {DEVICE} (path: {MODEL_PATH})...")
    t0 = time.time()

    common_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    )

    try:
        model = AutoModel.from_pretrained(
            MODEL_PATH, attn_implementation="flash_attention_2", **common_kwargs
        )
        attn_mode = "flash_attention_2"
    except Exception as e:
        logger.warning(f"Flash attention non disponibile ({type(e).__name__}), caricamento standard.")
        model = AutoModel.from_pretrained(MODEL_PATH, **common_kwargs)
        attn_mode = "standard"

    model.eval()
    elapsed = time.time() - t0
    vram_gb = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
    logger.info(f"Modello caricato in {elapsed:.1f}s | VRAM: {vram_gb:.2f} GB | Attn: {attn_mode}")


# ──────────────────────────────────────────────────────────────────────────────
# Utilità audio
# ──────────────────────────────────────────────────────────────────────────────
def chunk_text(text: str, max_words: int = 250) -> list[str]:
    """Splitta su confini di frase, mai a metà frase."""
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
    chunks, chunk, chunk_words = [], [], 0

    for sent in sentences:
        w = len(sent.split())
        if chunk_words + w > max_words and chunk:
            chunks.append(" ".join(chunk))
            chunk, chunk_words = [sent], w
        else:
            chunk.append(sent)
            chunk_words += w

    if chunk:
        chunks.append(" ".join(chunk))
    return chunks or [text]


def concatenate_wavs(wav_list: list[np.ndarray], sr: int, gap_ms: int = 80) -> np.ndarray:
    """Unisce i chunk con un piccolo silenzio (respiro naturale)."""
    gap = np.zeros(int(sr * gap_ms / 1000), dtype=wav_list[0].dtype)
    out = []
    for i, wav in enumerate(wav_list):
        out.append(wav)
        if i < len(wav_list) - 1:
            out.append(gap)
    return np.concatenate(out)


# ──────────────────────────────────────────────────────────────────────────────
# Schema request
# ──────────────────────────────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    voice_ref_audio_path: str
    voice_ref_text: Optional[str] = None
    language: str = "Italian"
    instruct: str = "Warm Italian male voice, professional audiobook narrator, calm and measured."

    # Parametri generazione
    non_streaming_mode: bool = True
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    output_sample_rate: int = 24000

    # Chunking
    max_words_per_chunk: int = 250
    gap_between_chunks_ms: int = 80

    # Output
    output_filename: str = "output.wav"


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Restituisce lo stato del server e VRAM attuale."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modello non ancora caricato")
    vram_gb = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
    return {
        "status": "ok",
        "model": "Qwen3-TTS-12Hz-1.7B-Base",
        "device": DEVICE,
        "vram_allocated_gb": round(vram_gb, 2),
        "output_dir": str(OUTPUT_DIR),
    }


@app.post("/tts")
def synthesize(req: TTSRequest):
    """Sintetizza testo con voice cloning e chunking automatico."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modello non caricato")

    t_start = time.time()

    # Carica audio di riferimento
    ref_path = Path(req.voice_ref_audio_path)
    if not ref_path.exists():
        raise HTTPException(status_code=400, detail=f"File ref non trovato: {req.voice_ref_audio_path}")

    try:
        ref_audio, ref_sr = sf.read(str(ref_path))
        if ref_audio.ndim > 1:
            ref_audio = ref_audio[:, 0]  # forzatura mono
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore lettura ref audio: {e}")

    ref_duration = len(ref_audio) / ref_sr
    if ref_duration > 30:
        logger.warning(f"Sample di riferimento lungo {ref_duration:.1f}s — rischio loop (max 30s).")

    # Chunking
    chunks = chunk_text(req.text, req.max_words_per_chunk)
    logger.info(f"Testo suddiviso in {len(chunks)} chunk | ref: {ref_path.name} ({ref_duration:.1f}s)")
    torch.cuda.reset_peak_memory_stats()

    wav_chunks = []
    output_sr = req.output_sample_rate

    for i, chunk_text_part in enumerate(chunks):
        n_words = len(chunk_text_part.split())
        logger.info(f"  Chunk {i+1}/{len(chunks)}: {n_words} parole")
        try:
            wavs, sr = model.generate_base_voice(
                text=chunk_text_part,
                ref_audio=ref_audio,
                ref_sr=ref_sr,
                ref_text=req.voice_ref_text,   # None è ok (Qwen3 lo gestisce)
                language=req.language,
                instruct=req.instruct,
                non_streaming_mode=req.non_streaming_mode,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
            )
            output_sr = sr
            wav_chunks.append(wavs[0] if isinstance(wavs, list) else wavs)
        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM su chunk {i+1} — chunk troppo lungo o VRAM insufficiente")
            raise HTTPException(status_code=500,
                                detail=f"CUDA OOM su chunk {i+1}/{len(chunks)}. Riduci max_words_per_chunk.")
        except Exception as e:
            logger.error(f"Errore chunk {i+1}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Errore inferenza chunk {i+1}: {e}")

    # Concatenazione
    final_wav = concatenate_wavs(wav_chunks, output_sr, req.gap_between_chunks_ms)

    # Salvataggio
    out_path = OUTPUT_DIR / req.output_filename
    sf.write(str(out_path), final_wav, output_sr)

    inference_time = time.time() - t_start
    duration = len(final_wav) / output_sr
    rtf = inference_time / duration if duration > 0 else 0
    vram_peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0

    logger.info(
        f"Completato: {duration:.1f}s audio | "
        f"inferenza: {inference_time:.1f}s | RTF: {rtf:.2f}x | "
        f"VRAM peak: {vram_peak:.2f} GB"
    )

    return {
        "status": "ok",
        "output_path": str(out_path),
        "output_filename": req.output_filename,
        "duration_seconds": round(duration, 2),
        "sample_rate": output_sr,
        "chunks_count": len(chunks),
        "inference_time_seconds": round(inference_time, 2),
        "rtf": round(rtf, 2),
        "vram_peak_gb": round(vram_peak, 2),
    }


@app.get("/outputs/{filename}")
def get_output(filename: str):
    """Serve un file WAV generato (usato dal worker DIAS per pull diretto)."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File non trovato: {filename}")
    return FileResponse(str(path), media_type="audio/wav")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    logger.info(f"Avvio server su {HOST}:{PORT}...")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
