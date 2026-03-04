import os
import io
import re
import time
import logging
import subprocess
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from qwen_tts import Qwen3TTSModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen3-tts-server")

MODEL_PATH = os.getenv("QWEN3_MODEL_PATH", "C:/Users/Roberto/aria/data/models/qwen3-tts-1.7b")
OUTPUT_DIR = os.getenv("ARIA_OUTPUT_DIR", "C:/Users/Roberto/aria/data/outputs")
HOST = os.getenv("QWEN3_HOST", "0.0.0.0")
PORT = int(os.getenv("QWEN3_PORT", "8083"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Qwen3-TTS Server", version="1.1.0")
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    global model
    logger.info(f"Caricamento Qwen3-TTS 1.7B su {device}...")
    try:
        model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
        logger.info(f"Modello caricato (FlashAttention). VRAM: {vram:.2f} GB")
    except Exception as e:
        logger.warning(f"Flash attention non disponibile ({e}), caricamento standard.")
        model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map=device,
        )
        vram = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
        logger.info(f"Modello caricato (Standard). VRAM: {vram:.2f} GB")


def chunk_text(text: str, max_words: int = 250) -> list[str]:
    """Spezza il testo in blocchi interi sulle frasi per non far esplodere il context window"""
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
    chunks, current_chunk, current_words = [], [], 0
    for sentence in sentences:
        words = len(sentence.split())
        if current_words + words > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk, current_words = [sentence], words
        else:
            current_chunk.append(sentence)
            current_words += words
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    # Previene lista vuota
    return chunks if chunks else [text]


def concatenate_wavs(wav_list: list, sr: int, gap_ms: int = 80) -> np.ndarray:
    """Unisce i chunk generati inserendo un respiro (pausa) in mezzo"""
    gap = np.zeros(int(sr * gap_ms / 1000))
    result = []
    for i, wav in enumerate(wav_list):
        result.append(wav)
        if i < len(wav_list) - 1:
            result.append(gap)
    if not result:
        return np.array([])
    return np.concatenate(result)


def ensure_padded_ref(ref_path: str) -> str:
    """Implementa l'Autor-Padding dinamico sul server:
    Controlla se esiste *_padded.wav. Se non c'è, lo crea al volo con ffmpeg
    aggiungendo 0.5s di silenzio per fixare il bleeding fonetico Qwen3.
    """
    base, ext = os.path.splitext(ref_path)
    if base.endswith("_padded"):
        padded_path = ref_path
    else:
        padded_path = f"{base}_padded{ext}"
        
    if os.path.exists(padded_path):
        return padded_path
        
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Il file voce originale non esiste: {ref_path}")
        
    logger.info(f"Auto-Padding: sto creando dinamicamente {padded_path} aggiungendo 0.5s di silenzio...")
    try:
        # ffmpeg via subprocess
        cmd = [
            "ffmpeg", "-y",
            "-i", ref_path,
            "-af", "apad=pad_dur=0.5",
            padded_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            logger.error(f"FFMPEG fallito. Ritorno il file non paddato. Errore: {res.stderr.decode('utf-8', errors='ignore')}")
            return ref_path
        return padded_path
    except Exception as e:
        logger.error(f"Errore Auto-Padding: {e}. Uso il file originale.")
        return ref_path


class TTSRequest(BaseModel):
    text: str
    voice_ref_audio_path: str
    voice_ref_text: Optional[str] = None
    language: str = "Italian"
    instruct: str = "Warm Italian male voice, professional audiobook narrator, calm and measured."
    non_streaming_mode: bool = True
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    output_sample_rate: int = 24000
    max_words_per_chunk: int = 250
    gap_between_chunks_ms: int = 80
    output_filename: str = "output.wav"


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Modello non caricato")
    vram = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
    return {"status": "ok", "device": device, "vram_gb": round(vram, 2)}


@app.post("/tts")
def synthesize(req: TTSRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modello non caricato")

    t_start = time.time()

    # AUTO-PADDING: controlla e crea se manca _padded.wav
    try:
        actual_ref_path = ensure_padded_ref(req.voice_ref_audio_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Carica audio di riferimento patchato
    try:
        ref_audio, ref_sr = sf.read(actual_ref_path)
        if ref_audio.ndim > 1:
            ref_audio = ref_audio[:, 0]  # mono
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore lettura ref audio: {e}")

    # CHUNKING
    chunks = chunk_text(req.text, req.max_words_per_chunk)
    logger.info(f"Testo suddiviso in {len(chunks)} chunk (Max words: {req.max_words_per_chunk})")

    wav_chunks = []
    output_sr = req.output_sample_rate

    # Il modello scala in zero-shot se non passiamo il ref_text
    is_x_vector_only = (req.voice_ref_text is None or req.voice_ref_text.strip() == "")
    ref_input = (ref_audio, ref_sr)

    for i, chunk_text_part in enumerate(chunks):
        logger.info(f"Chunk {i+1}/{len(chunks)}: elaborazione {len(chunk_text_part.split())} parole...")
        
        # Saltiamo chunk vuoti
        if not chunk_text_part.strip():
            continue
            
        try:
            wavs, sr = model.generate_voice_clone(
                text=chunk_text_part,
                ref_audio=ref_input,
                ref_text=req.voice_ref_text if not is_x_vector_only else None,
                language=req.language,
                instruct=req.instruct,
                non_streaming_mode=req.non_streaming_mode,
                x_vector_only_mode=is_x_vector_only,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
            )
            output_sr = sr
            wav_chunks.append(wavs[0] if isinstance(wavs, list) else wavs)
        except Exception as e:
            logger.error(f"Errore chunk {i+1}: {e}")
            raise HTTPException(status_code=500, detail=f"Errore inferenza chunk {i+1}: {e}")

    if not wav_chunks:
        raise HTTPException(status_code=400, detail="L'audio generato e' vuoto")

    # CONCATENAZIONE
    final_wav = concatenate_wavs(wav_chunks, output_sr, req.gap_between_chunks_ms)

    # SALVATAGGIO DISCO
    out_path = os.path.join(OUTPUT_DIR, req.output_filename)
    sf.write(out_path, final_wav, output_sr)

    inference_time = time.time() - t_start
    duration = len(final_wav) / output_sr
    rtf = inference_time / duration if duration > 0 else 0

    mode_str = "Zero-Shot" if is_x_vector_only else "ICL"
    logger.info(f"Finito [{mode_str} mode]: {duration:.1f}s audio in {inference_time:.1f}s (RTF {rtf:.2f}x)")

    return {
        "status": "ok",
        "output_path": out_path,
        "duration_seconds": round(duration, 2),
        "sample_rate": output_sr,
        "chunks_count": len(chunks),
        "inference_mode": mode_str,
        "inference_time_seconds": round(inference_time, 2),
        "rtf": round(rtf, 2),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else 0
    }


@app.get("/outputs/{filename}")
def get_output(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File non trovato")
    return FileResponse(path, media_type="audio/wav")


if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host=HOST, port=PORT)
