"""
Lifelog WhisperX Server — FastAPI on port 8091
Wraps WhisperX large-v3 + align (it) + pyannote speaker-diarization-community-1
+ pyannote wespeaker-resnet34-LM (voiceprint 256d).

Output contract matches qwen3-asr-1.7b so Stage C is model-agnostic.

Blackwell fixes:
  - compute_type="float16"  (cuBLAS int8 -> NOT_SUPPORTED on sm_120)
  - arch spoof pre-pyannote  (NVRTC Jiterator fails on complex FFT at sm_120)
"""

import os
import sys
import logging

# ── Path convention: mirrors flux_imagegen backend ──────────────────────────
ARIA_ROOT  = os.environ.get("ARIA_ROOT", r"C:\Users\Roberto\aria")
MODELS_DIR = os.path.join(ARIA_ROOT, "data", "assets", "models")

# Redirect all HF downloads to aria/data (alignment model, pyannote diarize)
os.environ["HF_HOME"] = MODELS_DIR
# orchestrator injects HF_HUB_OFFLINE=1; pop it so hf_hub can reach HF when a
# model is not yet cached in aria/data (e.g. alignment model on first start)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# Local model paths
WHISPER_MODEL_PATH = os.path.join(MODELS_DIR, "faster-whisper-large-v3")
WESPEAKER_PATH     = os.path.join(MODELS_DIR, "pyannote", "wespeaker-voxceleb-resnet34-LM")
ALIGN_CACHE_DIR    = os.path.join(MODELS_DIR, "whisperx-align")
DIARIZE_MODEL_PATH = os.path.join(MODELS_DIR, "pyannote", "speaker-diarization-community-1")

# Add conda Library/bin to PATH so whisperx finds ffmpeg.exe (conda-forge puts it there)
_env_root = os.path.dirname(sys.executable)
_ffmpeg_dir = os.path.join(_env_root, "Library", "bin")
if os.path.isdir(_ffmpeg_dir) and _ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

# --- BLACKWELL ARCH SPOOF — before any pyannote/torchaudio import ---
import torch
_orig_cap = torch.cuda.get_device_capability
def _patched_cap(device=None):
    cap = _orig_cap(device)
    return (9, 0) if cap[0] >= 12 else cap
torch.cuda.get_device_capability = _patched_cap
# ---

import torchaudio
from types import ModuleType

# Monkeypatch torchaudio for speechbrain/pyannote compatibility
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]
if not hasattr(torchaudio, "io"):
    _io = ModuleType("torchaudio.io")
    _io.StreamReader = object
    torchaudio.io = _io
    sys.modules["torchaudio.io"] = _io

import gc
import time
import tempfile
import numpy as np
import soundfile as sf
from urllib.parse import urlparse
from contextlib import asynccontextmanager

import whisperx
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import login
from minio import Minio
from pyannote.audio import Model, Inference

load_dotenv()

hf_token = os.getenv("HF_TOKEN", "")
if hf_token:
    try:
        login(token=hf_token)
    except Exception as e:
        logging.warning("HF Login failed: %s", e)

LOG_FILE = r"C:\Users\Roberto\aria\logs\lifelog_whisperx.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16"
MODEL_SIZE   = os.getenv("WHISPER_MODEL_SIZE", "large-v3")
LANGUAGE     = os.getenv("WHISPER_LANGUAGE", "it")

MINIO_ENDPOINT   = os.getenv("ARIA_MINIO_ENDPOINT",   "192.168.1.104:9000")
MINIO_ACCESS_KEY = os.getenv("ARIA_MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("ARIA_MINIO_SECRET_KEY", "minioadmin")

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

_model             = None
_align_model       = None
_align_meta        = None
_diarize_model     = None
_voiceprint_encoder = None   # pyannote wespeaker-resnet34-LM


def _load_models():
    global _model, _align_model, _align_meta, _diarize_model, _voiceprint_encoder

    logger.info("Loading WhisperX %s on %s (%s) from %s ...", MODEL_SIZE, DEVICE, COMPUTE_TYPE, WHISPER_MODEL_PATH)
    t0 = time.time()
    _model = whisperx.load_model(WHISPER_MODEL_PATH, DEVICE, compute_type=COMPUTE_TYPE)
    logger.info("ASR model loaded in %.1fs", time.time() - t0)

    t0 = time.time()
    _align_model, _align_meta = whisperx.load_align_model(
        language_code=LANGUAGE, device=DEVICE, model_dir=ALIGN_CACHE_DIR
    )
    logger.info("Align model loaded in %.1fs", time.time() - t0)

    t0 = time.time()
    from whisperx.diarize import DiarizationPipeline
    _diarize_model = DiarizationPipeline(model_name=DIARIZE_MODEL_PATH, token=hf_token or None, device=DEVICE)
    logger.info("Diarize model loaded in %.1fs", time.time() - t0)

    t0 = time.time()
    try:
        emb_model = Model.from_pretrained(WESPEAKER_PATH)
        _voiceprint_encoder = Inference(emb_model, window="whole", device=torch.device(DEVICE))
        logger.info("Voiceprint encoder loaded in %.1fs", time.time() - t0)
    except Exception as e:
        logger.warning("Voiceprint encoder unavailable: %s -- voiceprints will be empty", e)
        _voiceprint_encoder = None


def _unload_models():
    global _model, _align_model, _align_meta, _diarize_model, _voiceprint_encoder
    del _model, _align_model, _align_meta, _diarize_model, _voiceprint_encoder
    _model = _align_model = _align_meta = _diarize_model = _voiceprint_encoder = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Models unloaded, VRAM freed.")


def _extract_voiceprints(
    waveform: torch.Tensor,   # (1, T) float32 at 16kHz
    speaker_turns: list[dict],
) -> dict[str, list[float]]:
    """256d ResNet34 embedding per speaker, pooled over all their turns (max 30s)."""
    if _voiceprint_encoder is None:
        return {}

    sr = 16000
    max_samples = sr * 30

    # Collect audio crops per speaker
    crops: dict[str, list[torch.Tensor]] = {}
    for turn in speaker_turns:
        spk = turn["speaker"]
        s = int(turn["start_ms"] / 1000 * sr)
        e = int(turn["end_ms"]   / 1000 * sr)
        crop = waveform[:, s:e]
        if crop.shape[1] > sr // 2:   # skip segments < 0.5s
            crops.setdefault(spk, []).append(crop)

    voiceprints: dict[str, list[float]] = {}
    for spk, tensors in crops.items():
        concat = torch.cat(tensors, dim=1)[:, :max_samples]
        try:
            with torch.no_grad():
                emb = _voiceprint_encoder({"waveform": concat, "sample_rate": sr})
            voiceprints[spk] = emb.tolist()
        except Exception as exc:
            logger.warning("Voiceprint failed for %s: %s", spk, exc)

    return voiceprints


def _to_contract(
    wx_result: dict,
    audio_np: np.ndarray,   # float32 mono 16kHz
    language: str,
) -> dict:
    """Convert whisperx output to the Stage C contract (same as qwen3-asr)."""
    sr = 16000
    duration_ms = int(len(audio_np) / sr * 1000)
    segments = wx_result.get("segments", [])

    # speaker_turns: merge consecutive same-speaker segments, times in ms
    speaker_turns: list[dict] = []
    for seg in segments:
        spk  = seg.get("speaker", "SPEAKER_00")
        s_ms = int(seg.get("start", 0) * 1000)
        e_ms = int(seg.get("end",   0) * 1000)
        text = seg.get("text", "").strip()

        if speaker_turns and speaker_turns[-1]["speaker"] == spk:
            speaker_turns[-1]["end_ms"] = e_ms
            speaker_turns[-1]["text"]   = (speaker_turns[-1]["text"] + " " + text).strip()
        else:
            speaker_turns.append({"speaker": spk, "start_ms": s_ms, "end_ms": e_ms, "text": text})

    # word_timestamps: from word_segments, times in ms
    word_timestamps: list[dict] = []
    for w in wx_result.get("word_segments", []):
        word_timestamps.append({
            "word":     w.get("word", ""),
            "start_ms": int(w.get("start", 0) * 1000),
            "end_ms":   int(w.get("end",   0) * 1000),
        })

    # voiceprints: need waveform as torch tensor (1, T)
    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    voiceprints = _extract_voiceprints(waveform, speaker_turns)

    return {
        "transcript":      " ".join(s.get("text", "") for s in segments).strip(),
        "language":        language,
        "duration_ms":     duration_ms,
        "speaker_turns":   speaker_turns,
        "word_timestamps": word_timestamps,
        "voiceprints":     voiceprints,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield
    _unload_models()


app = FastAPI(title="Lifelog WhisperX", version="2.0.0", lifespan=lifespan)


def _download_file(url: str, dest: str):
    parsed = urlparse(url)
    if parsed.netloc and ("9000" in parsed.netloc or MINIO_ENDPOINT in parsed.netloc):
        parts = parsed.path.lstrip("/").split("/", 1)
        minio_client.fget_object(parts[0], parts[1], dest)
    else:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)


class TranscribeRequest(BaseModel):
    wav_url:    str
    segment_id: str
    language:   str = "it"


@app.get("/health")
def health():
    vram = round(torch.cuda.memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0.0
    return {
        "status": "ok",
        "model":  f"whisperx-{MODEL_SIZE}",
        "device": DEVICE,
        "vram_gb": vram,
        "voiceprint": _voiceprint_encoder is not None,
    }


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    t0 = time.perf_counter()
    logger.info("Transcription request: %s", req.segment_id)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_path = tmp.name
    tmp.close()

    try:
        _download_file(req.wav_url, wav_path)

        # Load as float32 mono 16kHz -- bypasses ffmpeg for WAV files
        audio_np, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != 16000:
            import resampy
            audio_np = resampy.resample(audio_np, sr, 16000)

        t_asr = time.perf_counter()
        wx_result = _model.transcribe(audio_np, batch_size=4, language=req.language)
        detected_lang = wx_result.get("language", req.language)
        logger.info("ASR done in %.1fs -- lang=%s", time.perf_counter() - t_asr, detected_lang)

        t_align = time.perf_counter()
        wx_result = whisperx.align(
            wx_result["segments"], _align_model, _align_meta,
            audio_np, DEVICE, return_char_alignments=False,
        )
        logger.info("Align done in %.1fs", time.perf_counter() - t_align)

        t_diar = time.perf_counter()
        diarize_segs = _diarize_model(audio_np)
        wx_result = whisperx.assign_word_speakers(diarize_segs, wx_result)
        logger.info("Diarize done in %.1fs", time.perf_counter() - t_diar)

        t_vp = time.perf_counter()
        output = _to_contract(wx_result, audio_np, detected_lang)
        logger.info("Voiceprint done in %.1fs -- %d speakers",
                    time.perf_counter() - t_vp, len(output["voiceprints"]))

    except Exception as exc:
        logger.error("Pipeline error for %s: %s", req.segment_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "Done %s in %.1fs -- %d chars, %d turns, %d voiceprints",
        req.segment_id, elapsed,
        len(output["transcript"]),
        len(output["speaker_turns"]),
        len(output["voiceprints"]),
    )

    return {"status": "done", "processing_time": elapsed, "output": output}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8091, log_level="info")
