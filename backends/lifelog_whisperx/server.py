"""
Lifelog WhisperX Server — FastAPI on port 8091
Wraps WhisperX large-v3 + align (it) + pyannote speaker-diarization-community-1
+ WeSpeaker ResNet293 (voiceprint 256d, better cross-condition invariance than ResNet34).

Output contract matches qwen3-asr-1.7b so Stage C is model-agnostic.

Voiceprint loading:
  - If ARIA_WESPEAKER_PATH env var points to a dir with avg_model.pt: loads from local disk
  - Otherwise: downloads Wespeaker/wespeaker-voxceleb-resnet293-LM from HuggingFace

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
from pyannote.audio.models.embedding.wespeaker import WeSpeakerResNet293

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

# WeSpeaker ResNet293 — auto-detect ARIA_ROOT local path, else HF download
WESPEAKER_HF_REPO = "Wespeaker/wespeaker-voxceleb-resnet293-LM"
_aria_root        = os.environ.get("ARIA_ROOT", "")
WESPEAKER_LOCAL   = os.getenv(
    "ARIA_WESPEAKER_PATH",
    os.path.join(_aria_root, "data", "assets", "models", "pyannote",
                 "wespeaker-voxceleb-resnet293-LM") if _aria_root else "",
)

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
_voiceprint_model  = None   # WeSpeakerResNet293 (256d)


def _load_wespeaker_resnet293(device: str):
    """Load WeSpeakerResNet293 from local wespeaker checkpoint or HuggingFace.

    Wespeaker saves weights as avg_model.pt with keys matching model.resnet.*
    (no extra prefix). We load directly into the pyannote wrapper class.
    """
    ckpt_local = os.path.join(WESPEAKER_LOCAL, "avg_model.pt") if WESPEAKER_LOCAL else ""
    if ckpt_local and os.path.exists(ckpt_local):
        logger.info("Loading WeSpeakerResNet293 from local: %s", ckpt_local)
        ckpt_path = ckpt_local
    else:
        from huggingface_hub import hf_hub_download
        logger.info("Downloading WeSpeakerResNet293 from HF: %s", WESPEAKER_HF_REPO)
        ckpt_path = hf_hub_download(WESPEAKER_HF_REPO, "avg_model.pt",
                                    token=hf_token or None)

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ws_state = raw.get("model", raw) if isinstance(raw, dict) else raw

    model = WeSpeakerResNet293()
    missing, unexpected = model.resnet.load_state_dict(ws_state, strict=False)
    if missing:
        logger.warning("WeSpeakerResNet293: %d missing / %d unexpected keys",
                       len(missing), len(unexpected))
    model.eval()
    model.to(torch.device(device))
    logger.info("WeSpeakerResNet293 loaded — embed_dim=256, device=%s", device)
    return model


def _load_models():
    global _model, _align_model, _align_meta, _diarize_model, _voiceprint_model

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
        _voiceprint_model = _load_wespeaker_resnet293(DEVICE)
        logger.info("Voiceprint encoder loaded in %.1fs", time.time() - t0)
    except Exception as e:
        logger.warning("Voiceprint encoder unavailable: %s -- voiceprints will be empty", e)
        _voiceprint_model = None


def _unload_models():
    global _model, _align_model, _align_meta, _diarize_model, _voiceprint_model
    del _model, _align_model, _align_meta, _diarize_model, _voiceprint_model
    _model = _align_model = _align_meta = _diarize_model = _voiceprint_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Models unloaded, VRAM freed.")


def _embed_turn(
    waveform: torch.Tensor,  # (1, T) float32 at 16kHz
    start_ms: int,
    end_ms: int,
) -> list[float] | None:
    """256d ResNet293 embedding for a single speaker turn. Returns None if too short."""
    if _voiceprint_model is None:
        return None
    sr = 16000
    s = int(start_ms / 1000 * sr)
    e = int(end_ms   / 1000 * sr)
    crop = waveform[:, s:e]
    if crop.shape[1] < sr // 2:   # skip turns < 0.5 s
        return None
    dev = next(_voiceprint_model.parameters()).device
    crop = crop.to(dev)
    try:
        with torch.no_grad():
            emb = _voiceprint_model(crop.unsqueeze(0))  # (1, 256)
        return emb[0].cpu().tolist()
    except Exception as exc:
        logger.warning("Voiceprint failed [%d–%d ms]: %s", start_ms, end_ms, exc)
        return None


def _vp_confidence(
    logprob: float,
    duration_s: float,
    word_count: int,
    no_speech_prob: float,
) -> float:
    """Reliability score [0.0–1.0] for a per-turn voiceprint embedding.

    Combines ASR quality (logprob), audio quantity (duration + word count),
    and speech presence (no_speech_prob) into a single trustworthiness score.
    Stage C uses this to weight identity matching and chimera detection.
    """
    lp  = max(0.0, min(1.0, (logprob + 0.45) / 0.45))         # 0 at −0.45, 1 at 0.0
    dur = max(0.0, min(1.0, (duration_s - 2.0) / 8.0))         # 0 at 2 s,   1 at 10 s+
    wc  = max(0.0, min(1.0, (word_count - 5)  / 15.0))         # 0 at 5 w,   1 at 20 w+
    ns  = max(0.0, min(1.0, (0.5 - no_speech_prob) / 0.4))     # penalises silence
    return round(0.40 * lp + 0.30 * dur + 0.20 * wc + 0.10 * ns, 3)


def _log_diarization_stats(diarize_df, speaker_embeddings=None) -> dict:
    """Diagnostica sull'output GREZZO di pyannote, prima dell'incrocio con le parole.

    Serve a distinguere due cause possibili dello sfarfallio dei parlanti:
      (a) pyannote produce intervalli brevi e spuri  → problema di diarizzazione
      (b) pyannote è stabile ma i timestamp delle parole sono imprecisi ai bordi
          → problema di riconciliazione (pyannote community-1 espone per questo
             `exclusive_speaker_diarization`, che whisperx non usa)
    """
    try:
        durs = [(float(r["end"]) - float(r["start"])) for _, r in diarize_df.iterrows()]
        spks = [str(r["speaker"]) for _, r in diarize_df.iterrows()]
    except Exception as exc:
        logger.warning("Diarize stats non calcolabili: %s", exc)
        return {}

    n = len(durs)
    if n == 0:
        logger.warning("Diarize: nessun intervallo prodotto")
        return {}

    srt = sorted(durs)
    # alternanze A-B-A dove B è molto breve: firma del rumore di diarizzazione
    flips = sum(
        1 for i in range(1, n - 1)
        if spks[i - 1] == spks[i + 1] != spks[i] and durs[i] < 1.0
    )
    stats = {
        "n_intervals":   n,
        "n_speakers":    len(set(spks)),
        "dur_min":       round(srt[0], 2),
        "dur_median":    round(srt[n // 2], 2),
        "dur_max":       round(srt[-1], 2),
        "n_under_0s5":   sum(1 for d in durs if d < 0.5),
        "n_under_1s":    sum(1 for d in durs if d < 1.0),
        "short_flips":   flips,
    }
    logger.info(
        "DIARIZE RAW: %d intervalli / %d parlanti | durata min=%.2fs mediana=%.2fs max=%.2fs "
        "| <0.5s: %d, <1s: %d | alternanze brevi A-B-A: %d",
        stats["n_intervals"], stats["n_speakers"], stats["dur_min"], stats["dur_median"],
        stats["dur_max"], stats["n_under_0s5"], stats["n_under_1s"], stats["short_flips"],
    )
    if speaker_embeddings is not None:
        try:
            dims = {k: (len(v) if hasattr(v, "__len__") else "?") for k, v in speaker_embeddings.items()}
            logger.info("DIARIZE EMBEDDINGS: %d parlanti, dimensioni %s", len(dims), dims)
            stats["embedding_dims"] = dims
        except Exception as exc:
            logger.warning("Embeddings pyannote non ispezionabili: %s", exc)
    return stats


def _to_contract(
    wx_result: dict,
    audio_np: np.ndarray,   # float32 mono 16kHz
    language: str,
    speaker_embeddings: dict | None = None,
    diarize_stats: dict | None = None,
) -> dict:
    """Convert whisperx output to the Stage C contract.

    Each speaker_turn now carries its own voiceprint embedding (256d ResNet293)
    and vp_confidence score, replacing the old top-level voiceprints dict that
    produced a single centroid per speaker label regardless of how many distinct
    speakers pyannote had conflated under that label.
    """
    sr = 16000
    duration_ms = int(len(audio_np) / sr * 1000)
    segments = wx_result.get("segments", [])

    raw_logprobs:  list[float] = []
    raw_nospeechs: list[float] = []

    speaker_turns:          list[dict]        = []
    turn_logprobs:          list[list[float]] = []
    turn_nospeechs:         list[list[float]] = []
    turn_word_counts:       list[int]         = []
    turn_word_scores:       list[list[float]] = []
    turn_compression_ratios: list[list[float]] = []

    # ── Costruzione turni: per SEGMENTO (comportamento storico) ──────────────
    # Il taglio per PAROLA (provato il 2026-07-28) e' tecnicamente corretto —
    # assign_word_speakers assegna lo speaker a ogni parola e whisperx lo
    # scarta — ma su questo audio AMPLIFICA il rumore invece di filtrarlo:
    # pyannote produce 159 intervalli in 5 minuti con 42 sotto il mezzo secondo
    # (minimo 0.02s), quindi tagliare a ogni cambio genera turni da una parola.
    # Misurato: 30% dei turni a 1-2 parole, 20 alternanze A-B-A spurie.
    # Il voto di maggioranza per segmento non e' preciso, ma FILTRA quel rumore.
    # Provati e scartati, tutti peggiorativi (vedi lifelog2-aria-diarization-
    # report): min_speakers 3/4/5, exclusive_speaker_diarization.
    # I segnali nuovi (speaker per parola, embedding per parlante,
    # diarization_stats) restano nel contratto: servono a Stage C1 per
    # ETICHETTARE l'affidabilita', non per ricostruire i turni.
    for seg in segments:
        spk  = seg.get("speaker", "SPEAKER_00")
        s_ms = int(seg.get("start", 0) * 1000)
        e_ms = int(seg.get("end",   0) * 1000)
        text = seg.get("text", "").strip()

        logprob = seg.get("avg_logprob")
        if logprob is None:
            logprob = -0.5
        nospeech = seg.get("no_speech_prob")
        if nospeech is None:
            nospeech = 0.1
        compression_ratio = seg.get("compression_ratio", 1.0)

        raw_logprobs.append(logprob)
        raw_nospeechs.append(nospeech)

        # word-level signals from wav2vec2 alignment
        seg_words = seg.get("words", [])
        seg_wc = len(seg_words)
        seg_word_scores = [w["score"] for w in seg_words if w.get("score") is not None]

        if speaker_turns and speaker_turns[-1]["speaker"] == spk:
            speaker_turns[-1]["end_ms"] = e_ms
            speaker_turns[-1]["text"]   = (speaker_turns[-1]["text"] + " " + text).strip()
            turn_logprobs[-1].append(logprob)
            turn_nospeechs[-1].append(nospeech)
            turn_word_counts[-1] += seg_wc
            turn_word_scores[-1].extend(seg_word_scores)
            turn_compression_ratios[-1].append(compression_ratio)
        else:
            speaker_turns.append({"speaker": spk, "start_ms": s_ms, "end_ms": e_ms, "text": text})
            turn_logprobs.append([logprob])
            turn_nospeechs.append([nospeech])
            turn_word_counts.append(seg_wc)
            turn_word_scores.append(list(seg_word_scores))
            turn_compression_ratios.append([compression_ratio])

    # Per-turn quality metrics + individual voiceprint embedding
    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    for i, turn in enumerate(speaker_turns):
        lps  = turn_logprobs[i]  or [-0.5]
        nsps = turn_nospeechs[i] or [0.1]
        crs  = turn_compression_ratios[i] or [1.0]
        lp  = round(sum(lps)  / len(lps),  4)
        nsp = round(sum(nsps) / len(nsps), 4)
        wc  = turn_word_counts[i]
        dur_s = (turn["end_ms"] - turn["start_ms"]) / 1000.0
        scores = turn_word_scores[i]

        turn["avg_logprob"]            = lp
        turn["no_speech_prob"]         = nsp
        turn["word_count"]             = wc
        turn["n_segments_merged"]      = len(turn_logprobs[i])
        turn["avg_word_score"]         = round(sum(scores) / len(scores), 4) if scores else None
        turn["avg_compression_ratio"]  = round(sum(crs) / len(crs), 4)

        emb = _embed_turn(waveform, turn["start_ms"], turn["end_ms"])
        turn["embedding"]     = emb
        turn["vp_confidence"] = (
            _vp_confidence(lp, dur_s, wc, nsp) if emb is not None else 0.0
        )

    # word_timestamps: flat list with ms timestamps + wav2vec2 alignment score
    # + speaker per parola (2026-07-28): assign_word_speakers lo assegna già,
    # prima veniva scartato qui. Permette al consumatore di ri-verificare o
    # ri-tagliare i turni senza rifare inferenza. Campo additivo.
    word_timestamps: list[dict] = []
    for w in wx_result.get("word_segments", []):
        entry: dict = {
            "word":     w.get("word", ""),
            "start_ms": int(w.get("start", 0) * 1000),
            "end_ms":   int(w.get("end",   0) * 1000),
        }
        if w.get("score") is not None:
            entry["score"] = round(w["score"], 4)
        if w.get("speaker") is not None:
            entry["speaker"] = w["speaker"]
        word_timestamps.append(entry)

    # Global transcription quality
    n_segments = len(segments)
    if n_segments > 0:
        avg_logprob_mean    = sum(raw_logprobs) / n_segments
        no_speech_prob_mean = sum(raw_nospeechs) / n_segments
        no_speech_prob_max  = max(raw_nospeechs)
    else:
        avg_logprob_mean = no_speech_prob_mean = no_speech_prob_max = 0.0

    transcription_quality = {
        "avg_logprob_mean":    round(avg_logprob_mean,    4),
        "no_speech_prob_mean": round(no_speech_prob_mean, 4),
        "no_speech_prob_max":  round(no_speech_prob_max,  4),
        "n_segments":          n_segments,
    }

    n_emb = sum(1 for t in speaker_turns if t.get("embedding") is not None)
    logger.info("_to_contract: %d turns, %d with embedding", len(speaker_turns), n_emb)

    # Embedding per PARLANTE calcolati da pyannote sui propri confini: più
    # robusti del per-turno quando i turni sono brevi (2026-07-28, additivo).
    # Le etichette (SPEAKER_00...) sono locali al segmento, come le altre.
    speaker_embeddings_out: dict | None = None
    if speaker_embeddings:
        speaker_embeddings_out = {}
        for spk, emb in speaker_embeddings.items():
            try:
                speaker_embeddings_out[str(spk)] = [round(float(x), 6) for x in emb]
            except Exception:
                continue

    out = {
        "transcript":            " ".join(s.get("text", "") for s in segments).strip(),
        "language":              language,
        "duration_ms":           duration_ms,
        "speaker_turns":         speaker_turns,
        "word_timestamps":       word_timestamps,
        "transcription_quality": transcription_quality,
    }
    if speaker_embeddings_out:
        out["speaker_embeddings"] = speaker_embeddings_out
    if diarize_stats:
        out["diarization_stats"] = diarize_stats
    return out


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
    # Vincoli opzionali sul numero di parlanti per la diarizzazione.
    # None = stima automatica di pyannote (comportamento storico).
    min_speakers: int | None = None
    max_speakers: int | None = None
    # Usa la diarizzazione "esclusiva" (non sovrapposta) di pyannote
    # community-1 invece di quella standard usata da whisperx.
    exclusive_diarization: bool = False


class VoiceprintRequest(BaseModel):
    wav_url:    str
    segment_id: str
    turns: list[dict]  # [{"speaker": str, "start_ms": int, "end_ms": int}, ...]


@app.get("/health")
def health():
    vram = round(torch.cuda.memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0.0
    return {
        "status": "ok",
        "model":  f"whisperx-{MODEL_SIZE}",
        "device": DEVICE,
        "vram_gb": vram,
        "voiceprint": _voiceprint_model is not None,
        "voiceprint_model": "resnet293",
    }


@app.post("/voiceprint")
def voiceprint(req: VoiceprintRequest):
    """Extract 256d ResNet293 embeddings for given speaker turns — no ASR.

    Used by re-enrollment scripts. Each unique speaker in `turns` gets one
    centroid embedding (concatenated audio, max 30s).

    Returns: {"status": "done", "voiceprints": {"SPEAKER_XX": [256 floats], ...}}
    """
    if _voiceprint_model is None:
        raise HTTPException(status_code=503, detail="Voiceprint model not loaded")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_path = tmp.name
    tmp.close()

    try:
        _download_file(req.wav_url, wav_path)
        audio_np, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != 16000:
            import resampy
            audio_np = resampy.resample(audio_np, sr, 16000)

        waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (1, T)

        # Embed each turn individually, then centroid per label (enrollment use case)
        from collections import defaultdict
        per_label: dict[str, list[list[float]]] = defaultdict(list)
        for turn in req.turns:
            emb = _embed_turn(waveform, turn["start_ms"], turn["end_ms"])
            if emb is not None:
                per_label[turn["speaker"]].append(emb)

        voiceprints: dict[str, list[float]] = {}
        for label, embs in per_label.items():
            dim = len(embs[0])
            centroid = [sum(e[d] for e in embs) / len(embs) for d in range(dim)]
            voiceprints[label] = centroid

    except Exception as exc:
        logger.error("Voiceprint error for %s: %s", req.segment_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    logger.info("Voiceprint %s -- %d speakers", req.segment_id, len(voiceprints))
    return {"status": "done", "output": {"voiceprints": voiceprints}}


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
        # return_embeddings: pyannote calcola già un embedding per PARLANTE sui
        # propri confini (più robusto del nostro per-turno, che su turni brevi
        # ha poco audio). Lo chiediamo per valutarne l'uso; se la versione di
        # whisperx non lo supporta si ricade sul comportamento precedente.
        speaker_embeddings = None
        # min/max_speakers opzionali dal payload: pyannote stima da solo il
        # numero di parlanti e sui nostri dati sbaglia per difetto (2 su 4-5
        # reali), collassando voci diverse nello stesso cluster. Passabili per
        # esperimento; None = comportamento automatico di prima.
        diar_kw = {}
        if req.min_speakers is not None:
            diar_kw["min_speakers"] = req.min_speakers
        if req.max_speakers is not None:
            diar_kw["max_speakers"] = req.max_speakers
        if diar_kw:
            logger.info("Diarize: vincoli sul numero di parlanti = %s", diar_kw)
        if req.exclusive_diarization:
            # pyannote community-1 espone `exclusive_speaker_diarization`, una
            # versione non sovrapposta pensata dagli autori proprio per
            # "semplificare la riconciliazione tra i timestamp fini della
            # diarizzazione e quelli (a volte non cosi' precisi) della
            # trascrizione". whisperx usa solo output.speaker_diarization,
            # quindi qui si chiama pyannote direttamente e si ricostruisce il
            # DataFrame nello stesso formato che assign_word_speakers si aspetta.
            import pandas as pd
            audio_data = {
                "waveform": torch.from_numpy(audio_np[None, :]),
                "sample_rate": 16000,
            }
            pyannote_out = _diarize_model.model(audio_data, **diar_kw)
            ann = getattr(pyannote_out, "exclusive_speaker_diarization", None)
            if ann is None:
                logger.warning("exclusive_speaker_diarization non disponibile — uso la standard")
                ann = pyannote_out.speaker_diarization
            else:
                logger.info("Diarize: uso exclusive_speaker_diarization")
            diarize_segs = pd.DataFrame(
                ann.itertracks(yield_label=True), columns=["segment", "label", "speaker"]
            )
            diarize_segs["start"] = diarize_segs["segment"].apply(lambda x: x.start)
            diarize_segs["end"]   = diarize_segs["segment"].apply(lambda x: x.end)
        else:
            try:
                diarize_segs, speaker_embeddings = _diarize_model(
                    audio_np, return_embeddings=True, **diar_kw)
            except TypeError:
                logger.info("Diarize: return_embeddings non supportato da questa versione di whisperx")
                diarize_segs = _diarize_model(audio_np, **diar_kw)
        diarize_stats = _log_diarization_stats(diarize_segs, speaker_embeddings)
        wx_result = whisperx.assign_word_speakers(diarize_segs, wx_result)
        logger.info("Diarize done in %.1fs", time.perf_counter() - t_diar)

        t_vp = time.perf_counter()
        output = _to_contract(wx_result, audio_np, detected_lang,
                              speaker_embeddings=speaker_embeddings,
                              diarize_stats=diarize_stats)
        n_emb = sum(1 for t in output["speaker_turns"] if t.get("embedding") is not None)
        logger.info("Voiceprint done in %.1fs -- %d turns with embedding",
                    time.perf_counter() - t_vp, n_emb)

    except Exception as exc:
        logger.error("Pipeline error for %s: %s", req.segment_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    elapsed = round(time.perf_counter() - t0, 2)
    n_emb = sum(1 for t in output["speaker_turns"] if t.get("embedding") is not None)
    logger.info(
        "Done %s in %.1fs -- %d chars, %d turns, %d with_embedding",
        req.segment_id, elapsed,
        len(output["transcript"]),
        len(output["speaker_turns"]),
        n_emb,
    )

    return {"status": "done", "processing_time": elapsed, "output": output}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8091, log_level="info")
