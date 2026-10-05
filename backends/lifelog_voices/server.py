"""
Lifelog Voices Server — FastAPI (porta ARIA_PORT, default 8094).

Backend «voci» di Lifelog (2026-10-05). Su un WAV 16 kHz mono di 5 minuti:
  - DiariZen (BUT, wavlm-large-s80-md-v2): chi parla quando, con i tratti a più voci (sovrapposizioni);
  - WavLM base-plus, strato 3 + classificatore lineare: probabilità «riprodotto da un altoparlante» per turno.
Nessuna identità: le voci sono etichette anonime del segmento (S0, S1…). ARIA misura, Lifelog decide.
Perché e misure: Lifelog2 docs/lifelog3-roadmap-modelli-ascolto.md (test 5–9), flusso in lifelog3-pipeline-voci.md.

Note di piattaforma:
  - arch spoof sm_120 → (9, 0) prima di pyannote, come lifelog_whisperx_v2;
  - TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1: i checkpoint ufficiali DiariZen/pyannote sono salvati con torch 2.1;
  - DiariZen va in errore sui segmenti senza parlato («negative dimensions»): si risponde status=no_speech.
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import io
import json
import logging
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import soundfile as sf

# --- BLACKWELL ARCH SPOOF — before any pyannote/torchaudio import ---
import torch
_orig_cap = torch.cuda.get_device_capability
def _patched_cap(device=None):
    cap = _orig_cap(device)
    return (9, 0) if cap[0] >= 12 else cap
torch.cuda.get_device_capability = _patched_cap
# ---

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ARIA_ROOT  = os.environ.get("ARIA_ROOT", r"C:\Users\Roberto\aria")
MODELS_DIR = Path(ARIA_ROOT) / "data" / "assets" / "models"
DIARIZEN_DIR = MODELS_DIR / "diarizen" / "wavlm-large-s80-md-v2"
EMBEDDING_CKPT = MODELS_DIR / "pyannote" / "wespeaker-voxceleb-resnet34-LM" / "pytorch_model.bin"
WAVLM_DIR = MODELS_DIR / "wavlm" / "wavlm-base-plus"
REPLAY_CLF = MODELS_DIR / "lifelog-voices" / "replay_wavlm-base-plus_L3_bench_v1.npz"

VERSION = "voices-v1"
SR = 16000
WAVLM_LAYER = 3
WAVLM_CHUNK_S = 20
MIN_TURN_S = 1.0

MINIO_ENDPOINT   = os.getenv("ARIA_MINIO_ENDPOINT",   "192.168.1.104:9000")
MINIO_ACCESS_KEY = os.getenv("ARIA_MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("ARIA_MINIO_SECRET_KEY", "minioadmin")

LOG_FILE = Path(ARIA_ROOT) / "logs" / "lifelog_voices.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()])
logger = logging.getLogger("lifelog_voices")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI(title="ARIA Lifelog Voices")

_diar = _wavlm = _fe = _clf = None


def _load():
    global _diar, _wavlm, _fe, _clf
    t0 = time.time()
    from diarizen.pipelines.inference import DiariZenPipeline
    _diar = DiariZenPipeline(diarizen_hub=DIARIZEN_DIR, embedding_model=str(EMBEDDING_CKPT))
    logger.info("DiariZen loaded in %.1fs", time.time() - t0)
    t0 = time.time()
    from transformers import AutoFeatureExtractor, WavLMModel
    _fe = AutoFeatureExtractor.from_pretrained(str(WAVLM_DIR))
    _wavlm = WavLMModel.from_pretrained(str(WAVLM_DIR)).to(DEVICE).eval()
    d = np.load(REPLAY_CLF)
    _clf = {k: d[k] for k in ("scaler_mean", "scaler_scale", "pca_mean", "pca_components", "coef", "intercept")}
    _clf["meta"] = json.loads(str(d["meta"]))
    logger.info("WavLM + replay classifier loaded in %.1fs (%s)", time.time() - t0, _clf["meta"]["name"])


def _download_file(url: str, dest: str):
    """Come lifelog_whisperx_v2: URL MinIO letti con il client, il resto via HTTP."""
    parsed = urlparse(url)
    if parsed.netloc and ("9000" in parsed.netloc or MINIO_ENDPOINT in parsed.netloc):
        from minio import Minio
        mc = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
        parts = parsed.path.lstrip("/").split("/", 1)
        mc.fget_object(parts[0], parts[1], dest)
    else:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)


def _overlaps(tracks):
    """Tratti con ≥2 voci attive: [[start, end, n_voci]]."""
    ev = sorted([(s, 1) for s, e, _ in tracks] + [(e, -1) for s, e, _ in tracks])
    out, n, last = [], 0, None
    for t, d in ev:
        if n >= 2 and last is not None and t > last:
            if out and out[-1][1] == last and out[-1][2] == n:
                out[-1][1] = t
            else:
                out.append([last, t, n])
        n += d
        last = t
    return [[round(a, 3), round(b, 3), k] for a, b, k in out]


def _p_replay(x: np.ndarray, turns: list[dict]) -> None:
    hs = []
    for a in range(0, len(x), WAVLM_CHUNK_S * SR):
        seg = x[a:a + WAVLM_CHUNK_S * SR]
        if len(seg) < SR // 2:
            continue
        inp = _fe(seg, sampling_rate=SR, return_tensors="pt").input_values.to(DEVICE)
        with torch.no_grad():
            hs.append(_wavlm(inp, output_hidden_states=True).hidden_states[WAVLM_LAYER][0].float().cpu().numpy())
    if not hs:
        return
    H = np.concatenate(hs)
    fps = H.shape[0] / (len(x) / SR)
    c = _clf
    for t in turns:
        a = int(t["start"] * fps)
        f = H[a:max(a + 2, int(t["end"] * fps))]
        v = np.concatenate([f.mean(0), f.std(0)])
        z = ((v - c["scaler_mean"]) / c["scaler_scale"] - c["pca_mean"]) @ c["pca_components"].T
        t["p_replay"] = round(float(1 / (1 + np.exp(-(z @ c["coef"] + c["intercept"])))), 3)


class AnalyzeRequest(BaseModel):
    wav_url: str
    segment_id: str
    replay: bool = True          # calcola p_replay per turno
    min_turn_s: float = MIN_TURN_S


@app.on_event("startup")
def _startup():
    _load()


@app.get("/health")
def health():
    vram = round(torch.cuda.memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0.0
    return {"status": "ok" if _diar is not None else "loading", "model": "lifelog-voices", "version": VERSION,
            "device": DEVICE, "vram_gb": vram,
            "diarization": "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
            "replay": _clf["meta"]["name"] if _clf else None}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if _diar is None:
        raise HTTPException(503, "modelli non ancora caricati")
    t_all = time.perf_counter()
    timing = {}
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "in.wav")
        _download_file(req.wav_url, wav)
        x, sr = sf.read(wav, dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != SR:
            raise HTTPException(400, f"atteso WAV {SR} Hz mono, ricevuto {sr} Hz")
        t = time.perf_counter()
        try:
            ann = _diar(wav)
            tracks = [[round(s.start, 3), round(s.end, 3), f"S{l}"] for s, _, l in ann.itertracks(yield_label=True)]
            status = "ok" if tracks else "no_speech"
        except Exception as exc:   # su audio senza parlato DiariZen fallisce in reconstruct
            logger.warning("DiariZen failed on %s: %s: %s", req.segment_id, type(exc).__name__, exc)
            tracks, status = [], "no_speech"
        timing["diarization"] = round(time.perf_counter() - t, 2)
    overlaps = _overlaps(tracks)
    turns = []
    for s, e, l in tracks:
        if e - s >= req.min_turn_s:
            ov = any(l2 != l and s2 < e and e2 > s for s2, e2, l2 in tracks)
            turns.append({"start": s, "end": e, "voice": l, "overlap": ov})
    if req.replay and turns:
        t = time.perf_counter()
        _p_replay(x, turns)
        timing["replay"] = round(time.perf_counter() - t, 2)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    timing["total"] = round(time.perf_counter() - t_all, 2)
    logger.info("analyze %s: %s, %d tracce, %d sovrapposizioni, %d turni in %.1fs",
                req.segment_id, status, len(tracks), len(overlaps), len(turns), timing["total"])
    return {"output": {
        "version": VERSION, "status": status, "segment_id": req.segment_id,
        "duration_s": round(len(x) / SR, 3),
        "models": {"diarization": "BUT-FIT/diarizen-wavlm-large-s80-md-v2", "replay": _clf["meta"]},
        "tracks": tracks, "overlaps": overlaps, "turns": turns, "timing_s": timing,
    }}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("ARIA_PORT", "8094")), log_level="info")
