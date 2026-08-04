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
ALIGN_CACHE_DIR    = os.path.join(MODELS_DIR, "whisperx-align")
DIARIZE_MODEL_PATH = os.path.join(MODELS_DIR, "pyannote", "speaker-diarization-community-1")
# Il path del voiceprint (resnet293) è calcolato più sotto in WESPEAKER_LOCAL —
# rimosso 2026-07-30 un WESPEAKER_PATH morto che puntava ancora al vecchio
# resnet34, mai usato da _load_wespeaker_resnet293.

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

# --- WARNING NOTI E INNOCUI — silenziati qui, PRIMA di importare pyannote,
# perché sono avvisi emessi all'IMPORT del modulo (torchcodec) o durante
# l'inferenza (TF32, Lightning). Verificati uno per uno (2026-07-30):
#
# 1. torchcodec: pyannote.audio lo controlla sempre all'import, ma questo
#    server non usa MAI l'I/O di pyannote/torchaudio per decodificare audio —
#    i WAV arrivano già come array da soundfile (bypass deliberato, vedi Fix 3
#    in docs/backends/lifelog-whisperx.md). L'avviso non ha alcun effetto qui.
# 2. TF32 disabilitato: scelta DELIBERATA di pyannote per riproducibilità,
#    non un errore né un problema di configurazione nostro.
# 3. "Lightning automatically upgraded your loaded checkpoint": adeguamento
#    automatico e innocuo del formato del checkpoint, nessun impatto sui
#    risultati.
#
# NON silenziato di proposito: "std(): degrees of freedom is <= 0" — è il
# sintomo di un bug noto e non risolto upstream in pyannote-audio (issue
# github.com/pyannote/pyannote-audio/issues/1861: il pooling statistico
# calcola una deviazione standard con correzione di Bessel su una finestra
# di un solo frame, producendo NaN). Qui si manifesta quando pyannote
# diarizza un audio senza voce rilevata. Il crash che ne conseguiva è
# corretto scartando l'embedding non valido invece di propagarlo (vedi
# _to_contract, sezione "embeddings per parlante") — ma il warning stesso
# resta visibile perché è un segnale diagnostico legittimo, già usato in
# docs/backends/lifelog-whisperx.md §2bis per misurare la qualità della
# diarizzazione.
import warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*automatically upgraded your loaded checkpoint.*")
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
import math
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
        # Atteso quando HF_HUB_OFFLINE è attivo (iniettato dall'orchestratore) e
        # i modelli sono già in cache locale: non serve raggiungere HF per
        # partire. INFO, non WARNING — non è un problema da controllare.
        logging.info(
            "Login HF non effettuato (modalità offline, modelli già in cache locale): %s", e
        )

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

    logger.info(
        "Tutti i modelli caricati. Promemoria avvisi innocui silenziati in questo avvio "
        "(verificati il 2026-07-30, non richiedono attenzione): "
        "'torchcodec non installato correttamente' — mai usato, l'audio arriva già "
        "come array da soundfile; 'TensorFloat-32 disabilitato' — scelta di pyannote "
        "per riproducibilità; 'checkpoint aggiornato automaticamente' — adeguamento "
        "di formato, nessun impatto. Resta invece visibile 'std(): degrees of freedom' "
        "perché segnala diarizzazione su audio poco affidabile — vedi "
        "docs/backends/lifelog-whisperx.md §2bis."
    )


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
    return _embed_intervals(waveform, [(start_ms, end_ms)])


def _embed_intervals(
    waveform: torch.Tensor,          # (1, T) float32 at 16kHz
    intervals: list[tuple[int, int]],  # [(start_ms, end_ms), ...]
) -> list[float] | None:
    """Embedding sul concatenato degli intervalli. None se il totale è < 0.5s.

    Nato per calcolare il voiceprint sugli intervalli ACUSTICI di pyannote
    invece che sulla finestra testuale di whisper (2026-07-29): i confini di
    whisper sono schiacciati di ~0.3s ai cambi di parlante (l'allineatore
    comprime le ultime parole dentro il segmento), quindi la finestra del turno
    successivo contiene la coda della voce precedente. Su un turno da 2-3s è il
    10-15% del campione — una delle cause della confusione dei turni brevi.
    Concatenare solo gli intervalli in cui pyannote sente QUESTA voce toglie
    dal campione il parlato dedicato degli altri.
    """
    if _voiceprint_model is None:
        return None
    sr = 16000
    crops = []
    for s_ms, e_ms in intervals:
        s = max(0, int(s_ms / 1000 * sr))
        e = min(waveform.shape[1], int(e_ms / 1000 * sr))
        if e > s:
            crops.append(waveform[:, s:e])
    if not crops:
        return None
    crop = torch.cat(crops, dim=1)
    if crop.shape[1] < sr // 2:   # skip < 0.5 s of usable speech
        return None
    dev = next(_voiceprint_model.parameters()).device
    crop = crop.to(dev)
    try:
        with torch.no_grad():
            emb = _voiceprint_model(crop.unsqueeze(0))  # (1, 256)
        return emb[0].cpu().tolist()
    except Exception as exc:
        logger.warning("Voiceprint failed on %d intervals: %s", len(intervals), exc)
        return None


# Quanto un intervallo pyannote può sporgere oltre la finestra whisper del
# turno prima di venire tagliato. Serve a due cose: tenere la sporgenza vera
# (lo squeeze misurato è ~0.3-0.5s) e impedire che un intervallo lungo che
# attraversa più turni A-B-A trascini dentro la finestra di un altro turno.
_DIAR_MARGIN_MS = 1000

# Soglia minima per un embedding voiceprint (stessa di _embed_intervals) — usata
# per accorpare cluster di intervalli grezzi troppo brevi da soli.
RAW_CLUSTER_MIN_MS = 500


def _embed_raw_clusters(
    waveform: torch.Tensor,
    diarization_raw: list[dict],
) -> list[dict]:
    """Un embedding voiceprint per gruppo di intervalli pyannote GREZZI (non
    ancora incrociati con nessuna logica di costruzione turni), pensato per
    essere l'unità più piccola e stabile possibile da archiviare (2026-08-03).

    Perché non "per turno": i turni sono una decisione di Lifelog2 (oggi
    _split_fused_turns, domani un'altra logica) — se l'embedding fosse legato
    al turno, cambiare quella logica renderebbe tutti gli embedding storici
    inutilizzabili e servirebbe ririlanciare Stage C su tutto. Legandolo invece
    all'intervallo grezzo di pyannote (che non dipende da nessuna scelta a
    valle), Lifelog2 può sempre ricombinare (media, poi normalizzazione L2 —
    stessa tecnica dei centroidi voiceprint) gli embedding già calcolati per
    qualunque raggruppamento in turni decida in futuro, senza mai richiamare
    né GPU né audio grezzo (cancellato subito dopo questa richiesta).

    Accorpa gli intervalli consecutivi della STESSA etichetta (ordinati nel
    tempo) finché non superano RAW_CLUSTER_MIN_MS — sotto soglia un embedding
    non è affidabile (vedi _embed_intervals). Un resto sotto soglia alla fine
    si attacca all'ultimo cluster della stessa etichetta se esiste; altrimenti
    resta senza embedding, esattamente come oggi per un turno troppo breve.

    Ogni cluster porta `interval_indices` — gli indici in `diarization_raw`
    che lo compongono — cosi il consumatore sa sempre esattamente quali
    intervalli grezzi condividono quell'embedding.

    Costo misurato (2026-08-03, riprocessamento storico): con 50-80 cluster
    per segmento (audio con diarizzazione frammentata), il ciclo originale —
    una chiamata al modello PER cluster — arrivava a 140-320s extra a
    segmento, senza nessuna proporzione con la quantità di parlato reale (un
    segmento da 469 caratteri con 82 intervalli costava quanto uno da 4000+
    caratteri). Da qui _embed_batch_grouped: stesso calcolo, un solo giro di
    GPU per gruppo invece che uno per cluster."""
    if _voiceprint_model is None or not diarization_raw:
        return []

    by_speaker: dict[str, list[int]] = {}
    for i, d in enumerate(diarization_raw):
        by_speaker.setdefault(d["speaker"], []).append(i)

    clusters: list[dict] = []
    for spk, idxs in by_speaker.items():
        idxs = sorted(idxs, key=lambda i: diarization_raw[i]["start_ms"])
        cur_idxs: list[int] = []
        cur_ms = 0
        for i in idxs:
            iv = diarization_raw[i]
            cur_idxs.append(i)
            cur_ms += iv["end_ms"] - iv["start_ms"]
            if cur_ms >= RAW_CLUSTER_MIN_MS:
                clusters.append({"speaker": spk, "interval_indices": list(cur_idxs)})
                cur_idxs = []
                cur_ms = 0
        if cur_idxs:
            if clusters and clusters[-1]["speaker"] == spk:
                clusters[-1]["interval_indices"].extend(cur_idxs)
            # altrimenti: troppo poco audio per questa etichetta in tutto il
            # segmento, nessun embedding — stesso comportamento di oggi.

    # Costruisci i crop (concatenato degli intervalli di ogni cluster) prima
    # di chiamare il modello — stessa logica di _embed_intervals, ripetuta qui
    # perché serve il crop grezzo per raggrupparli per durata sotto.
    sr = 16000
    crops: list[torch.Tensor] = []
    valid_clusters: list[dict] = []
    for c in clusters:
        pieces = []
        for i in c["interval_indices"]:
            iv = diarization_raw[i]
            s = max(0, int(iv["start_ms"] / 1000 * sr))
            e = min(waveform.shape[1], int(iv["end_ms"] / 1000 * sr))
            if e > s:
                pieces.append(waveform[:, s:e])
        if not pieces:
            continue
        crop = torch.cat(pieces, dim=1)
        if crop.shape[1] < sr // 2:   # < 0.5s, stessa soglia di _embed_intervals
            continue
        crops.append(crop)
        valid_clusters.append(c)

    if not crops:
        return []

    embeddings = _embed_batch_grouped(crops)

    out: list[dict] = []
    for c, emb in zip(valid_clusters, embeddings):
        if emb is None:
            continue
        intervals = [(diarization_raw[i]["start_ms"], diarization_raw[i]["end_ms"])
                     for i in c["interval_indices"]]
        out.append({
            "speaker":          c["speaker"],
            "interval_indices": c["interval_indices"],
            "start_ms":         intervals[0][0],
            "end_ms":           intervals[-1][1],
            "embedding":        emb,
        })
    return out


# Risoluzione dei bucket di durata (in campioni, 16kHz) — vedi _embed_batch_grouped.
# Misurato empiricamente il 2026-08-04 (script scratch/calibrate_ratio.py):
# il parametro `weights` di WeSpeakerResNet293.forward NON esclude
# correttamente il padding dal pooling statistico come la sua documentazione
# lascerebbe intendere — anche un solo frame di differenza tra il crop più
# corto e il più lungo nello stesso batch (rapporto 1.05, cioè 5%) produce
# similarità 0.996 col risultato singolo, ben sotto la soglia di sicurezza
# (0.999). L'ipotesi del padding mascherato è quindi scartata: si raggruppa
# SOLO se le lunghezze sono già identiche (troncando i crop più lunghi alla
# lunghezza del più corto del gruppo, mai riempiendo con zeri) — verificato
# dare similarità 1.00000 esatta quando il rapporto è 1.0.
_EMBED_BUCKET_SAMPLES = 8000  # 500ms a 16kHz — stessa risoluzione di RAW_CLUSTER_MIN_MS


def _embed_batch_grouped(crops: list[torch.Tensor]) -> list[list[float] | None]:
    """Embedding di N crop in meno passaggi di GPU invece di N (2026-08-04).

    Raggruppa SOLO crop la cui durata cade nello stesso bucket da
    _EMBED_BUCKET_SAMPLES campioni, troncando ogni crop del gruppo alla
    lunghezza del bucket (mai al massimo del gruppo, mai con padding) — un
    solo passaggio di GPU senza nessuna differenza di lunghezza da
    compensare, quindi nessun rischio di distorsione (vedi nota sopra sul
    parametro `weights` scartato). Un crop senza compagni della stessa durata
    resta da solo: nessun guadagno possibile per lui, nessun compromesso
    sulla qualità nemmeno per lui.

    Costo: fino a _EMBED_BUCKET_SAMPLES (0.5s) di coda troncata sui crop più
    lunghi del loro bucket — trascurabile per l'identità (il minimo già
    accettato altrove è 0.5s totali).

    Ritorna una lista nello stesso ordine di `crops` (non di elaborazione
    interna)."""
    n = len(crops)
    buckets: dict[int, list[int]] = {}
    for i, c in enumerate(crops):
        key = c.shape[1] // _EMBED_BUCKET_SAMPLES
        buckets.setdefault(key, []).append(i)

    results: list[list[float] | None] = [None] * n
    for key, idxs in buckets.items():
        target_len = key * _EMBED_BUCKET_SAMPLES
        if len(idxs) == 1 or target_len < 16000 // 2:
            # gruppo singolo, o sotto la soglia minima di 0.5s se troncato:
            # ogni membro va per conto suo, sulla sua lunghezza reale intera.
            for i in idxs:
                results[i] = _embed_single(crops[i])
            continue
        truncated = [crops[i][:, :target_len] for i in idxs]
        embs = _embed_group_same_length(truncated)
        for i, emb in zip(idxs, embs):
            results[i] = emb
    return results


def _embed_group_same_length(group: list[torch.Tensor]) -> list[list[float] | None]:
    """Un solo passaggio del modello per crop GIÀ della stessa identica
    lunghezza — nessun padding, nessun parametro `weights` necessario.

    Fallback: se qualcosa va storto, richiama _embed_single uno per uno sullo
    stesso gruppo — mai silenziosamente sbagliato, solo più lento per quel
    gruppo specifico."""
    dev = next(_voiceprint_model.parameters()).device
    try:
        batch = torch.stack([c.to(dev) for c in group], dim=0)  # (n, 1, samples)
        with torch.no_grad():
            embs = _voiceprint_model(batch)
        return [e.cpu().tolist() for e in embs]
    except Exception as exc:
        logger.warning(
            "Voiceprint batch (%d crop, stessa lunghezza) fallito, fallback a chiamate singole: %s",
            len(group), exc,
        )
        return [_embed_single(c) for c in group]


def _embed_single(crop: torch.Tensor) -> list[float] | None:
    """Stesso calcolo di _embed_intervals ma già sul crop pronto (concatenato,
    non ancora verificato) — usato dal fallback di _embed_group."""
    dev = next(_voiceprint_model.parameters()).device
    try:
        with torch.no_grad():
            emb = _voiceprint_model(crop.to(dev).unsqueeze(0))
        return emb[0].cpu().tolist()
    except Exception as exc:
        logger.warning("Voiceprint singolo fallito: %s", exc)
        return None


def _turn_diar_intervals(
    diar_by_spk: dict[str, list[tuple[int, int]]],
    speaker: str,
    start_ms: int,
    end_ms: int,
) -> list[list[int]]:
    """Intervalli acustici pyannote di `speaker` che toccano [start_ms, end_ms].

    Gli estremi restano quelli di pyannote (è il punto: sono i confini veri,
    non quelli schiacciati di whisper), tagliati solo oltre _DIAR_MARGIN_MS
    dalla finestra del turno.
    """
    lo = start_ms - _DIAR_MARGIN_MS
    hi = end_ms + _DIAR_MARGIN_MS
    out: list[list[int]] = []
    for s, e in diar_by_spk.get(speaker, ()):
        if e <= start_ms or s >= end_ms:      # nessuna sovrapposizione col turno
            continue
        out.append([max(s, lo), min(e, hi)])
    return out


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


# Soglie del taglio mirato.
#
# SPLIT_MIN_MERGED era 9 (2026-07-29 mattina): il taglio guardava solo i turni
# GRAVEMENTE fusi, per prudenza dopo la frammentazione del tentativo del 28/07.
# Ritirato la sera stessa, misurando il segmento golden 0fc60ef2: il turno #5
# fonde tre battute — Alex «Ma tutti nei contratti nuovi c'ha già quella cosa
# lì», l'utente «Non so se è una questione d'età o...», Alex «Eh, prima hai
# detto d'età, adesso hai detto un po'» — e ha n_segments_merged=2, quindi non
# veniva nemmeno guardato. Lo speaker per parola le separava correttamente:
# l'informazione c'era, la buttava via il voto di maggioranza per segmento.
#
# Il vincolo sulla fusione era il filtro sbagliato. Quelli giusti sono i due
# sotto, che misurano la SOSTANZA del cambio di parlante invece della sua
# posizione. Misurato sul golden togliendo il vincolo: 17 turni → 26, con i
# tre pezzi del #5 separati giusti, la battuta «Dici se li avessi investiti?»
# recuperata da dentro 53s di monologo, e nella discussione fitta finale solo
# 6 sequenze su 18 promosse a turno — le altre riassorbite. Nessuna traccia
# del 30% di turni da 1-2 parole che aveva affossato il tentativo del 28/07.
# Le due soglie sono in OR, non in AND (corretto 2026-07-29 sera): una battuta
# vera può essere breve — «Ci puoi mettere quello che vuoi» dura 0.8s ma sono 5
# parole, «Ah, ho capito cosa vuol dire» 1.2s e 7 parole. Pretendendo entrambe
# le condizioni venivano riassorbite, e con loro finiva nel turno del vicino
# anche la voce di chi le aveva dette. Misurato sul golden 0fc60ef2:
#
#   1500ms E 4 parole → 18 turni, 8 con dentro più voci
#   1500ms O 4 parole → 25 turni, 3 con dentro più voci
#
# Serve ancora che almeno UNA delle due tenga: chi non ha né durata né parole
# («come», «credere», «l'1%») resta un frammento e viene riassorbito, che è
# esattamente ciò che il filtro deve continuare a fare.
#
# 1500/4 → 700/2 (2026-07-29 sera). Misurato su quattro segmenti di contesti
# acustici diversi, contando i turni che contengono parole di PIÙ parlanti —
# il difetto vero, perché è lì che una voce finisce dentro il turno di un altro:
#
#   contesto              1500/4        1000/3        700/2
#   documentario in TV    10 turni,1    10 turni,1    10 turni,1
#   bar all'aperto 1      25 turni,9    31 turni,10   32 turni,9
#   bar all'aperto 2      31 turni,14   33 turni,12   41 turni,8
#   pub, botta e risposta 78 turni,21   89 turni,17   98 turni,11
#
# Il documentario NON CAMBIA a nessuna soglia: dove il parlato è pacato non
# c'è niente da spezzare, quindi una maglia più fine non costa nulla. Dove è
# concitato dimezza i turni misti. Per questo la soglia è unica e non adattiva
# al contesto: l'adattività viene già dal contenuto.
#
# Il prezzo sono più turni brevi — al pub quelli sotto il secondo passano da 13
# a 23 — ed è accettabile perché le due decisioni sono separate: un turno breve
# serve comunque a dire CHI ha detto quella frase, mentre per l'IDENTITÀ non
# conta, perché sotto il mezzo secondo di parlato utile _embed_intervals non
# calcola nemmeno l'embedding e il turno arriva a valle come voce ignota.
#
# Limite noto: sotto i 700ms e con una parola sola la sequenza resta assorbita.
# Nel campione del pub è il caso di «pericoli» (0.48s), un'interiezione dentro
# la frase dell'altro. Recuperarla richiederebbe di promuovere ogni singola
# parola, che è il tentativo fallito del 28/07.
SPLIT_MIN_RUN_MS   = 700    # abbastanza lunga da essere un turno anche se dice poco
SPLIT_MIN_RUN_WORDS = 2     # oppure abbastanza parole da esserlo anche se è veloce


def _split_fused_turns(
    turns: list[dict],
    logprobs: list[list[float]],
    nospeechs: list[list[float]],
    word_counts: list[int],
    word_scores: list[list[float]],
    compression_ratios: list[list[float]],
    word_segments: list[dict],
    segments: list[dict],
) -> tuple[list[dict], list[list[float]], list[list[float]], list[int],
           list[list[float]], list[list[float]]]:
    """Spezza i soli turni gravemente fusi nei punti in cui il parlante cambia
    **in modo sostenuto**, secondo lo speaker per parola di assign_word_speakers.

    I metriche per segmento (logprob, no_speech, compression_ratio) non sono
    ripartibili sui sotto-turni: un sotto-turno copre una porzione di segmenti
    che non conosciamo. Vengono ereditate dal turno padre, ed è
    un'approssimazione dichiarata — quelle grandezze servono a valutare il
    TESTO, che nel padre e nei figli è lo stesso materiale. Le grandezze che
    invece cambiano davvero (conteggio parole, punteggio di allineamento,
    n_segments_merged) sono ricalcolate sulle parole del sotto-turno.
    """
    out_turns: list[dict] = []
    out_lp: list[list[float]] = []
    out_ns: list[list[float]] = []
    out_wc: list[int] = []
    out_ws: list[list[float]] = []
    out_cr: list[list[float]] = []
    n_split = 0

    for i, turn in enumerate(turns):
        words = [
            w for w in word_segments
            if w.get("speaker") is not None
            and turn["start_ms"] <= int(w.get("start", 0) * 1000) < turn["end_ms"]
        ]

        # Serve solo abbastanza materiale perché possano esistere due sequenze
        # valide: sotto, non c'è niente da tagliare comunque.
        if len(words) < SPLIT_MIN_RUN_WORDS * 2:
            out_turns.append(turn)
            out_lp.append(logprobs[i]); out_ns.append(nospeechs[i])
            out_wc.append(word_counts[i]); out_ws.append(word_scores[i])
            out_cr.append(compression_ratios[i])
            continue

        # Sequenze consecutive di parole con lo stesso parlante.
        runs: list[list[dict]] = []
        for w in words:
            if runs and runs[-1][0].get("speaker") == w.get("speaker"):
                runs[-1].append(w)
            else:
                runs.append([w])

        # Riassorbe le sequenze troppo corte nella precedente: sono le
        # alternanze spurie che avevano rovinato il tentativo del 28/07.
        #
        # Il parlante di una sequenza va portato ESPLICITO, non riletto da
        # run[0]: una sequenza corta assorbita in testa cambierebbe la prima
        # parola e quindi l'identità apparente di tutta la sequenza. È il
        # difetto che il 2026-07-29 teneva insieme tre battute di due persone
        # nel segmento 0fc60ef2 — un solo "Ma" di 0.16s assorbito in testa alla
        # frase di Alex la faceva sembrare dell'utente, e la ricucitura qui
        # sotto la fondeva con la frase successiva dell'utente, inghiottendo
        # nel mezzo tutto ciò che aveva detto Alex.
        merged_runs: list[tuple[str, list[dict]]] = []   # (parlante, parole)
        pending: list[dict] = []                 # sequenze corte prima della prima valida
        for run in runs:
            dur_ms = int(run[-1].get("end", 0) * 1000) - int(run[0].get("start", 0) * 1000)
            # frammento solo se non regge NÉ per durata NÉ per parole
            too_short = dur_ms < SPLIT_MIN_RUN_MS and len(run) < SPLIT_MIN_RUN_WORDS
            if too_short:
                if merged_runs:
                    merged_runs[-1][1].extend(run)
                else:
                    # Nessuna sequenza valida ancora: si accoda a quella che
                    # verrà. Prima diventava un turno a sé, ed è così che sul
                    # segmento 06586ffb è uscito un frammento da 0.9s e 4 parole
                    # — esattamente ciò che il filtro doveva impedire.
                    pending.extend(run)
            else:
                parlante = run[0].get("speaker")
                if pending:
                    run = pending + run
                    pending = []
                merged_runs.append((parlante, run))
        if pending:
            if merged_runs:
                merged_runs[0] = (merged_runs[0][0], pending + merged_runs[0][1])
            else:
                # turno fatto di soli frammenti
                merged_runs.append((pending[0].get("speaker"), pending))

        # Ricucitura: riassorbire una sequenza spuria lascia due tratti dello
        # STESSO parlante separati dal buco che si è appena chiuso. Senza questo
        # passaggio il taglio produrrebbe frammentazione al posto di ripararla —
        # è il modo in cui il tentativo del 28/07 si autosabotava.
        coalesced: list[tuple[str, list[dict]]] = []
        for parlante, run in merged_runs:
            if coalesced and coalesced[-1][0] == parlante:
                coalesced[-1][1].extend(run)
            else:
                coalesced.append((parlante, run))
        merged_runs = [(p, sorted(r, key=lambda w: w.get("start", 0)))
                       for p, r in coalesced]

        if len(merged_runs) < 2:
            out_turns.append(turn)
            out_lp.append(logprobs[i]); out_ns.append(nospeechs[i])
            out_wc.append(word_counts[i]); out_ws.append(word_scores[i])
            out_cr.append(compression_ratios[i])
            continue

        n_split += 1
        for parlante, run in merged_runs:
            scores = [w["score"] for w in run if w.get("score") is not None]
            s_ms = int(run[0].get("start", 0) * 1000)
            e_ms = int(run[-1].get("end", 0) * 1000)
            # n_segments_merged va ricalcolato sul sotto-turno: ereditare quello
            # del padre (anche 29) lo farebbe marcare come fuso proprio dopo
            # averlo separato, che è il contrario di ciò che stiamo facendo.
            n_seg = sum(
                1 for sg in segments
                if int(sg.get("start", 0) * 1000) < e_ms
                and int(sg.get("end", 0) * 1000) > s_ms
            )
            out_turns.append({
                # il parlante DELLA SEQUENZA, non quello della prima parola:
                # in testa può esserci un frammento assorbito di un altro
                "speaker":  parlante or turn["speaker"],
                "start_ms": s_ms,
                "end_ms":   e_ms,
                "text":     " ".join(w.get("word", "") for w in run).strip(),
                "_n_segments_merged": max(1, n_seg),
                "split_from_fused": True,
            })
            out_lp.append(list(logprobs[i]))
            out_ns.append(list(nospeechs[i]))
            out_cr.append(list(compression_ratios[i]))
            out_wc.append(len(run))
            out_ws.append(scores)

    if n_split:
        logger.info(
            "split per cambio parlante: %d turni spezzati, %d → %d turni totali",
            n_split, len(turns), len(out_turns),
        )
    return out_turns, out_lp, out_ns, out_wc, out_ws, out_cr


def _to_contract(
    wx_result: dict,
    audio_np: np.ndarray,   # float32 mono 16kHz
    language: str,
    speaker_embeddings: dict | None = None,
    diarize_stats: dict | None = None,
    diarize_df=None,        # DataFrame pyannote (start/end in secondi, speaker)
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

    # Archivio grezzo (2026-08-04): i confini dei segmenti whisper, PRIMA di
    # qualunque voto di maggioranza o taglio — mancava dal pacchetto di §14.
    # Scoperto rileggendo assign_word_speakers: la maggior parte delle parole
    # in word_timestamps non ricade in NESSUN intervallo di pyannote e resta
    # senza `speaker` — è il voto di maggioranza per segmento whisper (fatto
    # qui sotto, su TUTTE le parole del segmento comprese quelle senza
    # etichetta diretta) a "prestare" loro il parlante del segmento. Senza
    # questi confini, ricostruire i turni da word_timestamps da solo perde
    # sistematicamente quelle parole — non è ricostruibile a posteriori
    # dall'audio, che a quel punto è già cancellato.
    whisper_segments: list[dict] = []
    for seg in segments:
        spk_ws = seg.get("speaker", "SPEAKER_00")
        lp_ws = seg.get("avg_logprob")
        if lp_ws is None:
            lp_ws = -0.5
        ns_ws = seg.get("no_speech_prob")
        if ns_ws is None:
            ns_ws = 0.1
        whisper_segments.append({
            "speaker":           spk_ws,
            "start_ms":          int(seg.get("start", 0) * 1000),
            "end_ms":            int(seg.get("end", 0) * 1000),
            "text":              seg.get("text", "").strip(),
            "avg_logprob":       round(float(lp_ws), 4),
            "no_speech_prob":    round(float(ns_ws), 4),
            "compression_ratio": round(float(seg.get("compression_ratio", 1.0)), 4),
        })

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

    # ── Taglio ai cambi di parlante sostenuti (2026-07-29) ───────────────────
    # Il voto di maggioranza qui sopra decide UN parlante per segmento whisper,
    # e quando in un segmento si alternano due voci la minoranza sparisce:
    # sul segmento golden 0fc60ef2 una frase intera dell'utente veniva
    # inghiottita in un turno di Alex, 19 parole contro 8. Lo speaker per
    # parola le distingueva correttamente — l'informazione c'era già.
    #
    # Il 2026-07-28 avevamo provato a costruire TUTTI i turni per parola:
    # troppa frammentazione (30% dei turni a 1-2 parole, alternanze A-B-A
    # spurie), annullato. Poi il taglio sui soli turni gravemente fusi
    # (n_segments_merged >= 9), che però non guardava proprio i casi come
    # quello sopra (quel turno ne aveva 2).
    #
    # Ora il taglio vale per tutti i turni, e a filtrare sono le SOGLIE DI
    # SOSTANZA (SPLIT_MIN_RUN_MS / _WORDS): una sequenza che non dura o non ha
    # parole viene riassorbita nel vicino invece di diventare un turno. È
    # sempre stato quello il filtro giusto — il vincolo sulla fusione guardava
    # la posizione del difetto invece della sua sostanza.
    #
    # Il taglio avviene prima del ciclo degli embedding, così ogni sotto-turno
    # riceve il proprio embedding sui confini nuovi invece di ereditare la media.
    speaker_turns, turn_logprobs, turn_nospeechs, turn_word_counts, \
        turn_word_scores, turn_compression_ratios = _split_fused_turns(
            speaker_turns, turn_logprobs, turn_nospeechs, turn_word_counts,
            turn_word_scores, turn_compression_ratios,
            wx_result.get("word_segments", []), segments,
        )

    # Intervalli acustici grezzi di pyannote, indicizzati per etichetta.
    # Fin qui venivano buttati dopo assign_word_speakers: l'unica griglia
    # temporale che sopravviveva era quella testuale di whisper, schiacciata
    # ai cambi di parlante. Da oggi ogni turno porta anche i SUOI intervalli
    # acustici (diar_intervals) e l'embedding è calcolato su quelli.
    diar_by_spk: dict[str, list[tuple[int, int]]] = {}
    if diarize_df is not None:
        try:
            for _, r in diarize_df.iterrows():
                diar_by_spk.setdefault(str(r["speaker"]), []).append(
                    (int(float(r["start"]) * 1000), int(float(r["end"]) * 1000)))
            for v in diar_by_spk.values():
                v.sort()
        except Exception as exc:
            logger.warning("Intervalli pyannote non indicizzabili: %s", exc)
            diar_by_spk = {}

    # Archivio grezzo (2026-08-03): TUTTI gli intervalli di pyannote, prima di
    # qualunque incrocio con turni/parole — non buttati via come accadeva finora
    # (qui sopravvivevano solo statistiche aggregate, vedi _log_diarization_stats).
    # Permette a Lifelog2 di ricostruire/ritarare la logica di costruzione turni
    # (oggi _split_fused_turns) senza mai richiamare WhisperX su questo audio.
    diarization_raw: list[dict] = []
    if diarize_df is not None:
        try:
            for _, r in diarize_df.iterrows():
                diarization_raw.append({
                    "start_ms": int(float(r["start"]) * 1000),
                    "end_ms":   int(float(r["end"]) * 1000),
                    "speaker":  str(r["speaker"]),
                })
            diarization_raw.sort(key=lambda d: d["start_ms"])
        except Exception as exc:
            logger.warning("diarization_raw non serializzabile: %s", exc)
            diarization_raw = []

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
        turn["n_segments_merged"]      = turn.pop("_n_segments_merged", len(turn_logprobs[i]))
        turn["avg_word_score"]         = round(sum(scores) / len(scores), 4) if scores else None
        turn["avg_compression_ratio"]  = round(sum(crs) / len(crs), 4)

        # Embedding: sugli intervalli acustici quando esistono, altrimenti
        # sulla finestra whisper (fallback, ed embedding_source lo dichiara).
        intervalli = _turn_diar_intervals(
            diar_by_spk, turn["speaker"], turn["start_ms"], turn["end_ms"])
        emb = None
        if intervalli:
            emb = _embed_intervals(waveform, [(s, e) for s, e in intervalli])
        if emb is not None:
            turn["embedding_source"] = "diar"
        else:
            emb = _embed_turn(waveform, turn["start_ms"], turn["end_ms"])
            turn["embedding_source"] = "window" if emb is not None else None
        turn["embedding"]     = emb
        turn["vp_confidence"] = (
            _vp_confidence(lp, dur_s, wc, nsp) if emb is not None else 0.0
        )
        if intervalli:
            turn["diar_intervals"]    = intervalli
            turn["acoustic_start_ms"] = intervalli[0][0]
            turn["acoustic_end_ms"]   = max(e for _, e in intervalli)

        # ── Due misure per il consumatore: «posso fidarmi di CHI ha parlato?» ──
        #
        # Sono separate dalla qualità del testo, e servono a non farsi ingannare
        # sulla provenienza (Lifelog2 le usa per dare una voce GENERICA invece di
        # un'identità specifica quando non c'è materiale per deciderla).
        #
        # usable_audio_ms: quanto parlato di QUESTA voce è entrato davvero
        #   nell'embedding. Non coincide con la durata del turno: gli intervalli
        #   acustici possono sporgere oltre la finestra testuale, quindi un turno
        #   da 0.36s può avere 1.4s di audio utile — meglio così per l'identità,
        #   ma va dichiarato perché testo ed embedding coprono finestre diverse.
        #
        # speaker_purity: frazione della durata delle parole del turno il cui
        #   parlante per-parola coincide con quello del turno. Sotto 1.0 il turno
        #   contiene parole di qualcun altro — accade nei botta e risposta serrati,
        #   dove l'assegnazione per parola sbaglia sui confini. Misurato: nel
        #   campione del pub 11 turni su 98 sono impuri, il peggiore a 0.37.
        turn["usable_audio_ms"] = sum(e - s for s, e in intervalli) if intervalli else 0

        parole_turno = [
            w for w in wx_result.get("word_segments", [])
            if w.get("speaker") is not None
            and turn["start_ms"] <= int(w.get("start", 0) * 1000) < turn["end_ms"]
        ]
        if parole_turno:
            tot_ms = sum(int((w.get("end", 0) - w.get("start", 0)) * 1000)
                         for w in parole_turno) or 1
            ok_ms = sum(int((w.get("end", 0) - w.get("start", 0)) * 1000)
                        for w in parole_turno if w.get("speaker") == turn["speaker"])
            turn["speaker_purity"] = round(ok_ms / tot_ms, 3)
        else:
            turn["speaker_purity"] = None

    # Embedding per cluster di intervalli grezzi (2026-08-03) — vedi
    # _embed_raw_clusters: unità stabile, indipendente dalla logica di
    # costruzione turni, così Lifelog2 può ricombinarli per qualunque
    # raggruppamento scelga in futuro senza mai richiamare GPU o audio grezzo.
    diarization_embeddings = _embed_raw_clusters(waveform, diarization_raw)

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
    #
    # Guardia NaN (2026-07-30): su audio senza voce rilevata, pyannote produce
    # comunque un intervallo fittizio per un "parlante" fantasma, e il suo
    # pooling statistico interno (bug noto e non risolto upstream, pyannote-
    # audio issue #1861 — std() con correzione di Bessel su una finestra di un
    # solo frame) può restituire un embedding con componenti NaN/Inf. Prima
    # veniva incluso così com'era e rompeva la serializzazione JSON della
    # risposta (ValueError: Out of range float values are not JSON compliant),
    # facendo fallire l'intera trascrizione con un 500 anche quando testo e
    # turni erano perfettamente vuoti/corretti. Ora lo scartiamo: il segmento
    # risulterà semplicemente senza embedding per quel parlante.
    speaker_embeddings_out: dict | None = None
    if speaker_embeddings:
        speaker_embeddings_out = {}
        for spk, emb in speaker_embeddings.items():
            try:
                vals = [round(float(x), 6) for x in emb]
            except Exception:
                continue
            if not all(math.isfinite(v) for v in vals):
                logger.info(
                    "Embedding parlante %s scartato (valori non validi, NaN/Inf) — "
                    "normale su audio senza voce rilevata, non influisce sulla "
                    "trascrizione: il parlante resterà semplicemente senza voiceprint.",
                    spk,
                )
                continue
            speaker_embeddings_out[str(spk)] = vals

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
    # Pacchetto grezzo archiviabile (2026-08-03, vedi _embed_raw_clusters):
    # tutti gli intervalli pyannote + un embedding per cluster minimo
    # embeddabile, indipendenti da _split_fused_turns e da qualunque logica
    # di costruzione turni Lifelog2 scelga in futuro — permette di ritarare
    # quella logica senza mai ririlanciare WhisperX su questo audio.
    if diarization_raw:
        out["diarization_raw"] = diarization_raw
    if diarization_embeddings:
        out["diarization_embeddings"] = diarization_embeddings
    # Segmenti grezzi di whisper (2026-08-04, vedi nota sopra dove vengono
    # costruiti) — l'ultimo pezzo mancante del pacchetto: senza i suoi confini
    # e il suo parlante di maggioranza, le parole senza `speaker` diretto in
    # word_timestamps non sono riattribuibili a posteriori.
    if whisper_segments:
        out["whisper_segments"] = whisper_segments
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
                              diarize_stats=diarize_stats,
                              diarize_df=diarize_segs)
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
