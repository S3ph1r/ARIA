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
SPLIT_MIN_RUN_MS   = 1500   # un cambio parlante più breve è rumore, non un turno
SPLIT_MIN_RUN_WORDS = 4     # idem: sotto 4 parole non si apre un turno nuovo


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
        merged_runs: list[list[dict]] = []
        pending: list[dict] = []                 # sequenze corte prima della prima valida
        for run in runs:
            dur_ms = int(run[-1].get("end", 0) * 1000) - int(run[0].get("start", 0) * 1000)
            too_short = dur_ms < SPLIT_MIN_RUN_MS or len(run) < SPLIT_MIN_RUN_WORDS
            if too_short:
                if merged_runs:
                    merged_runs[-1].extend(run)
                else:
                    # Nessuna sequenza valida ancora: si accoda a quella che
                    # verrà. Prima diventava un turno a sé, ed è così che sul
                    # segmento 06586ffb è uscito un frammento da 0.9s e 4 parole
                    # — esattamente ciò che il filtro doveva impedire.
                    pending.extend(run)
            else:
                if pending:
                    run = pending + run
                    pending = []
                merged_runs.append(run)
        if pending:
            if merged_runs:
                merged_runs[0] = pending + merged_runs[0]
            else:
                merged_runs.append(pending)      # turno fatto di soli frammenti

        # Ricucitura: riassorbire una sequenza spuria lascia due tratti dello
        # STESSO parlante separati dal buco che si è appena chiuso. Senza questo
        # passaggio il taglio produrrebbe frammentazione al posto di ripararla —
        # è il modo in cui il tentativo del 28/07 si autosabotava.
        coalesced: list[list[dict]] = []
        for run in merged_runs:
            if coalesced and coalesced[-1][0].get("speaker") == run[0].get("speaker"):
                coalesced[-1].extend(run)
            else:
                coalesced.append(run)
        merged_runs = coalesced

        if len(merged_runs) < 2:
            out_turns.append(turn)
            out_lp.append(logprobs[i]); out_ns.append(nospeechs[i])
            out_wc.append(word_counts[i]); out_ws.append(word_scores[i])
            out_cr.append(compression_ratios[i])
            continue

        n_split += 1
        for run in merged_runs:
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
                "speaker":  run[0].get("speaker") or turn["speaker"],
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
