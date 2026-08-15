"""
ARIA FLUX.2-klein-4B Image Generation Server — FastAPI on port 8092

Architecture:
  - Flux2KleinPipeline (diffusers ≥0.39.0.dev0)
  - Text encoder: Qwen3-4B loaded BF16, quantized INT8 via optimum-quanto (~3.75 GB VRAM)
  - Transformer: BF16 (~6.5 GB VRAM)
  - VAE: BF16 (~0.5 GB VRAM)
  - Total: ~12.8 GB VRAM, ~3.2 GB headroom on RTX 5060 Ti 16 GB

Blackwell SM_120 notes:
  - No Flash Attention (lacks TMA/UTMA), no xformers
  - BF16 is fastest; FP8 slower on consumer Blackwell
  - PyTorch 2.7.0 cu128 first stable SM_120 support

JIT pattern: loaded on startup (lifespan), unloaded on shutdown.
Output: JPEG saved to ARIA_OUTPUT_DIR, served via asset server (port 8082).

2026-08-15: aggiunto heartbeat + traceback completo attorno a ogni fase di
_load_models() — prima un'eccezione in from_pretrained() (o un blocco senza
eccezione, es. I/O lento) spariva nel silenzio: l'unico log era "Loading
Flux2KleinPipeline from ..." seguito da niente, per minuti, senza dire se il
processo stava ancora lavorando o era già morto. Vedi indagine 2026-08-15
sui crash ricorrenti di ricaricamento dopo uno swap GPU Exclusivity.

2026-08-15 (2): self-reporting del PID reale su file (vedi PID_FILE sotto).
Causa radice trovata in aria_node_controller/core/orchestrator.py: il
processo lanciato con 'start "titolo" cmd.exe /k ...' fa sì che l'oggetto
Popen tracciato da ARIA sia il lanciatore 'start', che termina da solo
entro pochi secondi — rendendo proc.poll() inaffidabile per sapere se
QUESTO backend è ancora vivo. _kill_proc() controllava proc.poll() prima
di tentare il kill: quasi sempre falso per un processo idle da tempo,
quindi il kill non partiva mai e ARIA si "dimenticava" del backend senza
ucciderlo — VRAM/RAM restavano occupate. Scrivendo qui il PID vero (+
create_time, per proteggersi da riuso del PID) su file, l'orchestratore può
verificare/uccidere il processo esatto invece di affidarsi al Popen rotto o
al match per titolo finestra con wildcard (rischio di colpire finestre con
titolo simile).
"""

import os
import gc
import io
import time
import json
import logging
import random
import threading
from pathlib import Path

# 2026-08-15: crash ripetuti (5+ occorrenze) in torch_cpu.dll, sempre stesso
# offset esatto (0x6019bb4), codice 0xc0000005 (access violation) — firma
# identica ogni volta = bug deterministico nel path di threading nativo di
# PyTorch (OpenMP/MKL), non corruzione di memoria casuale. Va impostato
# PRIMA di 'import torch': il runtime OpenMP/MKL inizializza il pool di
# thread all'import, torch.set_num_threads() da solo non basta a vincolarlo
# retroattivamente. Macchina ha 12 processori logici — 4 è un compromesso
# prudente (il caricamento è comunque solo qualche secondo più lento, non
# tocca l'inferenza che gira su GPU). Se il crash persiste anche con questo,
# il sospetto si sposta da "race condition di threading" a un problema più
# di fondo (versione torch/CPU incompatibile) — vedi nota in _load_models.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import psutil
import torch

torch.set_num_threads(4)

LOG_FILE = r"C:\Users\Roberto\aria\logs\flux_imagegen.log"
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

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

load_dotenv()

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE    = torch.bfloat16

ARIA_ROOT      = Path(os.environ.get("ARIA_ROOT", r"C:\Users\Roberto\ARIA"))
MODEL_PATH     = ARIA_ROOT / "data" / "assets" / "models" / "flux2-klein-4b"
ARIA_OUTPUT_DIR = ARIA_ROOT / "data" / "outputs"

# Self-reporting del PID reale per l'orchestratore (vedi nota in cima al file).
# Stessa convenzione di nome che orchestrator.py si aspetta: {model_id}.pid
# dentro aria_root/logs/pids/ — "flux2-klein-4b" è il model_id nel manifest.
PID_DIR  = Path(r"C:\Users\Roberto\aria\logs\pids")
PID_FILE = PID_DIR / "flux2-klein-4b.pid"

_pipe = None


def _write_pid_file() -> None:
    try:
        PID_DIR.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        create_time = psutil.Process(pid).create_time()
        PID_FILE.write_text(json.dumps({"pid": pid, "create_time": create_time}))
        logger.info("PID file scritto: %s (pid=%d)", PID_FILE, pid)
    except Exception:
        logger.exception("Errore scrivendo il PID file %s", PID_FILE)


def _remove_pid_file() -> None:
    try:
        PID_FILE.unlink(missing_ok=True)
        logger.info("PID file rimosso: %s", PID_FILE)
    except Exception:
        logger.exception("Errore rimuovendo il PID file %s", PID_FILE)


class _Heartbeat:
    """Logga un battito ogni interval_s finché il blocco `with` non esce —
    senza questo, una fase lenta (I/O disco, allocazione RAM/VRAM) è
    indistinguibile nel log da un processo già morto in silenzio."""

    def __init__(self, label: str, t0: float, interval_s: float = 5.0):
        self.label = label
        self.t0 = t0
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval_s):
            logger.info(
                "%s: ancora in corso... %.1fs trascorsi",
                self.label, time.time() - self.t0,
            )

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=1.0)
        return False


def _log_resource_snapshot(prefix: str) -> None:
    """RAM libera + VRAM allocata/riservata, per correlare rallentamenti coi
    log di sistema (indagine 2026-08-15: sospetto pressione RAM/VRAM non
    rilasciata dopo un kill di GPU Exclusivity, non ancora confermato)."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_msg = f"RAM libera: {vm.available / 1e9:.1f}/{vm.total / 1e9:.1f} GB"
    except Exception as exc:
        ram_msg = f"RAM libera: n/d ({exc})"
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        vram_msg = f"VRAM allocata={alloc:.1f}GB riservata={reserved:.1f}GB"
    else:
        vram_msg = "VRAM: n/d (no CUDA)"
    logger.info("%s — %s | %s", prefix, ram_msg, vram_msg)


def _load_models():
    global _pipe
    from diffusers import Flux2KleinPipeline
    from optimum.quanto import quantize, freeze, qint8

    logger.info("Loading Flux2KleinPipeline from %s ...", MODEL_PATH)
    t0 = time.time()
    _log_resource_snapshot("Snapshot risorse prima del caricamento")

    try:
        with _Heartbeat("from_pretrained (lettura pesi da disco + istanziazione componenti)", t0):
            _pipe = Flux2KleinPipeline.from_pretrained(
                str(MODEL_PATH),
                torch_dtype=DTYPE,
                local_files_only=True,
            )
    except Exception:
        logger.exception(
            "Flux2KleinPipeline.from_pretrained FALLITO dopo %.1fs", time.time() - t0
        )
        _log_resource_snapshot("Snapshot risorse al momento del fallimento")
        raise
    logger.info("Pipeline loaded from disk in %.1fs", time.time() - t0)

    # INT8 quantize text encoder (Qwen3-4B) on CPU before moving to GPU:
    # 7.50 GB BF16 → 3.75 GB INT8
    logger.info("Quantizing text encoder (Qwen3-4B) INT8 via optimum-quanto ...")
    t1 = time.time()
    try:
        with _Heartbeat("quantizzazione text encoder", t1):
            quantize(_pipe.text_encoder, weights=qint8)
            freeze(_pipe.text_encoder)
    except Exception:
        logger.exception("Quantizzazione text encoder FALLITA dopo %.1fs", time.time() - t1)
        _log_resource_snapshot("Snapshot risorse al momento del fallimento")
        raise
    logger.info("Text encoder quantized in %.1fs", time.time() - t1)

    t2 = time.time()
    try:
        with _Heartbeat("spostamento pipeline su GPU (.to(cuda))", t2):
            _pipe.to(DEVICE)
    except Exception:
        logger.exception("_pipe.to(%s) FALLITO dopo %.1fs", DEVICE, time.time() - t2)
        _log_resource_snapshot("Snapshot risorse al momento del fallimento")
        raise

    vram = round(torch.cuda.memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0
    logger.info("Pipeline on GPU — VRAM allocated: %.1f GB (total: %.1fs)", vram, time.time() - t0)


def _unload_models():
    global _pipe
    t0 = time.time()
    try:
        del _pipe
        _pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Models unloaded, VRAM freed. (%.1fs)", time.time() - t0)
    except Exception:
        logger.exception("Errore durante _unload_models dopo %.1fs", time.time() - t0)
        raise
    finally:
        _log_resource_snapshot("Snapshot risorse dopo unload")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Scritto PRIMA di _load_models(): così l'orchestratore vede che il
    # processo esiste (magari ancora in caricamento) anche se il caricamento
    # stesso si blocca o fallisce — è esattamente il segnale mancante finora.
    _write_pid_file()
    try:
        _load_models()
        yield
    finally:
        _unload_models()
        _remove_pid_file()


app = FastAPI(title="ARIA FLUX.2-klein ImageGen", version="1.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    vram = round(torch.cuda.memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0.0
    return {
        "status": "ok",
        "model":  "flux2-klein-4b",
        "device": DEVICE,
        "vram_gb": vram,
        "ready": _pipe is not None,
    }


class GenerateRequest(BaseModel):
    prompt:          str
    output_filename: str = "output.jpeg"
    width:           int = 512
    height:          int = 512
    steps:           int = 20
    guidance:        float = 3.5
    seed:            int = -1


@app.post("/generate")
def generate(req: GenerateRequest):
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    ARIA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARIA_OUTPUT_DIR / req.output_filename

    t0 = time.perf_counter()
    seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    logger.info(
        "generate — seed=%d steps=%d size=%dx%d out=%s prompt='%.80s'",
        seed, req.steps, req.width, req.height, req.output_filename, req.prompt,
    )

    try:
        result = _pipe(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            generator=generator,
        )
        image = result.images[0]
    except Exception as exc:
        logger.error("Generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    image.save(str(out_path), format="JPEG", quality=90)
    elapsed = round(time.perf_counter() - t0, 2)
    logger.info("Generated in %.1fs → %s", elapsed, out_path)

    return {
        "output_path":     str(out_path),
        "output_filename": req.output_filename,
        "processing_time": elapsed,
        "seed":            seed,
        "width":           req.width,
        "height":          req.height,
    }


@app.delete("/output/{filename}")
def delete_output(filename: str):
    out_path = ARIA_OUTPUT_DIR / filename
    if out_path.exists():
        out_path.unlink()
        logger.info("Deleted output: %s", out_path)
        return {"deleted": str(out_path)}
    return {"deleted": None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8092, log_level="info")
