"""
ARIA FLUX.2-klein-4B Image Generation Server — FastAPI on port 8092

Architecture:
  - Flux2KleinPipeline (diffusers ≥0.37.0.dev0)
  - Text encoder: Qwen3-4B loaded BF16, quantized INT8 via optimum-quanto (~3.75 GB VRAM)
  - Transformer: BF16 (~6.5 GB VRAM)
  - VAE: BF16 (~0.5 GB VRAM)
  - Total: ~11 GB VRAM, 5 GB headroom on RTX 5060 Ti 16 GB

Blackwell SM_120 notes:
  - No Flash Attention (lacks TMA/UTMA), no xformers
  - BF16 is fastest; FP8 slower on consumer Blackwell
  - PyTorch 2.7.0 cu128 first stable SM_120 support

JIT pattern: loaded on startup (lifespan), unloaded on shutdown.
Output: PNG saved to MinIO warehouse.
"""

import os
import sys
import gc
import io
import time
import logging
import random

import torch

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
from minio import Minio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

load_dotenv()

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE        = torch.bfloat16

ARIA_ROOT    = os.environ.get("ARIA_ROOT", r"C:\Users\Roberto\ARIA")
MODEL_PATH   = os.path.join(ARIA_ROOT, "data", "assets", "models", "flux2-klein-4b")

MINIO_ENDPOINT   = os.getenv("ARIA_MINIO_ENDPOINT",   "192.168.1.104:9000")
MINIO_ACCESS_KEY = os.getenv("ARIA_MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("ARIA_MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET     = os.getenv("ARIA_MINIO_BUCKET",     "aria-warehouse")

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

_pipe = None


def _load_models():
    global _pipe
    from diffusers import Flux2KleinPipeline
    from optimum.quanto import quantize, freeze, qint8

    logger.info("Loading Flux2KleinPipeline from %s ...", MODEL_PATH)
    t0 = time.time()

    # Load full pipeline to CPU in BF16 — text encoder stays on CPU until quantized
    _pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        local_files_only=True,
    )
    logger.info("Pipeline loaded from disk in %.1fs", time.time() - t0)

    # INT8 quantize text encoder (Qwen3-4B) while still on CPU:
    # 7.50 GB BF16 → 3.75 GB INT8, reduces peak VRAM by ~3.75 GB
    logger.info("Quantizing text encoder (Qwen3-4B) INT8 via optimum-quanto ...")
    t1 = time.time()
    quantize(_pipe.text_encoder, weights=qint8)
    freeze(_pipe.text_encoder)
    logger.info("Text encoder quantized in %.1fs", time.time() - t1)

    # Move entire pipeline to GPU — text encoder already INT8 (~3.75 GB)
    _pipe.to(DEVICE)

    vram = round(torch.cuda.memory_allocated() / 1e9, 1) if torch.cuda.is_available() else 0
    logger.info("Pipeline on GPU — VRAM allocated: %.1f GB (total: %.1fs)", vram, time.time() - t0)


def _unload_models():
    global _pipe
    del _pipe
    _pipe = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Models unloaded, VRAM freed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield
    _unload_models()


app = FastAPI(title="ARIA FLUX.2-klein ImageGen", version="1.0.0", lifespan=lifespan)


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
    prompt:     str
    width:      int = 512
    height:     int = 512
    steps:      int = 20
    guidance:   float = 3.5
    seed:       int = -1           # -1 = random
    output_key: Optional[str] = None   # MinIO object key; if None returns no URL


@app.post("/generate")
def generate(req: GenerateRequest):
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    t0 = time.perf_counter()
    seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    logger.info(
        "generate — seed=%d steps=%d size=%dx%d prompt='%.80s'",
        seed, req.steps, req.width, req.height, req.prompt,
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

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info("Generated in %.1fs", elapsed)

    # Encode to PNG in memory
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    png_bytes = buf.getvalue()

    minio_url = None
    if req.output_key:
        try:
            _ensure_bucket()
            minio_client.put_object(
                MINIO_BUCKET,
                req.output_key,
                io.BytesIO(png_bytes),
                length=len(png_bytes),
                content_type="image/png",
            )
            minio_url = f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{req.output_key}"
            logger.info("Saved to MinIO: %s", minio_url)
        except Exception as exc:
            logger.warning("MinIO upload failed: %s", exc)

    return {
        "status": "done",
        "processing_time": elapsed,
        "output": {
            "seed":      seed,
            "width":     req.width,
            "height":    req.height,
            "steps":     req.steps,
            "minio_url": minio_url,
        },
    }


def _ensure_bucket():
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8092, log_level="info")
