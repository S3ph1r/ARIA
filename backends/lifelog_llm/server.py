"""
ARIA — Lifelog LLM Backend (Parametric)

Generic llama.cpp FastAPI server for text enrichment models.
Exposes an OpenAI-compatible /v1/chat/completions API.

Usage:
    python server.py --model-path <path-to.gguf> --port 8089 [--n-ctx 16384]

Called by ARIA orchestrator via backends_manifest.json.
Supports any GGUF text model — currently deployed with Qwen3-14B-Q4_K_M.
"""

import argparse
import logging
import re
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from llama_cpp import Llama
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("aria.lifelog-llm")

app = FastAPI(title="ARIA Lifelog LLM Backend")

# Global model instance — loaded once at startup
_llm: Optional[Llama] = None
_model_id: str = "unknown"


# ── Request / Response schemas ────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "lifelog-llm"
    messages: list[ChatMessage]
    max_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    stream: bool = False
    stop: Optional[list[str]] = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage


# ── Model loader ──────────────────────────────────────────────────────────────


def _load_model(model_path: str, n_ctx: int) -> None:
    global _llm, _model_id
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    _model_id = path.stem
    logger.info("Loading %s (n_ctx=%d)...", _model_id, n_ctx)
    t0 = time.time()

    _llm = Llama(
        model_path=str(path),
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        n_batch=512,
        flash_attn=True,
        type_k=8,
        type_v=8,
        verbose=False,
    )

    logger.info("Model loaded in %.1fs", time.time() - t0)


# ── Chat endpoint (OpenAI-compatible) ─────────────────────────────────────────


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Build ChatML prompt from messages list."""
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _strip_thinking(text: str) -> tuple[str, str]:
    """Extract and remove <think>...</think> block from output."""
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if m:
        thinking = m.group(1).strip()
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        return clean, thinking
    return text.strip(), ""


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest):
    if _llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = _messages_to_prompt(req.messages)
    stop_tokens = req.stop or ["<|im_end|>", "<|endoftext|>"]

    logger.info("Inference request: %d chars prompt, max_tokens=%d", len(prompt), req.max_tokens)

    output = _llm(
        prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        min_p=req.min_p,
        stop=stop_tokens,
        echo=False,
    )

    raw_text = output["choices"][0]["text"]
    clean_text, _thinking = _strip_thinking(raw_text)

    return ChatResponse(
        id=f"chatcmpl-lifelog-{int(time.time())}",
        model=_model_id,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=clean_text),
                finish_reason=output["choices"][0].get("finish_reason", "stop"),
            )
        ],
        usage=ChatUsage(
            prompt_tokens=output["usage"]["prompt_tokens"],
            completion_tokens=output["usage"]["completion_tokens"],
            total_tokens=output["usage"]["total_tokens"],
        ),
    )


# ── Health check ──────────────────────────────────────────────────────────────


@app.get("/v1/health")
async def health():
    if _llm is None:
        return JSONResponse({"status": "loading", "model": None}, status_code=503)
    return {"status": "ready", "model": _model_id, "device": "CUDA"}


@app.get("/health")
async def health_legacy():
    return await health()


# ── Entry point ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to .gguf file")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--n-ctx", type=int, default=16384)
    args = parser.parse_args()

    _load_model(args.model_path, args.n_ctx)
    uvicorn.run(app, host=args.host, port=args.port)
