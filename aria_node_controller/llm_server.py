import os
import sys
import json
import logging
import asyncio
import time
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

# === NH-MINI GROUNDED CONFIG ===
# Model: Qwen3.5-35B-A3B-MoE-Q3_K_S
# Path: C:\Users\Roberto\aria\data\models\Qwen3.5-35B-A3B-GGUF
# SM: sm_120 (Integrated via llama-cpp-python CUDA build)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ARIA-LLM-Backend")

app = FastAPI(title="ARIA Qwen3.5 LLM Backend")

class LLMRequest(BaseModel):
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.8
    thinking: bool = True
    stop: Optional[List[str]] = ["<|im_end|>", "<|endoftext|>", "\n\n\n"]

class LLMResponse(BaseModel):
    text: str
    thinking: Optional[str] = None
    usage: Dict[str, int]

# Global LLM instance
llm = None

def load_model():
    global llm
    model_path = r"C:\Users\Roberto\aria\data\models\Qwen3.5-35B-A3B-GGUF\Qwen3.5-35B-A3B-Q3_K_S.gguf"
    
    logger.info(f"Loading model from {model_path}...")
    start_time = time.time()
    
    # Configuration optimized for RTX 5060 Ti 16GB
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,      # All layers to GPU
        n_ctx=32768,          # Large context for DIAS
        n_batch=512,          
        flash_attn=True,      # Enabled for sm_120
        type_k=8,             # 8-bit KV Cache (K)
        type_v=8,             # 8-bit KV Cache (V)
        verbose=True
    )
    
    logger.info(f"Model loaded in {time.time() - start_time:.2f}s")

@app.get("/v1/health")
async def health_check():
    global llm
    if llm is None:
        return {"status": "starting", "model": None}
    return {
        "status": "ready",
        "model": "Qwen3.5-35B-A3B-GGUF",
        "device": "CUDA"
    }

@app.on_event("startup")
async def startup_event():
    # Pre-load model to verify VRAM
    logger.info("Auto-loading model on startup...")
    try:
        load_model()
    except Exception as e:
        logger.error(f"AUTO-LOAD FAILED: {e}")

@app.get("/health")
async def legacy_health():
    return await health_check()

def extract_thinking(text: str) -> (str, Optional[str]):
    """
    Extracts content between <thought> tags (common in Qwen/DeepSeek).
    """
    import re
    thought_match = re.search(r'<thought>(.*?)</thought>', text, re.DOTALL)
    if thought_match:
        thinking = thought_match.group(1).strip()
        # Remove thought section from final text
        clean_text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL).strip()
        return clean_text, thinking
    return text, None

@app.post("/generate", response_model=LLMResponse)
async def generate(request: LLMRequest):
    global llm
    if llm is None:
        try:
            load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail="Model not loaded")

    logger.info(f"Generating for prompt (len={len(request.prompt)})...")
    
    # Process request
    output = llm(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop,
        echo=False
    )
    
    raw_text = output['choices'][0]['text']
    clean_text, thinking_text = extract_thinking(raw_text)
    
    return LLMResponse(
        text=clean_text,
        thinking=thinking_text if request.thinking else None,
        usage=output['usage']
    )

if __name__ == "__main__":
    import uvicorn
    # Using local IP logic from ARIA core if available
    port = 8085
    logger.info(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
