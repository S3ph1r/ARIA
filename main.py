#!/usr/bin/env python3
"""
ARIA - Distributed GPU Inference Broker
Main application entry point
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import torch

# Import API routers
from aria_server.api.tts import router as tts_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('ARIA_LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ARIA GPU Inference Broker",
    description="Distributed GPU inference broker for homelab",
    version="0.1.0"
)

# Include API routers
app.include_router(tts_router)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        return {
            "status": "healthy",
            "gpu_available": gpu_available,
            "gpu_count": gpu_count,
            "cuda_version": torch.version.cuda if gpu_available else None,
            "service": "aria-server",
            "version": "0.1.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ARIA GPU Inference Broker",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv('ARIA_HOST', '0.0.0.0')
    port = int(os.getenv('ARIA_PORT', 7860))
    
    logger.info(f"Starting ARIA server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)