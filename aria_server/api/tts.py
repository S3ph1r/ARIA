"""
TTS API endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import logging

from ..backends.tts import OrpheusTTSBackend

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tts", tags=["text-to-speech"])

# Placeholder backend - will be initialized with actual model
tts_backend = None

@router.get("/voices")
async def list_voices():
    """List available TTS voices"""
    if not tts_backend:
        return {"voices": [], "message": "TTS backend not initialized"}
    
    try:
        voices = await tts_backend.list_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Failed to list voices: {e}")
        raise HTTPException(status_code=500, detail="Failed to list voices")

@router.post("/synthesize")
async def synthesize_speech(text: str, voice: Optional[str] = None):
    """Synthesize text to speech"""
    if not tts_backend:
        raise HTTPException(status_code=503, detail="TTS backend not available")
    
    try:
        audio_data = await tts_backend.synthesize(text, voice)
        return {
            "success": True,
            "audio_length": len(audio_data),
            "voice": voice or "default"
        }
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="TTS synthesis not yet implemented")
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail="Synthesis failed")