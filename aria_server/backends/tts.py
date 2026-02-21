"""
TTS Backend - Text to Speech
Placeholder per implementazione futura con Orpheus TTS
"""

from abc import ABC, abstractmethod
from typing import Optional

class TTSBackend(ABC):
    """Abstract base class for TTS backends"""
    
    @abstractmethod
    async def synthesize(self, text: str, voice: Optional[str] = None) -> bytes:
        """Convert text to speech audio bytes"""
        pass
    
    @abstractmethod
    async def list_voices(self) -> list:
        """List available voices"""
        pass

class OrpheusTTSBackend(TTSBackend):
    """Orpheus TTS implementation (placeholder)"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        # TODO: Load model when available
    
    async def synthesize(self, text: str, voice: Optional[str] = None) -> bytes:
        """Convert text to speech using Orpheus TTS"""
        # TODO: Implement actual synthesis
        raise NotImplementedError("Orpheus TTS model not yet implemented")
    
    async def list_voices(self) -> list:
        """List available Orpheus voices"""
        # TODO: Return actual voice list
        return []