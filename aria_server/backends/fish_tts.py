import os
import json
import time
import requests
import base64
from typing import Optional, Dict

from aria_server.models import AriaTaskPayload, AriaTaskResult
from aria_server.backends.base import BaseAriaBackend
from aria_server.logger import get_logger

logger = get_logger("aria.backends.fish_tts")

class FishTTSBackend(BaseAriaBackend):
    """
    Backend for Fish TTS running on an external Windows machine.
    Orchestrates two endpoints:
      1) Voice Cloning Encoder (:8081/encode) -> converts WAV to pre-computed tokens
      2) Fish-Speech Modded API (:8080/v1/tts) -> runs inference using the tokens
      
    Features an in-memory token cache to avoid re-encoding the same WAV.
    """
    def __init__(self, encoder_url: str, tts_url: str):
        self.encoder_url = encoder_url
        self.tts_url = tts_url
        self._token_cache: Dict[str, str] = {}
        
    @property
    def model_type(self) -> str:
        return "tts"
        
    @property
    def model_id(self) -> str:
        return "fish-s1-mini"
        
    def estimated_vram_gb(self) -> float:
        return 0.0  # Runs externally
        
    def load(self) -> None:
        try:
            r = requests.get(f"{self.tts_url}/v1/health", timeout=5)
            r.raise_for_status()
            logger.info("FishTTSBackend: External TTS server health check OK.")
        except Exception as e:
            logger.warning(f"FishTTSBackend: External TTS server is unreachable: {e}")
            
    def unload(self) -> None:
        pass

    def _resolve_samba_path(self, external_path: str) -> str:
        return external_path.replace("/mnt/aria-shared", "/aria-shared")

    def _get_or_create_tokens(self, audio_path: str, reference_id: str) -> Optional[str]:
        if reference_id in self._token_cache:
            return self._token_cache[reference_id]
            
        logger.info(f"Encoding reference audio for {reference_id}...")
        try:
            with open(audio_path, "rb") as f:
                wav_bytes = f.read()
        except Exception as e:
            logger.error(f"Failed to read reference audio: {e}")
            return None
            
        try:
            files = {"file": (f"{reference_id}.wav", wav_bytes, "audio/wav")}
            headers = {"Connection": "close"}
            response = requests.post(f"{self.encoder_url}/encode", files=files, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            tokens_b64 = data.get("npy_base64")
            
            if tokens_b64:
                self._token_cache[reference_id] = tokens_b64
                return tokens_b64
            else:
                logger.error("Encoder returned 200 OK but no base64 tokens were found in the JSON.")
                return None
        except Exception as e:
            logger.error(f"Failed to call Encoder API: {e}")
            return None

    def process_task(self, task: AriaTaskPayload) -> AriaTaskResult:
        start_time = time.time()
        
        text = task.payload.get("text")
        if not text:
            return AriaTaskResult(
                job_id=task.job_id, client_id=task.client_id,
                model_type=self.model_type, model_id=self.model_id,
                status="error", processing_time_seconds=0.0,
                error="Missing 'text' in payload", error_code="INVALID_PAYLOAD"
            )
            
        output_path = None
        input_ref_id = None
        input_audio_path = None
        
        if task.file_refs:
            if task.file_refs.output:
                for ref in task.file_refs.output:
                    if ref.ref_id == task.payload.get("output_ref", "audio_output"):
                        output_path = self._resolve_samba_path(ref.shared_path)
                        break
            if task.file_refs.input:
                for ref in task.file_refs.input:
                    if ref.ref_id == task.payload.get("voice_ref", "voice_input"):
                        input_audio_path = self._resolve_samba_path(ref.shared_path)
                        input_ref_id = ref.ref_id
                        break
                        
        if not output_path:
            return AriaTaskResult(
                job_id=task.job_id, client_id=task.client_id,
                model_type=self.model_type, model_id=self.model_id,
                status="error", processing_time_seconds=0.0,
                error="No valid output path referenced", error_code="FILE_NOT_FOUND"
            )
            
        references = []
        if input_audio_path and input_ref_id:
            tokens_b64 = self._get_or_create_tokens(input_audio_path, input_ref_id)
            if tokens_b64:
                # Need the original text exactly spoken in the reference file.
                # Expected to be passed in payload["prompt_text"]
                prompt_text = task.payload.get("prompt_text", "...")
                references.append({
                    "audio": base64.b64encode(b"DUMMY_AUDIO").decode("utf-8"), # Required by schema, bypassed by patch
                    "text": prompt_text,
                    "tokens": tokens_b64
                })
            else:
                logger.warning("No tokens generated. TTS will run without cloning / zero-shot.")

        payload = {
            "text": text,
            "format": "wav",
            "references": references,
            "use_memory_cache": "off", # We handle caching our own way for tokens
            "normalize": True,
            "streaming": False,
            "max_new_tokens": task.payload.get("max_new_tokens", 8192),
            "top_p": task.payload.get("top_p", 0.7),
            "repetition_penalty": task.payload.get("repetition_penalty", 1.1),
            "temperature": task.payload.get("temperature", 0.7),
        }
        
        try:
            logger.info(f"Sending request to Fish TTS (Text length: {len(text)} chars)")
            headers = {"Connection": "close"}
            response = requests.post(
                f"{self.tts_url}/v1/tts",
                json=payload,
                headers=headers,
                timeout=180
            )
            
            if response.status_code != 200:
                error_msg = response.text if response.text else f"Code {response.status_code}"
                raise ValueError(f"TTS inference failed: {error_msg}")
                
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            wav_bytes = response.content
            with open(output_path, "wb") as f:
                f.write(wav_bytes)
                
            # Compute roughly the duration (WAV headers ~44 bytes)
            # Assuming format: WAV PCM 16-bit Mono, at 44.1kHz -> 88200 bytes per second
            # Fish is actually 44.1kHz, 16bit, mono
            audio_size = len(wav_bytes) - 44
            duration_s = max(0.1, audio_size / 88200)
            
            return AriaTaskResult(
                job_id=task.job_id, client_id=task.client_id,
                model_type=self.model_type, model_id=self.model_id,
                status="done", processing_time_seconds=time.time() - start_time,
                output={
                    "audio_ref": task.payload.get("output_ref", "audio_output"),
                    "duration_seconds": round(duration_s, 2),
                    "sample_rate": 44100
                }
            )
        except Exception as e:
            logger.exception("FishTTSBackend: inference error")
            return AriaTaskResult(
                job_id=task.job_id, client_id=task.client_id,
                model_type=self.model_type, model_id=self.model_id,
                status="error", processing_time_seconds=time.time() - start_time,
                error=str(e), error_code="INFERENCE_FAILED"
            )
