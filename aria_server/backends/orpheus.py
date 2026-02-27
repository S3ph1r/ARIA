import json
import logging
import re
import time
import os
import requests
import numpy as np
import scipy.io.wavfile as wavfile
import torch

from snac import SNAC
from typing import List, Optional
from aria_server.models import AriaTaskPayload, AriaTaskResult
from aria_server.backends.base import BaseAriaBackend
from aria_server.logger import get_logger

logger = get_logger("aria.backends.orpheus")

class OrpheusBackend(BaseAriaBackend):
    """
    Backend for Orpheus TTS.
    Acts as an intelligent proxy:
    1. Sends text to native llama-server.exe over HTTP stream.
    2. Extracts SNAC audio tokens from the incoming text stream.
    3. Decodes the SNAC tokens locally (CPU) into PCM audio.
    4. Writes the final .wav to the Samba share path requested by DIAS.
    """
    def __init__(self, llama_url: str):
        self.llama_url = llama_url
        self._snac_model = None
        self._snac_device = "cpu"
        
        # Audio generation constants
        self.START_TOKEN_ID = 128259
        self.END_TOKEN_IDS = [128009, 128260, 128261, 128257]
        self.CUSTOM_TOKEN_PREFIX = "<custom_token_"
        
        # We lazily load the SNAC decoder on first use to save memory if not called
        
    @property
    def model_type(self) -> str:
        return "tts"
        
    @property
    def model_id(self) -> str:
        return "orpheus-3b"
        
    def estimated_vram_gb(self) -> float:
        return 7.0
        
    def load(self) -> None:
        """
        Verify native server is reachable.
        We don't load into VRAM here because llama-server.exe is native Windows.
        """
        try:
            r = requests.get(f"{self.llama_url}/health", timeout=5)
            r.raise_for_status()
            logger.info("OrpheusBackend: llama-server.exe health check OK.")
        except Exception as e:
            logger.warning(f"OrpheusBackend: llama-server.exe health check failed: {e}")
            
        # Also pre-load the SNAC decoder to CPU
        if self._snac_model is None:
            logger.info("Loading SNAC decoder to CPU...")
            self._snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self._snac_device)
            
    def unload(self) -> None:
        """No-op for hybrid native Llama."""
        pass

    def _turn_token_into_id(self, token_text: str) -> Optional[int]:
        """Extracts the integer ID from <custom_token_XXXX> strings."""
        if not token_text:
            return None
        if token_text.startswith(self.CUSTOM_TOKEN_PREFIX):
            try:
                num_str = token_text[len(self.CUSTOM_TOKEN_PREFIX):-1]
                return int(num_str)
            except ValueError:
                return None
        
        # Fallback regex just in case
        match = re.search(r"<custom_token_(\d+)>", token_text)
        if match:
             return int(match.group(1))
        return None

    def _decode_snac_window(self, window_tokens: List[int]) -> Optional[bytes]:
        """
        Decodes a single window of SNAC tokens (must be multiple of 7) into raw PCM bytes.
        Exactly replicates convert_to_audio() from Orpheus-FastAPI speechpipe.py.
        Returns raw int16 bytes for this window, or None if invalid.
        """
        if len(window_tokens) < 7:
            return None
            
        num_frames = len(window_tokens) // 7
        frame = window_tokens[:num_frames * 7]
        
        # Build the hierarchical codes exactly as speechpipe.py does
        codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=self._snac_device)
        codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=self._snac_device)
        codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=self._snac_device)
        
        frame_tensor = torch.tensor(frame, dtype=torch.int32, device=self._snac_device)
        
        for j in range(num_frames):
            idx = j * 7
            codes_0[j] = frame_tensor[idx]
            codes_1[j*2] = frame_tensor[idx+1]
            codes_1[j*2+1] = frame_tensor[idx+4]
            codes_2[j*4] = frame_tensor[idx+2]
            codes_2[j*4+1] = frame_tensor[idx+3]
            codes_2[j*4+2] = frame_tensor[idx+5]
            codes_2[j*4+3] = frame_tensor[idx+6]
            
        codes = [
            codes_0.unsqueeze(0), 
            codes_1.unsqueeze(0), 
            codes_2.unsqueeze(0)
        ]
        
        # Check ranges
        if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or 
            torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or 
            torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
            logger.error("Out of range tokens in SNAC window decoding.")
            return None
            
        with torch.no_grad():
            audio_hat = self._snac_model.decode(codes)
            # Slice [2048:4096] — this is REQUIRED by the SNAC codec to remove
            # warmup/padding samples. This is how speechpipe.py does it.
            audio_slice = audio_hat[:, :, 2048:4096]
            audio_np = audio_slice.detach().cpu().numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            return audio_int16.tobytes()

    def _convert_to_audio(self, token_ids: List[int]) -> Optional[np.ndarray]:
        """
        Decodes a flat list of SNAC tokens into PCM audio using the EXACT same
        sliding-window approach as Orpheus-FastAPI's tokens_decoder + convert_to_audio.
        
        We process the buffer in overlapping windows of ideal_frames (49 tokens = 7 frames),
        stepping by 7 tokens, and slice [2048:4096] from each decode call.
        """
        if not self._snac_model:
            self._snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(self._snac_device)
            
        if len(token_ids) < 7:
            return None
        
        IDEAL_WINDOW = 49   # 7 frames * 7 tokens = 49 tokens per window
        MIN_WINDOW = 28     # 4 frames * 7 tokens = minimum window size
        STEP = 7            # Step by one frame (7 tokens) at a time
        
        audio_chunks = []
        count = 0
        
        # Simulate the streaming window approach from tokens_decoder()
        # Process every STEP tokens, using the last IDEAL_WINDOW tokens as the decode window
        for i in range(STEP, len(token_ids) + 1, STEP):
            count += STEP
            
            if i >= IDEAL_WINDOW:
                window = token_ids[i - IDEAL_WINDOW:i]
            elif i >= MIN_WINDOW:
                window = token_ids[i - MIN_WINDOW:i]
            else:
                continue  # Not enough tokens yet
            
            chunk = self._decode_snac_window(window)
            if chunk is not None:
                audio_chunks.append(chunk)
        
        # Process any remaining tokens at the end
        remaining = len(token_ids) % STEP
        if remaining > 0 and len(token_ids) >= MIN_WINDOW:
            window = token_ids[-IDEAL_WINDOW:] if len(token_ids) >= IDEAL_WINDOW else token_ids[-MIN_WINDOW:]
            chunk = self._decode_snac_window(window)
            if chunk is not None:
                audio_chunks.append(chunk)
        
        if not audio_chunks:
            logger.error("No audio chunks produced from sliding window decode.")
            return None
            
        # Concatenate all raw PCM bytes and convert to numpy int16
        all_bytes = b''.join(audio_chunks)
        audio_int16 = np.frombuffer(all_bytes, dtype=np.int16)
        
        logger.info(f"Decoded {len(audio_chunks)} audio windows, total {len(audio_int16)} samples")
        return audio_int16
            
    def _resolve_samba_path(self, external_path: str) -> str:
        """
        Maps MiniPC perspective (/mnt/aria-shared/) 
        to container perspective (/aria-shared/)
        """
        return external_path.replace("/mnt/aria-shared", "/aria-shared")

    def process_task(self, task: AriaTaskPayload) -> AriaTaskResult:
        start_time = time.time()
        
        # Extract parameters
        text = task.payload.get("text")
        if not text:
            return AriaTaskResult(
                job_id=task.job_id, client_id=task.client_id,
                model_type=self.model_type, model_id=self.model_id,
                status="error", processing_time_seconds=0.0,
                error="Missing 'text' in payload", error_code="INVALID_PAYLOAD"
            )
            
        # Get output path
        output_path = None
        if task.file_refs and task.file_refs.output:
            for ref in task.file_refs.output:
                if ref.ref_id == task.payload.get("output_ref", "audio_output"):
                    output_path = self._resolve_samba_path(ref.shared_path)
                    break
                    
        if not output_path:
            return AriaTaskResult(
                job_id=task.job_id, client_id=task.client_id,
                model_type=self.model_type, model_id=self.model_id,
                status="error", processing_time_seconds=0.0,
                error="No valid output path referenced", error_code="FILE_NOT_FOUND"
            )

        # Orpheus is a multimodal model. It REQUIRES special tokens to switch to audio generation mode.
        # Format must identically match: <|audio|>{voice}: {text}<|eot_id|>
        formatted_prompt = f"<|audio|>{text}<|eot_id|>"

        llama_payload = {
            "prompt": formatted_prompt,
            "stream": True,
            "temperature": 0.65,
            "repeat_penalty": 1.1,
            "max_tokens": 8192,    # Max context window for Orpheus 3b
            "ignore_eos": False
        }
        
        token_ids_buffer = []
        
        try:
            logger.info(f"Sending prompt to local Orpheus: {formatted_prompt}")
            response = requests.post(
                f"{self.llama_url}/v1/completions",
                json=llama_payload,
                stream=True,
                timeout=120
            )
            
            # CHECK THE STATUS FIRST
            if response.status_code != 200:
                logger.error(f"Llama Server returned HTTP {response.status_code}: {response.text}")
                raise ValueError(f"Inference failed with HTTP {response.status_code}")

            current_frame = []
            expected_group = 0
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str == "[DONE]":
                            break
                            
                        try:
                            chunk = json.loads(data_str)
                            token_text = chunk.get("choices", [{}])[0].get("text", "")
                            
                            tid = self._turn_token_into_id(token_text)
                            if tid is not None:
                                # CRITICAL: subtract the base offset of 10 BEFORE group math!
                                # Official formula: raw - 10 - (group * 4096)
                                tid_adjusted = tid - 10
                                group = tid_adjusted // 4096
                                snac_code = tid_adjusted % 4096
                                
                                if group == expected_group:
                                    current_frame.append(snac_code)
                                    expected_group += 1
                                    
                                    # If we collected a full frame of 7 tokens
                                    if expected_group == 7:
                                        token_ids_buffer.extend(current_frame)
                                        current_frame = []
                                        expected_group = 0
                                else:
                                    # Misaligned! Llama generated an extraneous token (like the initial group 6)
                                    # Throw away current frame and attempt to resync to a new group 0
                                    logger.debug(f"SNAC Alignment shift! Expected {expected_group}, got {group}")
                                    if group == 0:
                                        current_frame = [snac_code]
                                        expected_group = 1
                                    else:
                                        current_frame = []
                                        expected_group = 0
                        except json.JSONDecodeError:
                            continue
                            
            logger.info(f"Total perfectly aligned SNAC codes extracted: {len(token_ids_buffer)}")
            
            # Decode to audio
            pcm_audio = self._convert_to_audio(token_ids_buffer)
            if pcm_audio is None:
                logger.error(f"Buffer length was {len(token_ids_buffer)}. Token IDs: {token_ids_buffer[:20]}...")
                raise ValueError("Failed to decode tokens, invalid or empty frame.")
                
            # Make sure parent directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write to disk
            sample_rate = task.payload.get("sample_rate", 24000)
            wavfile.write(output_path, sample_rate, pcm_audio)
            
            # Calculate duration
            duration_s = float(len(pcm_audio)) / sample_rate
            
            return AriaTaskResult(
                job_id=task.job_id, client_id=task.client_id,
                model_type=self.model_type, model_id=self.model_id,
                status="done", processing_time_seconds=time.time() - start_time,
                output={
                    "audio_ref": task.payload.get("output_ref", "audio_output"),
                    "duration_seconds": round(duration_s, 2),
                    "sample_rate": sample_rate
                }
            )

        except Exception as e:
            logger.exception("OrpheusBackend: inference error")
            return AriaTaskResult(
                job_id=task.job_id, client_id=task.client_id,
                model_type=self.model_type, model_id=self.model_id,
                status="error", processing_time_seconds=time.time() - start_time,
                error=str(e), error_code="INFERENCE_FAILED"
            )
