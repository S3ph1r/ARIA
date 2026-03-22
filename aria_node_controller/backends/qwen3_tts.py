"""
ARIA — Backend Qwen3-TTS 1.7B
======================================
Segue il pattern External HTTP Backend. 
Supporta varianti multiple (Base, CustomVoice) tramite model_id.

Flusso:
  1. Orchestrator riceve task dalla coda Redis gpu:queue:tts:qwen3-tts-*
  2. Orchestrator garantisce lo swap JIT del modello corretto su porta 8083.
  3. Chiama Qwen3TTSBackend.run(payload)
  4. POST verso http://127.0.0.1:8083/tts
  5. Audio rintracciabile via {job_id}.wav per Remote Skip logic.
"""

import uuid
import time
import logging
import requests
from pathlib import Path

logger = logging.getLogger("aria.backend.qwen3_tts")

# URL del server FastAPI standalone
QWEN3_SERVER_URL = "http://127.0.0.1:8083"


class Qwen3TTSBackend:
    """
    Backend TTS per Qwen3-TTS-12Hz-1.7B-Base.
    Implementa l'interfaccia usata da NodeOrchestrator.
    """

    model_id   = "qwen3-tts-1.7b"
    model_type = "tts"

    def load(self, model_path: str, config: dict) -> None:
        """
        Verifica che il server Qwen3 sia raggiungibile.
        Non carica nulla localmente (processo esterno).
        """
        try:
            r = requests.get(f"{QWEN3_SERVER_URL}/health", timeout=5)
            r.raise_for_status()
            info = r.json()
            logger.info(
                f"Qwen3-TTS server OK: device={info.get('device')} "
                f"VRAM={info.get('vram_allocated_gb')} GB"
            )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Qwen3-TTS server non raggiungibile su {QWEN3_SERVER_URL}. "
                "Avvia start-qwen3-tts.bat prima di caricare questo backend."
            )
        except Exception as e:
            raise RuntimeError(f"Health check Qwen3 fallito: {e}")

    def unload(self) -> None:
        """No-op — il server è un processo esterno gestito dal bat/Task Scheduler."""
        logger.info("Qwen3-TTS backend: unload (no-op, processo esterno)")

    def estimated_vram_gb(self) -> float:
        """VRAM attesa (per ARIA VRAMManager)."""
        return 6.5

    def is_loaded(self) -> bool:
        """Controlla se il server risponde."""
        try:
            r = requests.get(f"{QWEN3_SERVER_URL}/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def run(self, payload: dict, aria_root: Path, local_ip: str) -> dict:
        """
        Sintetizza un task TTS per Qwen3.

        Args:
            payload  : dict del task ARIA (da Redis)
            aria_root: Path base dell'installazione ARIA su Windows
            local_ip : IP LAN di questo nodo (per costruire l'audio_url)

        Returns:
            dict con audio_url, local_path, duration_seconds, ecc.
        """
        text = payload.get("text", "").strip()
        if not text:
            raise ValueError("Campo 'text' obbligatorio nel payload Qwen3-TTS")

        # ── Risoluzione voice ref ──────────────────────────────────────────
        voice_id = payload.get("voice_id")
        voice_ref_path_str = payload.get("voice_ref_audio_path")

        if voice_id and not voice_ref_path_str:
            # Risolvi dalla Voice Library ARIA (preferisce ref_padded.wav)
            voice_dir = aria_root / "data" / "voices" / voice_id
            ref_padded = voice_dir / "ref_padded.wav"
            ref_plain   = voice_dir / "ref.wav"
            if ref_padded.exists():
                voice_ref_path_str = str(ref_padded)
                logger.info(f"Resolved voice '{voice_id}' → ref_padded.wav")
            elif ref_plain.exists():
                voice_ref_path_str = str(ref_plain)
                logger.warning(
                    f"ref_padded.wav non trovato per '{voice_id}', uso ref.wav "
                    "(potrebbe esserci bleeding fonetico — esegui create_padded_ref.py)"
                )
            else:
                logger.info(f"Ricevuto voice_id='{voice_id}' ma nessun ref.wav trovato in {voice_dir}. Procedo (ok se CustomVoice).")

        # ── Parametri ────────────────────────────────────────────────────────
        job_id          = payload.get("job_id") or payload.get("unique_aria_job_id") or str(uuid.uuid4())
        output_filename = f"{job_id}.wav"
        logger.info(f"Target output filename: {output_filename}")

        # L'instruct viene da Stage C (campo qwen3_instruct nella scena)
        # oppure dal payload diretto. Fallback alla mappa statica via emotion.
        instruct = payload.get("qwen3_instruct") or payload.get("instruct")
        if not instruct:
            emotion  = payload.get("scene_metadata", {}).get("primary_emotion", "neutral")
            instruct = _emotion_to_instruct(emotion)
            if payload.get("scene_metadata", {}).get("pace_factor", 1.0) < 0.8:
                instruct += " Very slow and deliberate pace."

        # Arricchisci instruct con le note dialogiche (se la scena contiene dialogo)
        dialogue_notes = payload.get("dialogue_notes")
        if dialogue_notes and payload.get("has_dialogue", False):
            instruct = f"{instruct} Character notes: {dialogue_notes}"
            logger.info(f"Enriched instruct with dialogue_notes: {dialogue_notes[:60]}...")

        # ── Chiamata al server ───────────────────────────────────────────────
        request_body = {
            "text":                   text,
            "voice_id":               voice_id,
            "voice_ref_audio_path":   voice_ref_path_str or "",
            "voice_ref_text":         payload.get("voice_ref_text"),
            "language":               payload.get("language", "Italian"),
            "instruct":               instruct,
            "non_streaming_mode":     payload.get("non_streaming_mode", True),
            "max_new_tokens":         payload.get("max_new_tokens", 4096),
            "temperature":            payload.get("temperature", 0.7),
            "top_p":                  payload.get("top_p", 0.9),
            "repetition_penalty":     payload.get("repetition_penalty", 1.1),
            "output_sample_rate":     payload.get("output_sample_rate", 24000),
            "max_words_per_chunk":    payload.get("chunking", {}).get("max_words_per_chunk", 250),
            "gap_between_chunks_ms":  payload.get("chunking", {}).get("gap_between_chunks_ms", 80),
            "output_filename":        output_filename,
            "subtalker_temperature":  payload.get("subtalker_temperature", 0.4),
            "subtalker_top_k":        payload.get("subtalker_top_k", 50),
            "subtalker_top_p":        payload.get("subtalker_top_p", 0.9),
        }

        timeout = payload.get("timeout_seconds", 3600)
        voice_ref_name = Path(voice_ref_path_str).name if voice_ref_path_str else "None"
        logger.info(
            f"Qwen3 request | job={job_id} | words={len(text.split())} | "
            f"voice={voice_ref_name} | emotion={instruct[:40]}..."
        )

        try:
            response = requests.post(
                f"{QWEN3_SERVER_URL}/tts",
                json=request_body,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Timeout ({timeout}s) durante inferenza Qwen3-TTS — job={job_id}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Errore chiamata Qwen3-TTS server: {e}")

        # ── Costruisce audio_url pubblica ────────────────────────────────────
        # Il server qwen3 serve gli output su /outputs/{filename}
        # ma il porto pubblico è l'asset server ARIA (porta 8082), che serve
        # la stessa OUTPUT_DIR. Usiamo il porto 8082 per uniformità con Fish.
        audio_url = f"http://{local_ip}:8082/outputs/{output_filename}"

        logger.info(
            f"Qwen3 completato | job={job_id} | "
            f"duration={result.get('duration_seconds')}s | "
            f"RTF={result.get('rtf')} | URL={audio_url}"
        )

        return {
            "audio_url":          audio_url,
            "local_path":         result.get("output_path"),
            "duration_seconds":   result.get("duration_seconds"),
            "sample_rate":        result.get("sample_rate"),
            "chunks_count":       result.get("chunks_count"),
            "metrics": {
                "inference_time_seconds": result.get("inference_time_seconds"),
                "rtf":                    result.get("rtf"),
                "vram_peak_gb":           result.get("vram_peak_gb"),
            }
        }


# ──────────────────────────────────────────────────────────────────────────────
# Mappa emozione → instruct (fallback statico)
# ──────────────────────────────────────────────────────────────────────────────
_EMOTION_TO_INSTRUCT = {
    "neutral":   "Warm male voice, Italian audiobook narrator, calm and measured, moderate pace.",
    "neutro":    "Warm male voice, Italian audiobook narrator, calm and measured, moderate pace.",
    "suspense":  "Warm male voice, Italian audiobook narrator, tense and restrained, slightly slower pace, hushed intensity.",
    "tensione":  "Warm male voice, Italian audiobook narrator, tense and restrained, slightly slower pace, hushed intensity.",
    "fear":      "Warm male voice, Italian audiobook narrator, anxious and cautious, slow deliberate pace, quiet.",
    "paura":     "Warm male voice, Italian audiobook narrator, anxious and cautious, slow deliberate pace, quiet.",
    "sadness":   "Warm male voice, Italian audiobook narrator, melancholic and subdued, slow pace, gentle.",
    "tristezza": "Warm male voice, Italian audiobook narrator, melancholic and subdued, slow pace, gentle.",
    "joy":       "Warm male voice, Italian audiobook narrator, warm and bright, energetic, slightly faster pace.",
    "gioia":     "Warm male voice, Italian audiobook narrator, warm and bright, energetic, slightly faster pace.",
    "anger":     "Warm male voice, Italian audiobook narrator, firm and intense, controlled anger, strong pace.",
    "rabbia":    "Warm male voice, Italian audiobook narrator, firm and intense, controlled anger, strong pace.",
    "curiosity": "Warm male voice, Italian audiobook narrator, inquisitive and engaged, moderate pace, slightly raised.",
}

def _emotion_to_instruct(emotion: str) -> str:
    return _EMOTION_TO_INSTRUCT.get(
        emotion.lower(),
        _EMOTION_TO_INSTRUCT["neutral"]
    )
