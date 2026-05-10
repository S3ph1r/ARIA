import os

# SOLUZIONE ATOMICA PER BLACKWELL (sm_120) + PYTORCH 2.11
# Disabilita il controllo 'weights_only' che causa pickle.UnpicklingError su Pyannote.
# Deve essere impostata PRIMA di importare torch.
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import logging
import torch
import numpy as np
import soundfile as sf
import warnings
import gc

# Soppressione warning per terminale pulito (Blackwell Optimized)
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", message="triton not found")

# Import specifici per Qwen3-ASR
try:
    from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
except ImportError:
    Qwen3ASRModel = None

logger = logging.getLogger(__name__)

# Percorsi locali ARIA (PC 139)
MODEL_DIR = r"C:\Users\Roberto\aria\data\assets\models"
ASR_MODEL_PATH = os.path.join(MODEL_DIR, "qwen3-asr-1.7b")
ALIGNER_MODEL_PATH = os.path.join(MODEL_DIR, "qwen3-forced-aligner-0.6b")
PYANNOTE_CONFIG = os.path.join(MODEL_DIR, "pyannote", "config.yaml")

class ASRPipeline:
    def __init__(self):
        self._loaded = False
        self.asr_pipeline = None 
        self.diarizer = None

    def load(self):
        if self._loaded:
            return

        if Qwen3ASRModel is None:
            raise ImportError("Pacchetto 'qwen_asr' non trovato nell'ambiente.")

        logger.info("Loading Qwen3-ASR-1.7B (Local, Blackwell sm_120, BF16)...")
        # Caricamento Qwen3 con wrapper nativo in BF16 per Blackwell
        self.asr_pipeline = Qwen3ASRModel.from_pretrained(
            pretrained_model_name_or_path=ASR_MODEL_PATH,
            forced_aligner=ALIGNER_MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True
        )

        logger.info(f"Loading pyannote diarization (Standard ACE-Step Stack Compatibility)...")
        from pyannote.audio import Pipeline
        # Grazie a TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1, questo caricamento ora avrà successo su Torch 2.11
        self.diarizer = Pipeline.from_pretrained(PYANNOTE_CONFIG)
        self.diarizer = self.diarizer.to(torch.device("cuda"))

        self._loaded = True
        vram_gb = torch.cuda.memory_allocated() / 1e9
        logger.info("ASR Pipeline ready. VRAM: %.1f GB", vram_gb)

    def unload(self):
        if not self._loaded:
            return
        del self.asr_pipeline, self.diarizer
        gc.collect()
        torch.cuda.empty_cache()
        self._loaded = False

    def run(
        self,
        wav_path: str,
        language: str | None = None,
        return_timestamps: bool = True,
        return_speaker_turns: bool = True,
    ) -> dict:
        if not self._loaded:
            self.load()

        audio, sr = sf.read(wav_path)
        duration_ms = int(len(audio) / sr * 1000)

        logger.info(f"Transcribing: {wav_path}")
        
        # Mapping ISO -> Qwen3 Names (evita errore Unsupported language: It)
        lang_map = {
            "it": "Italian", "en": "English", "zh": "Chinese", 
            "es": "Spanish", "fr": "French", "de": "German",
            "ja": "Japanese", "ko": "Korean", "ru": "Russian"
        }
        q_lang = lang_map.get(language.lower(), language) if language else None

        results = self.asr_pipeline.transcribe(
            audio=wav_path,
            language=q_lang,
            return_time_stamps=return_timestamps
        )
        
        main_res = results[0]
        transcript = main_res.text
        detected_lang = main_res.language or q_lang or "Italian"

        result: dict = {
            "transcript": transcript,
            "language": detected_lang,
            "duration_ms": duration_ms,
        }

        word_ts = []
        if return_timestamps and main_res.time_stamps:
            for it in main_res.time_stamps.items:
                word_ts.append({
                    "word": it.text,
                    "start_ms": int(it.start_time * 1000),
                    "end_ms": int(it.end_time * 1000)
                })
        result["word_timestamps"] = word_ts

        if return_speaker_turns:
            try:
                result["speaker_turns"] = self._diarize(wav_path, transcript, word_ts)
            except Exception as exc:
                logger.warning("Diarization skip: %s", exc)
                result["speaker_turns"] = [
                    {
                        "speaker": "SPEAKER_00",
                        "start_ms": 0,
                        "end_ms": duration_ms,
                        "text": transcript,
                    }
                ]

        return result

    def _diarize(self, wav_path: str, transcript: str, word_timestamps: list) -> list[dict]:
        diarization = self.diarizer({"uri": "segment", "audio": wav_path})
        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            
            words_in_turn = [
                w["word"]
                for w in word_timestamps
                if w["start_ms"] >= start_ms - 50 and w["end_ms"] <= end_ms + 150
            ]
            
            turns.append(
                {
                    "speaker": speaker,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": " ".join(words_in_turn) if words_in_turn else "",
                }
            )
        return turns
