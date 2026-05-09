import gc
import logging
import torch
import numpy as np
import soundfile as sf
import os
import warnings
import functools

# Soppressione dei warning superflui per un terminale pulito
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", message="triton not found")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Fix per caricamento modelli con versioni recenti di Torch (PyTorch 2.6+ / 2.11 Nightly)
# Questo permette il caricamento di classi specifiche in modalità weights_only=True
import torch.torch_version
try:
    from pyannote.audio.core.task import Specifications
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    
    safe_globals = [
        torch.torch_version.TorchVersion,
        Specifications,
        ListConfig,
        DictConfig,
        np.dtype,
        np.core.multiarray._reconstruct,
        np.ndarray,
    ]
    
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals(safe_globals)
        # Alcune versioni di torch usano una lista interna diversa, aggiungiamo per sicurezza
        if hasattr(torch.serialization, '_get_safe_globals'):
             torch.serialization._get_safe_globals().update(safe_globals)
except Exception:
    pass

# Import specifici per Qwen3-ASR
try:
    from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
except ImportError:
    Qwen3ASRModel = None

logger = logging.getLogger(__name__)

# Percorsi locali per autonomia totale (PC 139)
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

        logger.info("Loading Qwen3-ASR-1.7B (Local, Blackwell sm_120)...")
        # Caricamento Qwen3 con wrapper nativo in BF16 per Blackwell
        self.asr_pipeline = Qwen3ASRModel.from_pretrained(
            pretrained_model_name_or_path=ASR_MODEL_PATH,
            forced_aligner=ALIGNER_MODEL_PATH,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True
        )

        logger.info(f"Loading pyannote diarization (Offline Mode)...")
        from pyannote.audio import Pipeline
        # Caricamento autonomo da config locale. 
        # NOTA: Pyannote 3.1 internamente usa torch.load, i safe_globals sopra risolvono il blocco.
        self.diarizer = Pipeline.from_pretrained(PYANNOTE_CONFIG)
        self.diarizer = self.diarizer.to(torch.device("cuda"))

        self._loaded = True
        vram_gb = torch.cuda.memory_allocated() / 1e9
        logger.info("Models loaded successfully. VRAM used: %.1f GB", vram_gb)

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
        results = self.asr_pipeline.transcribe(
            audio=wav_path,
            language=language,
            return_time_stamps=return_timestamps
        )
        
        main_res = results[0]
        transcript = main_res.text
        detected_lang = main_res.language or language or "it"

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
