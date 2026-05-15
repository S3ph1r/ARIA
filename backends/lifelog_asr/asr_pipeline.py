import os
import sys
import logging
import torch
import torchaudio
from types import ModuleType

# MONKEYPATCH TORCHAUDIO PER COMPATIBILITÀ CON SPEECHBRAIN (usato internamente da pyannote.audio)
# 1. list_audio_backends (rimosso in Torchaudio 2.11)
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['ffmpeg']

# 2. io (StreamReader) - Mock per evitare crash all'import di SpeechBrain.inference.ASR
if not hasattr(torchaudio, 'io'):
    io_mock = ModuleType('torchaudio.io')
    io_mock.StreamReader = object
    io_mock.AudioDecoder = object
    torchaudio.io = io_mock
    sys.modules['torchaudio.io'] = io_mock

import numpy as np
import soundfile as sf
import warnings
import gc

# SOLUZIONE ATOMICA PER BLACKWELL (sm_120) + PYTORCH 2.11
# Disabilita il controllo 'weights_only' che causa pickle.UnpicklingError su Pyannote.
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Soppressione warning per terminale pulito (Blackwell Optimized)
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", message="triton not found")

# Import specifici per Qwen3-ASR e Pyannote
try:
    from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
except ImportError:
    Qwen3ASRModel = None

from pyannote.audio import Pipeline, Model, Inference

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
        self.voiceprint_encoder = None

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

        # Caricamento Diarizzatore
        hf_token = os.getenv("HF_TOKEN")
        logger.info("Loading pyannote diarization (Standard ACE-Step Stack Compatibility)...")
        try:
            # Usiamo il login globale effettuato in server.py
            self.diarizer = Pipeline.from_pretrained(PYANNOTE_CONFIG)
            if self.diarizer and torch.cuda.is_available():
                self.diarizer = self.diarizer.to(torch.device("cuda"))
        except Exception as e:
            logger.warning("Failed to load pyannote diarization: %s. Diarization will be skipped.", e)
            self.diarizer = None

        # Caricamento Voiceprint Encoder (Pyannote Native Embedding - ResNet34)
        logger.info("Loading Pyannote Embedding (wespeaker-voxceleb-resnet34-LM)...")
        try:
            # Usiamo il login globale effettuato in server.py
            emb_model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
            self.voiceprint_encoder = Inference(emb_model, window="whole", device=torch.device("cuda"))
            logger.info("Pyannote Embedding loaded (256d vectors).")
        except Exception as e:
            logger.warning("Failed to load Pyannote Embedding model: %s. Voiceprint extraction will be skipped.", e)
            self.voiceprint_encoder = None

        self._loaded = True
        vram_gb = torch.cuda.memory_allocated() / 1e9
        logger.info("ASR Pipeline ready. VRAM: %.1f GB", vram_gb)

    def unload(self):
        if not self._loaded:
            return
        del self.asr_pipeline, self.diarizer, self.voiceprint_encoder
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

        # Caricamento audio con soundfile (più robusto di torchaudio su Windows/Blackwell)
        try:
            audio_data, sr = sf.read(wav_path)
            waveform = torch.from_numpy(audio_data).float()
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=1)  # Mixdown to mono
            waveform = waveform.unsqueeze(0)  # (1, num_samples)
        except Exception as e:
            logger.error("Error loading audio with soundfile: %s. Falling back to torchaudio.", e)
            waveform, sr = torchaudio.load(wav_path)
        duration_ms = int(waveform.shape[1] / sr * 1000)

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
                # Passiamo direttamente waveform e sr caricati con soundfile per evitare crash di torchaudio interni a pyannote
                turns = self._diarize(waveform, sr, transcript, word_ts)
                result["speaker_turns"] = turns
                
                # Estrazione Voiceprints con Pooling (Native Pyannote)
                if self.voiceprint_encoder:
                    logger.info("Extracting pooled voiceprints for detected speakers...")
                    result["voiceprints"] = self._extract_voiceprints(waveform, sr, turns)
                
            except Exception as exc:
                logger.warning("Diarization/Voiceprint skip: %s", exc)
                result["speaker_turns"] = [
                    {
                        "speaker": "SPEAKER_00",
                        "start_ms": 0,
                        "end_ms": duration_ms,
                        "text": transcript,
                    }
                ]

        return result

    def _extract_voiceprints(self, waveform: torch.Tensor, sr: int, turns: list[dict]) -> dict[str, list[float]]:
        """
        Estrae un unico embedding (pooling) per ogni speaker nel file audio usando Pyannote Inference.
        """
        try:
            # Resampling a 16kHz se necessario (Pyannote standard)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
                sr = 16000
            
            speaker_tensors = {}
            for turn in turns:
                spk = turn["speaker"]
                if spk not in speaker_tensors:
                    speaker_tensors[spk] = []
                
                start_sample = int((turn["start_ms"] / 1000) * sr)
                end_sample = int((turn["end_ms"] / 1000) * sr)
                
                # Crop del tensore
                crop = waveform[:, start_sample:end_sample]
                
                # Consideriamo solo segmenti con un minimo di sostanza (> 0.5s)
                if crop.shape[1] > 8000:
                    speaker_tensors[spk].append(crop)
            
            voiceprints = {}
            for spk, tensors in speaker_tensors.items():
                if not tensors: continue
                
                # Concatenazione dei segmenti (Pooling)
                concatenated = torch.cat(tensors, dim=1)
                
                # Limitiamo a max 30 secondi per speaker per efficienza VRAM
                if concatenated.shape[1] > sr * 30:
                    concatenated = concatenated[:, :sr * 30]
                
                # Calcolo Embedding con Pyannote Inference
                # Pyannote Inference({"waveform": ..., "sample_rate": ...}) accetta tensori (C, T)
                with torch.no_grad():
                    embedding = self.voiceprint_encoder({"waveform": concatenated, "sample_rate": sr})
                    # Conversione in lista per JSON
                    voiceprints[spk] = embedding.tolist()
                    
            return voiceprints
        except Exception as e:
            logger.error("Error in voiceprint extraction: %s", e)
            return {}

    def _diarize(self, waveform: torch.Tensor, sr: int, transcript: str, word_timestamps: list) -> list[dict]:
        # Passiamo il waveform tensor invece del path per bypassare i problemi di loading di torchaudio
        res = self.diarizer({"waveform": waveform, "sample_rate": sr})
        
        # Supporto per Pyannote 4.x (restituisce un oggetto DiarizeOutput con diversi attributi)
        if hasattr(res, "speaker_diarization"):
            diarization = res.speaker_diarization
        elif hasattr(res, "annotation"):
            diarization = res.annotation
        else:
            diarization = res
        
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

