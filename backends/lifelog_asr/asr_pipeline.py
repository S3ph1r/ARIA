import gc
import logging

import torch

logger = logging.getLogger(__name__)

MODEL_DIR = r"C:\Users\Roberto\aria\data\assets\models"
ASR_MODEL_PATH = MODEL_DIR + r"\qwen3-asr-1.7b"
ALIGNER_MODEL_PATH = MODEL_DIR + r"\qwen3-forced-aligner-0.6b"
PYANNOTE_MODEL = "pyannote/speaker-diarization-community-1"


class ASRPipeline:
    def __init__(self):
        self._loaded = False
        self.asr_model = None
        self.asr_processor = None
        self.aligner = None
        self.aligner_proc = None
        self.diarizer = None

    def load(self):
        if self._loaded:
            return

        logger.info("Loading Qwen3-ASR-1.7B ...")
        from transformers import AutoModelForCTC, AutoProcessor
        self.asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_PATH)
        self.asr_model = AutoModelForCTC.from_pretrained(
            ASR_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        logger.info("Loading Qwen3-ForcedAligner-0.6B ...")
        from transformers import AutoModelForCTC as AlignerModel, AutoProcessor as AlignerProc
        self.aligner_proc = AlignerProc.from_pretrained(ALIGNER_MODEL_PATH)
        self.aligner = AlignerModel.from_pretrained(
            ALIGNER_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        logger.info("Loading pyannote speaker-diarization-community-1 ...")
        from pyannote.audio import Pipeline
        self.diarizer = Pipeline.from_pretrained(PYANNOTE_MODEL)
        self.diarizer = self.diarizer.to(torch.device("cuda"))

        self._loaded = True
        vram_gb = torch.cuda.memory_allocated() / 1e9
        logger.info("Models loaded. VRAM used: %.1f GB", vram_gb)

    def unload(self):
        if not self._loaded:
            return
        del self.asr_model, self.asr_processor, self.aligner, self.aligner_proc, self.diarizer
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
        import soundfile as sf

        audio, sr = sf.read(wav_path)
        duration_ms = int(len(audio) / sr * 1000)

        # ASR transcription
        inputs = self.asr_processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        gen_kwargs = {}
        if language:
            gen_kwargs["language"] = language

        with torch.no_grad():
            output = self.asr_model.generate(**inputs, **gen_kwargs, return_dict_in_generate=True)

        transcript = self.asr_processor.batch_decode(
            output.sequences, skip_special_tokens=True
        )[0].strip()
        detected_lang = language or "it"

        result: dict = {
            "transcript": transcript,
            "language": detected_lang,
            "duration_ms": duration_ms,
        }

        # Word timestamps via ForcedAligner
        if return_timestamps:
            try:
                result["word_timestamps"] = self._align(audio, sr, transcript)
            except Exception as exc:
                logger.warning("ForcedAligner failed: %s", exc)
                result["word_timestamps"] = []

        # Speaker diarization
        if return_speaker_turns:
            try:
                word_ts = result.get("word_timestamps", [])
                result["speaker_turns"] = self._diarize(wav_path, transcript, word_ts)
            except Exception as exc:
                logger.warning("Diarization failed: %s", exc)
                result["speaker_turns"] = [
                    {
                        "speaker": "SPEAKER_00",
                        "start_ms": 0,
                        "end_ms": duration_ms,
                        "text": transcript,
                    }
                ]

        return result

    def _align(self, audio, sr: int, transcript: str) -> list[dict]:
        inputs = self.aligner_proc(audio, text=transcript, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.aligner(**inputs)
        decoded = self.aligner_proc.batch_decode(outputs.logits, output_word_offsets=True)
        words = []
        if hasattr(decoded, "word_offsets") and decoded.word_offsets:
            time_offset_ms = 20  # 50fps → 20ms per frame
            for entry in decoded.word_offsets[0]:
                words.append(
                    {
                        "word": entry["word"],
                        "start_ms": entry["start_offset"] * time_offset_ms,
                        "end_ms": entry["end_offset"] * time_offset_ms,
                    }
                )
        return words

    def _diarize(self, wav_path: str, transcript: str, word_timestamps: list) -> list[dict]:
        diarization = self.diarizer({"uri": "segment", "audio": wav_path})
        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            words_in_turn = [
                w["word"]
                for w in word_timestamps
                if w["start_ms"] >= start_ms and w["end_ms"] <= end_ms + 100
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
