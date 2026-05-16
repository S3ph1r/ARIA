# Lifelog WhisperX — Backend STT per ARIA

> **Aggiornato**: 2026-05-14
> **Ambiente**: `%ARIA_ROOT%\envs\lifelog-whisperx` (Python 3.12)
> **Porta**: 8091
> **Stato**: ✅ Operativo (Blackwell Stable, confermato 2026-05-14)
> **Client principale**: Lifelog2 (CT190 via Redis)

---

## 1. Panoramica

Lifelog WhisperX è il backend di trascrizione audio di ARIA basato su WhisperX large-v3.
Sostituisce Qwen3-ASR-1.7B come backend primario di Lifelog2, aggiungendo voiceprint 256d
integrati (pyannote wespeaker-resnet34-LM) e diarizzazione nativa.

### Funzionalità principali

- **Trascrizione multilingue**: 99 lingue, rilevamento automatico. Primario: italiano
- **Word timestamps**: wav2vec2 forced alignment, ~30ms di precisione
- **Diarizzazione speaker**: pyannote community-1 — chi parla, quando
- **Voiceprint embedding**: wespeaker-resnet34-LM, vettore 256d per speaker, pooling su max 30s
- **Output contract identico a Qwen3-ASR**: Stage C è model-agnostic

---

## 2. Stack modelli

```
WAV input (16kHz mono, ~5 min)
         │
         ▼
┌──────────────────────────────────┐
│  WhisperX large-v3               │  ← trascrizione + language detection
│  (encoder-decoder Transformer)  │    output: testo raw + lingua
│  ~6 GB VRAM float16              │
└─────────────┬────────────────────┘
              │ testo grezzo
              ▼
┌──────────────────────────────────┐
│  wav2vec2 ITA (align model)      │  ← allineamento forzato testo → audio
│  (whisperx.load_align_model)    │    output: word timestamps (ms)
│  ~0.5 GB VRAM                   │
└─────────────┬────────────────────┘
              │ timestamps parola
              ▼
┌──────────────────────────────────┐
│  pyannote community-1            │  ← diarizzazione speaker
│  (pyannote.audio 3.x)           │    output: SPEAKER_00, SPEAKER_01...
│  ~2 GB VRAM                     │
└─────────────┬────────────────────┘
              │ turn boundaries
              ▼
┌──────────────────────────────────┐
│  wespeaker-resnet34-LM           │  ← estrazione voiceprint pooling
│  (pyannote Inference, window=whole)│   output: vettore 256d
│  ~0.5 GB VRAM                   │
└─────────────┬────────────────────┘
              │
              ▼
         merge: testo + speaker + embedding
              │
              ▼
         SpeakerTurns strutturate → JSON output
```

**VRAM totale**: ~9-10 GB su 16 GB disponibili.
**Startup da cache HF**: ~22s (7.6s ASR + 1.5s align + 1.2s diarize + 2s voiceprint).
**Latency su ~5min audio**: ~30s totali (1.5x realtime, GPU RTX 5060 Ti).

---

## 3. Blackwell Fixes (RTX 5060 Ti, sm_120)

Due problemi noti su architettura Blackwell, risolti in `server.py`:

### Fix 1 — compute_type float16
cuBLAS int8 non è supportato su sm_120 (`CUBLAS_STATUS_NOT_SUPPORTED`).
```python
_model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type="float16")
```

### Fix 2 — Arch spoof per pyannote
NVRTC Jiterator fallisce su FFT complessa a sm_120.
Il monkey-patch deve essere applicato **prima** di qualsiasi import pyannote/torchaudio.
```python
_orig_cap = torch.cuda.get_device_capability
def _patched_cap(device=None):
    cap = _orig_cap(device)
    return (9, 0) if cap[0] >= 12 else cap
torch.cuda.get_device_capability = _patched_cap
```

### Fix 3 — soundfile bypass per audio loading
conda-forge ffmpeg DLL crash su Windows 11 Blackwell (exit 0xC0000139 = STATUS_ENTRYPOINT_NOT_FOUND).
Il server bypassa `whisperx.load_audio()` e legge i WAV direttamente con soundfile:
```python
audio_np, sr = sf.read(wav_path, dtype="float32", always_2d=False)
```

---

## 4. Coda Redis e Payload

### Coda input

```
aria:q:stt:local:whisperx-large-v3:lifelog
```

Pattern standard ARIA: `aria:q:{type}:local:{model_id}:{client_id}`

### Payload task (CT190 → Redis → ARIA)

```json
{
  "job_id": "uuid-v4",
  "client_id": "lifelog",
  "model_type": "stt",
  "model_id": "whisperx-large-v3",
  "callback_key": "aria:c:lifelog:{job_id}",
  "timeout_seconds": 1800,
  "priority": 1,
  "payload": {
    "wav_url": "http://192.168.1.104:9000/lifelog/normalized-audio/{user_id}/{yyyy}/{mm}/{dd}/{segment_id}.wav",
    "segment_id": "uuid-del-segmento",
    "language": "it"
  }
}
```

### Risultato (Redis → CT190)

```json
{
  "job_id": "uuid-v4",
  "status": "done",
  "processing_time": 29.5,
  "output": {
    "transcript": "Allora oggi ho parlato con Francesco del progetto...",
    "language": "it",
    "duration_ms": 299800,
    "speaker_turns": [
      {"speaker": "SPEAKER_00", "start_ms": 0, "end_ms": 12400, "text": "Allora oggi ho parlato con Francesco del progetto"},
      {"speaker": "SPEAKER_01", "start_ms": 12800, "end_ms": 31200, "text": "Sì esatto, e la scadenza è giovedì"}
    ],
    "word_timestamps": [
      {"word": "Allora", "start_ms": 0, "end_ms": 420},
      {"word": "oggi", "start_ms": 440, "end_ms": 680}
    ],
    "voiceprints": {
      "SPEAKER_00": [0.123, -0.456, ...],
      "SPEAKER_01": [0.789, 0.012, ...]
    }
  }
}
```

---

## 5. Architettura del server FastAPI (porta 8091)

```
backends/lifelog_whisperx/
└── server.py   ← FastAPI entrypoint (avviato JIT dall'orchestratore)
```

### Endpoints

```
GET  /health
     → {"status": "ok", "model": "whisperx-large-v3", "device": "cuda", "vram_gb": 9.4, "voiceprint": true}

POST /transcribe
     Body: {"wav_url": "...", "segment_id": "...", "language": "it"}
     → SpeakerTurns JSON con voiceprints (vedi Payload risultato sopra)
```

### Ciclo di vita JIT

```
1. Orchestratore riceve task su aria:q:stt:local:whisperx-large-v3:lifelog
2. subprocess.Popen → avvia envs\lifelog-whisperx\python.exe backends\lifelog_whisperx\server.py
3. LifelogWhisperXBackend.load() → polling GET :8091/health (timeout 150s)
4. Server carica WhisperX + align + diarize + wespeaker (una volta sola in VRAM)
5. Per ogni task: POST /transcribe → scarica WAV da URL → pipeline → JSON
6. Inattività > 45 min → orchestratore termina processo → VRAM liberata
```

---

## 6. Ambiente conda `lifelog-whisperx`

**Path**: `C:\Users\Roberto\aria\envs\lifelog-whisperx\`

| Componente | Versione | Note |
|------------|----------|------|
| Python | 3.12 | |
| PyTorch | 2.8.0+cu128 | sm_120 (Blackwell) — versione richiesta da whisperx |
| `whisperx` | 3.8.5 | WhisperX large-v3 + wav2vec2 align |
| `pyannote.audio` | 3.x | speaker-diarization-community-1 + wespeaker-resnet34-LM |
| `soundfile` | latest | Bypass ffmpeg DLL crash |
| `resampy` | latest | Resampling audio se SR ≠ 16kHz |

### Comandi setup (già eseguito 2026-05-14)

```cmd
conda create --prefix C:\Users\Roberto\aria\envs\lifelog-whisperx python=3.12 -y

:: PyTorch cu128 (whisperx richiede 2.8.x)
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 torchvision==0.23.0+cu128 ^
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

:: WhisperX
pip install whisperx==3.8.5

:: pyannote (wespeaker embedding)
pip install pyannote.audio

:: utility
pip install soundfile resampy fastapi uvicorn python-dotenv minio huggingface_hub requests
```

---

## 7. Modelli su disco

Tutti i pesi in HF cache (`~/.cache/huggingface/`) — scaricati automaticamente al primo avvio.

| Modello | Dimensione approssimativa |
|---------|--------------------------|
| `openai/whisper-large-v3` | ~3 GB |
| `jonatasgrosman/wav2vec2-large-xlsr-53-italian` (align ITA) | ~1.3 GB |
| `pyannote/speaker-diarization-community-1` | ~2 GB |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | ~0.5 GB |

---

## 8. Wrapper orchestratore

```python
# aria_node_controller/backends/lifelog_whisperx.py
class LifelogWhisperXBackend:
    model_id   = "whisperx-large-v3"
    model_type = "stt"
    SERVER_URL = "http://127.0.0.1:8091"

    def load(self, model_path, config):
        # polling GET /health fino a server ready
        ...

    def run(self, payload) -> dict:
        # POST /transcribe → restituisce SpeakerTurns JSON con voiceprints
        ...

    def estimated_vram_gb(self) -> float:
        return 12.0
```

---

## 9. Integrazione Lifelog2 (Stage C)

Stage C su CT190 (`lifelog:stream:asr`):

1. Legge evento da `lifelog:stream:asr` (emesso da Stage B)
2. Costruisce payload ARIA con `wav_url` MinIO del segmento
3. `LPUSH aria:q:stt:local:whisperx-large-v3:lifelog` → job inviato
4. `BRPOP aria:c:lifelog:{job_id}` → attende risultato (timeout 1800s)
5. Parsa `speaker_turns` e `voiceprints` → crea `SpeakerTurn` records in CT105
6. Salva transcript grezzo in MinIO `transcripts/raw/{user_id}/{YYYY}/{MM}/{DD}/{segment_id}.json`
7. Aggiorna `Segment.pipeline_status = "enriching"`
8. Emette su `lifelog:stream:enrich`

---

*Lifelog WhisperX Backend — Maggio 2026*
*Documenti correlati: [ARIA Service Registry](../ARIA-Service-Registry.md), [Lifelog ASR (qwen3)](lifelog-asr.md)*
