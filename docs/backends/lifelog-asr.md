# Lifelog ASR — Backend STT per ARIA

> **Aggiornato**: 2026-05-11
> **Ambiente**: `%ARIA_ROOT%\envs\lifelog-asr` (Python 3.12)
> **Porta**: 8087
> **Stato**: ✅ Operativo (Blackwell Stable)
> **Client principale**: Lifelog2 (CT190 via Redis)

---

## 1. Panoramica

Lifelog ASR è il backend di trascrizione audio di ARIA, progettato per la pipeline
di memoria personale Lifelog2. Riceve segmenti audio WAV (5 min, 16kHz mono) da MinIO
e produce trascrizioni strutturate con diarizzazione speaker e timestamp a livello parola.

### Funzionalità principali

- **Trascrizione multilingue**: 52 lingue, rilevamento automatico. Primario: italiano
- **Word timestamps**: allineamento forzato a ~43ms di precisione media (ForcedAligner)
- **Diarizzazione speaker**: chi parla, quando — output pronto per SpeakerTurns
- **Rilevamento lingua**: automatico, con possibilità di forzare `it` o `en`
- **Batch inference**: supporta fino a 32 file in parallelo (default 1 per Lifelog2)

---

## 2. Stack modelli

```
WAV input (16kHz mono, ~5 min)
         │
         ▼
┌──────────────────────────────────┐
│  Qwen3-ASR-1.7B                  │  ← trascrizione + language detection
│  (decoder-only + encoder audio)  │    output: testo grezzo + lingua
│  ~4.5 GB VRAM bfloat16           │
└─────────────┬────────────────────┘
              │ testo grezzo
              ▼
┌──────────────────────────────────┐
│  Qwen3-ForcedAligner-0.6B        │  ← allineamento forzato testo → audio
│  (forced alignment)              │    output: word timestamps (ms)
│  ~2 GB VRAM                      │    supporta italiano ✅
└─────────────┬────────────────────┘
              │ timestamps parola
              ▼
┌──────────────────────────────────┐
│  pyannote community-1            │  ← diarizzazione speaker
│  (pyannote.audio 4.0)            │    output: SPEAKER_00, SPEAKER_01...
│  ~2 GB VRAM                      │    DER ~11.2% (gold standard)
└─────────────┬────────────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  Pyannote Embedding (ResNet34)   │  ← estrazione voiceprint pooling
│  (wespeaker-voxceleb)            │    output: vettore 256d
│  ~1.5 GB VRAM                    │
└─────────────┬────────────────────┘
              │
              ▼
         merge: testo + speaker + embedding
              │
              ▼
         SpeakerTurns strutturate → JSON output
```

**VRAM totale stimata**: ~9 GB su 16 GB disponibili. Margine sufficiente per
coesistenza con altri backend piccoli, ma per design ARIA un backend alla volta.

---

## 3. Modello: Qwen3-ASR-1.7B vs WhisperX large-v3

La scelta di Qwen3-ASR al posto di WhisperX è basata su benchmark italiani documentati.

### Confronto WER (Word Error Rate) su italiano

| Dataset | Qwen3-ASR-1.7B | Whisper large-v3 | Vantaggio |
|---------|---------------|-----------------|-----------|
| Common Voice IT | **5.40%** | ~8-10% | ~45% meno errori |
| Fleurs IT | **2.41%** | ~3-5% | ~50% meno errori |
| VRAM (bfloat16) | ~4.5 GB | ~6 GB | più leggero |
| Word timestamps | Nativi (ForcedAligner 0.6B) | wav2vec2 esterno | integrati |
| Language detection | Automatico (52 lingue) | Automatico | equivalente |

### Vantaggi tecnici vs WhisperX

| Aspetto | Qwen3-ASR | WhisperX |
|---------|-----------|---------|
| Architettura | Decoder-only LLM + encoder audio | Encoder-decoder Transformer |
| Word timestamps | ForcedAligner 0.6B (stesso team) | wav2vec2 (libreria separata) |
| Manutenzione | Attiva (Alibaba Qwen, 2025-2026) | Ridotta — fork BetterWhisperX attivo |
| sm_120 (Blackwell) | ✅ PyTorch 2.11+cu128, confermato | ✅ PyTorch 2.8+cu128, confermato V1 |
| Lingue certificate IT | ✅ benchmark pubblici | ⚠️ nessun benchmark IT ufficiale |

### Stabilità Blackwell (Fix 2026-05-11)
A causa di un bug noto nella libreria `libtorchcodec` su Windows 11 con architettura Blackwell, il caricamento audio via `torchaudio.load()` causa crash intermittenti. Il backend è stato patchato per usare **`soundfile.read()`**, garantendo stabilità totale.

### Limitazione nota

`transformers >= 5.x` degrada l'accuracy in modo significativo (issue #138).
L'ambiente `lifelog-asr` usa **`transformers==4.57.6`** pinnato — non aggiornare.

---

## 4. Coda Redis e Payload

### Coda input

```
aria:q:stt:local:qwen3-asr-1.7b:lifelog
```

Pattern standard ARIA: `aria:q:{type}:local:{model_id}:{client_id}`

### Payload task (CT190 → Redis → ARIA)

```json
{
  "job_id": "uuid-v4",
  "client_id": "lifelog",
  "model_type": "stt",
  "model_id": "qwen3-asr-1.7b",
  "callback_key": "aria:c:lifelog:{job_id}",
  "timeout_seconds": 300,
  "priority": 1,
  "payload": {
    "wav_url": "http://192.168.1.104:9000/lifelog/normalized-audio/roberto/2025/10/17/{segment_id}.wav",
    "segment_id": "uuid-del-segmento",
    "language": null,
    "return_timestamps": true,
    "return_speaker_turns": true
  }
}
```

**Campi payload:**

| Campo | Tipo | Default | Descrizione |
|-------|------|---------|-------------|
| `wav_url` | string | — | URL MinIO del WAV normalizzato (16kHz mono) |
| `segment_id` | string | — | UUID del Segment in CT105 |
| `language` | string\|null | null | Lingua forzata (`"it"`, `"en"`) o null per auto-detect |
| `return_timestamps` | bool | true | Include word timestamps nel risultato |
| `return_speaker_turns` | bool | true | Include diarizzazione speaker |

### Risultato (Redis → CT190)

```json
{
  "job_id": "uuid-v4",
  "status": "done",
  "processing_time": 28.4,
  "output": {
    "transcript": "Allora oggi ho parlato con Francesco del progetto...",
    "language": "it",
    "duration_ms": 299800,
    "speaker_turns": [
      {
        "speaker": "SPEAKER_00",
        "start_ms": 0,
        "end_ms": 12400,
        "text": "Allora oggi ho parlato con Francesco del progetto"
      },
      {
        "speaker": "SPEAKER_01",
        "start_ms": 12800,
        "end_ms": 31200,
        "text": "Sì esatto, e la scadenza è giovedì"
      }
    ],
    "word_timestamps": [
      {"word": "Allora", "start_ms": 0, "end_ms": 420},
      {"word": "oggi", "start_ms": 440, "end_ms": 680}
    ]
  }
}
```

---

## 5. Architettura del server FastAPI (porta 8087)

```
backends/lifelog_asr/
├── server.py          ← FastAPI entrypoint (avviato JIT dall'orchestratore)
├── asr_pipeline.py    ← logica Qwen3-ASR + ForcedAligner + pyannote
└── requirements.txt   ← snapshot dipendenze (informativo — env gestito conda)
```

### Endpoints

```
GET  /health
     → {"status": "ok", "model": "qwen3-asr-1.7b", "device": "cuda", "vram_gb": 9.1}

POST /transcribe
     Body: {"wav_url": "...", "segment_id": "...", "language": null, ...}
     → SpeakerTurns JSON (vedi Payload risultato sopra)
```

### Ciclo di vita JIT

```
1. Orchestratore riceve task su aria:q:stt:local:qwen3-asr-1.7b:lifelog
2. subprocess.Popen → avvia envs\lifelog-asr\python.exe backends\lifelog_asr\server.py
3. LifelogASRBackend.load() → polling GET :8087/health (timeout 120s)
4. Server carica Qwen3-ASR + ForcedAligner + pyannote (una volta sola in VRAM)
5. Per ogni task: POST /transcribe → scarica WAV da URL → pipeline → JSON
6. Inattività > 30 min → orchestratore termina processo → VRAM liberata
```

---

## 6. Ambiente conda `lifelog-asr`

**Path**: `C:\Users\Roberto\aria\envs\lifelog-asr\`

| Componente | Versione | Note |
|------------|----------|------|
| Python | 3.12 | |
| PyTorch | 2.11.0+cu128 | sm_120 (Blackwell) confermato |
| `qwen-asr` | latest | porta `transformers==4.57.6` come dipendenza |
| `transformers` | **4.57.6** | ⚠️ NON aggiornare a 5.x |
| `pyannote.audio` | **4.0.1** | Gestisce diarizzazione ed embedding (wespeaker) |

### Comandi setup (da eseguire su PC139)

```cmd
:: Crea env (già eseguito 2026-05-07)
conda create --prefix C:\Users\Roberto\aria\envs\lifelog-asr python=3.12 -y

:: PyTorch cu128 (sm_120 native)
C:\Users\Roberto\aria\envs\lifelog-asr\python.exe -m pip install ^
    torch --index-url https://download.pytorch.org/whl/cu128

:: Qwen3-ASR (porta transformers==4.57.6)
C:\Users\Roberto\aria\envs\lifelog-asr\python.exe -m pip install -U qwen-asr

:: pyannote compatibile (senza pin torch rigido)
C:\Users\Roberto\aria\envs\lifelog-asr\python.exe -m pip install ^
    "pyannote.audio==4.0.1"

:: HF login globale (gestito ora in server.py all'avvio)
:: Assicurarsi che HF_HUB_OFFLINE=0 sia settato.
```

---

## 7. Modelli su disco

```
C:\Users\Roberto\aria\data\assets\models\
├── qwen3-asr-1.7b\              ← ~3.5 GB (download HuggingFace)
└── qwen3-forced-aligner-0.6b\  ← ~1.3 GB (download HuggingFace)
```

Il modello `pyannote/speaker-diarization-community-1` viene scaricato automaticamente
nella cache HuggingFace (`~/.cache/huggingface/`) al primo avvio del server.

### Download modelli

```python
# Da eseguire nell'env lifelog-asr (una volta sola, richiede HF token)
from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen3-ASR-1.7B",
    local_dir=r"C:\Users\Roberto\aria\data\assets\models\qwen3-asr-1.7b"
)
snapshot_download(
    "Qwen/Qwen3-ForcedAligner-0.6B",
    local_dir=r"C:\Users\Roberto\aria\data\assets\models\qwen3-forced-aligner-0.6b"
)
# pyannote si scarica automaticamente in HF cache al primo avvio
```

---

## 8. Wrapper orchestratore

```python
# aria_node_controller/backends/lifelog_asr.py
class LifelogASRBackend:
    model_id   = "qwen3-asr-1.7b"
    model_type = "stt"
    SERVER_URL = "http://127.0.0.1:8087"

    def load(self, model_path, config):
        # polling GET /health fino a server ready (timeout 120s)
        ...

    def unload(self):
        # no-op — processo esterno gestito dall'orchestratore
        ...

    def run(self, payload) -> dict:
        # POST /transcribe → restituisce SpeakerTurns JSON
        ...

    def estimated_vram_gb(self) -> float:
        return 9.0
```

---

## 9. Integrazione Lifelog2 (Stage C)

Il consumer Redis di Stage C su CT190 (`lifelog:stream:asr`):

1. Legge evento da `lifelog:stream:asr` (emesso da Stage B)
2. Costruisce payload ARIA con `wav_url` MinIO del segmento
3. `LPUSH aria:q:stt:local:qwen3-asr-1.7b:lifelog` → job inviato
4. `BRPOP aria:c:lifelog:{job_id}` → attende risultato (timeout 300s)
5. Parsa `speaker_turns` → crea `SpeakerTurn` records in CT105
6. Salva transcript grezzo in MinIO `transcripts/raw/roberto/{YYYY}/{MM}/{DD}/{segment_id}.json`
7. Aggiorna `Segment.pipeline_status = "enriching"`
8. Emette su `lifelog:stream:enrich`

---

*Lifelog ASR Backend — Maggio 2026*
*Documenti correlati: [ARIA Service Registry](../ARIA-Service-Registry.md), [hardware-environments-setup.md](../hardware-environments-setup.md)*
