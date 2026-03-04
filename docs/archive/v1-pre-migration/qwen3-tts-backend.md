# Qwen3-TTS 1.7B — Backend TTS per ARIA
## Affiancamento a Fish S1-mini: Specifiche Complete per Sviluppo e Test Comparativo

> **Contesto**: Questo documento specifica il backend `qwen3_tts.py` come secondo
> backend TTS nell'architettura ARIA, da far girare **in parallelo** a `fish_tts.py`.
> L'obiettivo è un confronto qualitativo controllato su italiano e prosodia prima di
> decidere quale modello usare in produzione per DIAS. Tutto il resto dell'architettura
> — BatchOptimizer, Redis, code DIAS, Samba, semaforo — resta invariato.

---

## INDICE

1. [Perché Qwen3-TTS come candidato](#1-perché-qwen3-tts-come-candidato)
2. [Architettura Tecnica Interna](#2-architettura-tecnica-interna)
3. [Setup Ambiente — Windows 11 Nativo](#3-setup-ambiente--windows-11-nativo)
4. [Download e Configurazione Modello](#4-download-e-configurazione-modello)
5. [Gestione Voice Sample e Ref.txt per Italiano](#5-gestione-voice-sample-e-reftxt-per-italiano)
6. [Controllo Prosodia ed Emozione in Italiano](#6-controllo-prosodia-ed-emozione-in-italiano)
7. [Chunking per Testi Lunghi (Audiolibri)](#7-chunking-per-testi-lunghi-audiolibri)
8. [Schema Task ARIA — Payload Qwen3-TTS](#8-schema-task-aria--payload-qwen3-tts)
9. [Implementazione Backend `qwen3_tts.py`](#9-implementazione-backend-qwen3_ttspy)
10. [Avvio Server HTTP Locale](#10-avvio-server-http-locale)
11. [Configurazione ARIA (`config.yaml`)](#11-configurazione-aria-configyaml)
12. [Aggiornamento DIAS — SceneDirector per Qwen3](#12-aggiornamento-dias--scenedirector-per-qwen3)
13. [Setup Test Comparativo A/B con Fish](#13-setup-test-comparativo-ab-con-fish)
14. [Problemi Noti e Workaround](#14-problemi-noti-e-workaround)
15. [Roadmap Sviluppo](#15-roadmap-sviluppo)

---

## 1. Perché Qwen3-TTS come Candidato

### Il problema che si vuole risolvere rispetto a Fish S1-mini

Fish Audio S1-mini è il modello con il ranking più alto su TTS-Arena-V2 (ELO 1339)
e gira perfettamente sull'hardware attuale. Tuttavia presenta due limitazioni
architetturali che non possono essere risolte senza cambiare modello:

1. **Pause e break non nativi**: i tag `(break)` e `(long-break)` non sono
   implementati nel server self-hosted v0.1.0-S1. Il workaround via split+concatenazione
   funziona ma aggiunge complessità e latenza.

2. **Accenti italiani statistici**: Fish non usa un dizionario fonetico ma modellazione
   probabilistica. Parole rare, termini storici e nomi propri vengono accentati per
   approssimazione statistica, non per regola linguistica.

### Cosa offre Qwen3-TTS in aggiunta

| Capacità | Fish S1-mini | Qwen3-TTS 1.7B |
|---|---|---|
| Pause native nel testo | ❌ workaround | ✅ nativo |
| Controllo ritmo/velocità | ❌ no | ✅ via istruzione naturale |
| Controllo emozione | ✅ tag `(emotion)` | ✅ istruzione naturale |
| Italiano nel training set | 🟡 statistico | ✅ lingua dichiarata |
| Voice cloning da sample | ✅ 10-30s | ✅ 3-10s |
| Compatibilità RTX 5060 Ti sm_120 | ✅ cu128 | ✅ cu128 (stesso wheel) |
| Setup Windows nativo | ✅ Python 3.10 | ✅ Python 3.12 |
| Sviluppo attivo | ✅ | ✅ (team Alibaba Qwen) |
| Licenza | CC-BY-NC-4.0 | Apache 2.0 |

### Limiti noti di Qwen3-TTS (da tenere presenti nel test)

- Voice cloning cross-linguale (sample inglese → testo italiano) produce accento
  americano. Il tuo caso è italiano→italiano: scenario ottimale.
- Il modello 0.6B ha un bug di long-silence (pause fino a 27s su testi lunghi).
  **Usare esclusivamente il modello 1.7B per audiolibri.**
- Primo token generato può avere bleeding dell'ultimo fonema del sample di riferimento.
  Workaround documentato: aggiungere 0.5s di silenzio alla fine del ref.wav.
- Non ha emotion tag discreti come Fish (`(scared)`, `(hesitating)`). Il controllo
  emotivo avviene tramite istruzione in linguaggio naturale nel campo `instruct`.

---

## 2. Architettura Tecnica Interna

Qwen3-TTS usa un'architettura **autoregressive transformer con speech tokenizer a 12Hz**:

```
TESTO INPUT + ISTRUZIONE STILE
         │
         ▼
┌────────────────────────────────┐
│  LLM TRANSFORMER (1.7B param)  │  ← interpreta testo + istruzione stile
│  "cosa dire e come dirlo"      │    genera discrete speech codes a 12 token/s
└────────────────┬───────────────┘
                 │ speech tokens (12Hz)
                 ▼
┌────────────────────────────────┐
│  DECODER (vocoder neurale)     │  ← converte speech codes → waveform PCM
│  "waveform finale"             │
└────────────────┬───────────────┘
                 │
                 ▼
             WAV OUTPUT (24kHz o 16kHz)
```

**Differenza chiave rispetto a Fish Dual-AR**: Qwen3 usa un singolo transformer
grande che integra semantica e stile vocale nello stesso forward pass, guidato
dall'istruzione in linguaggio naturale. Fish usa due transformer separati (semantico +
acustico) con emotion tag discreti. Né superiore né inferiore: sono approcci diversi
con trade-off diversi.

**Non-streaming mode per audiolibri**: il parametro `non_streaming_mode=True`
inserisce l'intero testo nel prefill prima di generare. Questo migliora la coerenza
prosodica su frasi lunghe. Usare sempre questa modalità per DIAS.

---

## 3. Setup Ambiente — Windows 11 Nativo

### Prerequisiti

- Windows 11 con driver NVIDIA recenti (CUDA 12.8+)
- Miniconda installato in `C:\Users\gemini\miniconda3`
- RTX 5060 Ti 16GB VRAM (architettura Blackwell sm_120)
- Python 3.12 (diverso da Fish che usa Python 3.10 — ambienti completamente separati)

> ⚠️ **NON usare Python 3.10 per Qwen3-TTS**. Il modello richiede Python 3.12+
> per compatibilità con cu128 e flash-attention 2. Gli ambienti sono isolati:
> `fish-speech` (Python 3.10) e `qwen3-tts` (Python 3.12) coesistono senza conflitti.

### Creazione ambiente conda

```cmd
:: Apri Anaconda Prompt come amministratore
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
```

### Installazione PyTorch cu128 (critico per sm_120)

Questo è lo stesso wheel usato per Fish. Su RTX 5060 Ti è obbligatorio:

```cmd
pip install torch torchvision torchaudio ^
  --index-url https://download.pytorch.org/whl/cu128
```

Verifica installazione:
```python
import torch
print(torch.__version__)           # deve essere 2.7.0+cu128 o superiore
print(torch.cuda.is_available())   # deve essere True
print(torch.cuda.get_device_name(0))  # RTX 5060 Ti
```

### Dipendenze Qwen3-TTS

Qwen3-TTS 1.7B Base richiede il pacchetto ufficiale `qwen-tts` (che installa
sotto il cofano `transformers 4.57.3` e `accelerate`). Il modello non è
pienamente supportato dalle versioni stabili di `transformers` standard.

```cmd
pip install -U qwen-tts
pip install fastapi uvicorn requests soundfile

:: Flash attention (opzionale ma raccomandato — riduce VRAM e latenza)
:: Scarica il wheel precompilato corretto per Python 3.12 + cu128
:: Da: https://github.com/Dao-AILab/flash-attention/releases
pip install flash_attn-2.8.3+cu128torch2.8.0-cp312-cp312-win_amd64.whl
```

> **Nota flash-attention**: Se il wheel non è disponibile per la tua versione
> esatta, omettere. Il modello funziona ugualmente (RTF leggermente superiore
> come ~6x invece di ~4x). Non è un prerequisito bloccante.


### Struttura directory

```
C:\
├── models\
│   ├── fish-s1-mini\          ← già presente
│   └── qwen3-tts-1.7b\        ← nuovo, da creare con download
│
├── Users\gemini\aria\
│   ├── data\
│   │   └── voices\
│   │       └── luca\
│   │           ├── ref.wav         ← sample originale
│   │           ├── ref_padded.wav  ← sample con 0.5s silenzio finale (da creare)
│   │           └── ref.txt         ← trascrizione esatta del sample
│   └── backends\
│       ├── fish_tts.py             ← già presente
│       └── qwen3_tts.py            ← da creare (questo progetto)
│
└── fish-speech\               ← già presente
```

---

## 4. Download e Configurazione Modello

### Quale variante scaricare

Qwen3-TTS esiste in 4 varianti su Hugging Face. Per DIAS audiolibri, usare
**esclusivamente** `Qwen3-TTS-12Hz-1.7B-Base`:

| Variante HF | Parametri | VRAM | Uso | Note |
|---|---|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | ~6GB | **Voice cloning** | ✅ Per DIAS |
| `Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | ~4GB | Voice cloning | ❌ Bug long-silence |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B | ~6GB | Speaker predefiniti | ❌ No cloning |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | 0.6B | ~4GB | Speaker predefiniti | ❌ No cloning |

### Download

```cmd
conda activate qwen3-tts

:: Metodo 1: huggingface-cli (raccomandato)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base ^
  --local-dir C:\models\qwen3-tts-1.7b

:: Metodo 2: Python (alternativa)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-Base',
    local_dir='C:/models/qwen3-tts-1.7b'
)
"
```

Dimensione attesa: ~3.5GB (pesi in bfloat16).

### Test inferenza diretta (prima di integrare in ARIA)

Il modello Base supporta solo il voice cloning. Per testarlo in modalità **Zero-Shot**
(senza passare il testo del sample `ref_text`, come fa ARIA), è obbligatorio
passare `x_vector_only_mode=True`, altrimenti il modello andrà in crash cercando
di eseguire un In-Context Learning ibrido. 

```python
# test_qwen3_direct.py
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "C:/models/qwen3-tts-1.7b"

print(f"Caricamento modello su {device}...")
model = Qwen3TTSModel.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # omettere se non installato
    device_map=device
)
print(f"VRAM usata: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Test voice cloning dal sample del luca in Zero-Shot mode
ref_audio, ref_sr = sf.read("C:/Users/gemini/aria/data/voices/luca/ref_padded.wav")
# generate_voice_clone accetta tupla (array, sample_rate)
ref_input = (ref_audio, ref_sr)

wavs_cloned, sr = model.generate_voice_clone(
    text="La pàtina del tempo rivelava i segni di un'epoca lontana.",
    ref_audio=ref_input,
    ref_text=None,
    x_vector_only_mode=True,       # <--- CRITICO PER ZERO-SHOT
    instruct="Warm male voice, Italian audiobook narrator, calm and measured.",
    non_streaming_mode=True
)
sf.write("test_cloned.wav", wavs_cloned[0], sr)
print("Scritto: test_cloned.wav")
```

---

## 5. Gestione Voice Sample e Ref.txt per Italiano

### Il problema del bleeding fonetico

Qwen3-TTS ha un comportamento noto: il primo token generato si condiziona
sull'ultimo fonema del sample audio di riferimento. Se il sample finisce
bruscamente, si sente un artefatto sonoro all'inizio dell'audio generato.

**Fix automatizzato**: Lo script `scripts/voice_prepper.py` (che eseguiamo da LXC per estrarre l'audio da YouTube e trascriverlo con Gemini) **genera automaticamente**, oltre al `ref.wav` originale, la versione `ref_padded.wav` con 0.5 secondi di silenzio. Dunque non devi fare nessuna elaborazione manuale.

### Lunghezza ottimale del sample

| Durata sample | Qualità cloning | Note |
|---|---|---|
| < 3s | ❌ insufficiente | Modello non ha abbastanza caratteristiche vocali |
| 3-10s | ✅ buona | Range consigliato per test rapidi |
| 10-20s | ✅ ottima | Range ottimale per produzione |
| > 30s | ⚠️ rischio | Può causare loop di generazione. Hard cap a 30s |

**Raccomandazione**: usare il campione del luca tra 10 e 20 secondi.
Se il `ref.wav` attuale è più lungo, tagliarlo a 20s prima di creare `ref_padded.wav`.

### Il campo ref.txt

A differenza di Fish (che usa il ref.txt come hint per il modello semantico),
Qwen3-TTS usa il ref.txt opzionalmente per migliorare l'allineamento fonetico
del sample. Va passato solo se si conosce la trascrizione **esatta** del sample.

**Regola**: se non sei sicuro al 100% delle parole esatte pronunciate nel sample,
**non passare il ref.txt**. Un ref.txt sbagliato degrada la qualità più che
non passarlo. Qwen3 funziona bene anche senza.

### Struttura Voice Library per Qwen3

```
C:\Users\gemini\aria\data\voices\
├── luca\
│   ├── ref.wav             ← sample originale (10-20s)
│   ├── ref_padded.wav      ← sample + 0.5s silenzio (usato da Qwen3)
│   └── ref.txt             ← trascrizione esatta (opzionale, solo se certa)
│
└── (eventuali altri personaggi, stessa struttura)
```

---

## 6. Controllo Prosodia ed Emozione in Italiano

### La differenza fondamentale rispetto a Fish

Fish usa **emotion tag discreti** inseriti nel testo: `(scared)Aprì la porta.`
Qwen3 usa un **campo istruzione separato** in linguaggio naturale: il testo
rimane pulito, l'emozione è nel campo `instruct`.

Questo ha un vantaggio per DIAS: il copione testuale generato da Gemini
(Stage C — SceneDirector) non deve essere modificato per inserire tag. I tag
emotivi diventano metadati separati nel payload del task.

### Sintassi del campo `instruct`

Il campo `instruct` è una stringa in inglese (per massima affidabilità) che
descrive la voce desiderata. Combinare sempre questi elementi:

```
"{genere} {timbro}, {ruolo}, {emozione corrente}, {ritmo}."
```

### Mappa Emozioni DIAS → Istruzione Qwen3

Questa è la mappa di traduzione da usare nello SceneDirector quando genera
i task per il backend Qwen3. Il campo `primary_emotion` dell'analisi DIAS
viene convertito nell'istruzione corrispondente:

| primary_emotion (DIAS) | instruct Qwen3 (italiano→inglese) |
|---|---|
| `neutral` | `"Warm male voice, Italian audiobook narrator, calm and measured, moderate pace."` |
| `suspense` | `"Warm male voice, Italian audiobook narrator, tense and restrained, slightly slower pace, hushed intensity."` |
| `fear` | `"Warm male voice, Italian audiobook narrator, anxious and cautious, slow deliberate pace, quiet."` |
| `sadness` | `"Warm male voice, Italian audiobook narrator, melancholic and subdued, slow pace, gentle."` |
| `joy` | `"Warm male voice, Italian audiobook narrator, warm and bright, energetic, slightly faster pace."` |
| `anger` | `"Warm male voice, Italian audiobook narrator, firm and intense, controlled anger, strong pace."` |
| `curiosity` | `"Warm male voice, Italian audiobook narrator, inquisitive and engaged, moderate pace, slightly raised."` |

### Controllo pause native (vantaggio su Fish)

Qwen3 rispetta la punteggiatura italiana in modo nativo. Queste istruzioni
aggiuntive nel campo `instruct` attivano comportamenti prosodici specifici:

```python
# Pause lunghe tra sezioni
"...pause naturally at paragraph breaks, longer silence between chapters."

# Testo drammatico con pause teatrali
"...dramatic pauses on ellipsis and em-dashes, hold breath before revelations."

# Dialogo
"...differentiate narrative voice from quoted speech, slight tonal shift for dialogue."
```

### Esempio istruzione completa per una scena di suspense

```python
instruct = (
    "Warm Italian male voice, professional audiobook narrator, "
    "tense and restrained atmosphere, hushed intensity as if revealing a secret, "
    "slow deliberate pace, pause naturally at ellipsis and commas, "
    "slight drop in volume toward sentence endings."
)
```

---

## 7. Chunking per Testi Lunghi (Audiolibri)

### Limite contestuale del modello

Qwen3-TTS 1.7B ha un limite pratico di circa **300-400 parole per inferenza**
prima che la coerenza prosodica degradi. Per le scene DIAS (max 300 parole
per specifica in `pipeline.scene_max_words`) questo è normalmente sufficiente.

Per sicurezza, il backend implementa chunking automatico trasparente al chiamante.

### Strategia di chunking

```python
def chunk_text_for_qwen3(text: str, max_words: int = 250) -> list[str]:
    """
    Splitta il testo in chunk da max_words parole.
    Split sempre su confini di frase (punto, punto esclamativo, punto interrogativo).
    NON spezzare mai a metà frase.
    """
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_words = 0
    
    for sentence in sentences:
        words = len(sentence.split())
        if current_words + words > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_words = words
        else:
            current_chunk.append(sentence)
            current_words += words
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
```

### Concatenazione output

I chunk vengono sintetizzati sequenzialmente (stessa ref audio, stessa
istruzione di stile per coerenza timbrica) e concatenati in un singolo WAV.
Tra i chunk viene inserito un silenzio di 80ms per simulare il respiro naturale.

```python
def concatenate_chunks(wav_list: list, sr: int, gap_ms: int = 80) -> np.ndarray:
    gap_samples = int(sr * gap_ms / 1000)
    gap = np.zeros(gap_samples)
    result = []
    for i, wav in enumerate(wav_list):
        result.append(wav)
        if i < len(wav_list) - 1:
            result.append(gap)
    return np.concatenate(result)
```

---

## 8. Schema Task ARIA — Payload Qwen3-TTS

Il task segue lo schema ARIA standard. La coda Redis è:

```
gpu:queue:tts:qwen3-tts-1.7b
```

### Task completo (inviato da DIAS VoiceGenerator)

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440001",
  "client_id": "dias-minipc",
  "client_ip": "192.168.1.10",
  "client_version": "1.0.0",

  "model_type": "tts",
  "model_id": "qwen3-tts-1.7b",

  "queued_at": "2026-03-04T10:00:00Z",
  "priority": 1,
  "timeout_seconds": 3600,

  "callback_key": "gpu:result:dias-minipc:550e8400-e29b-41d4-a716-446655440001",

  "file_refs": {
    "input": [
      {
        "ref_id": "voice_ref_audio",
        "local_path": "C:\\Users\\gemini\\aria\\data\\voices\\luca\\ref_padded.wav",
        "size_bytes": 441000
      }
    ],
    "output": [
      {
        "ref_id": "audio_output",
        "expected_filename": "550e8400-e29b-41d4-a716-446655440001.wav",
        "server_delivery": "http"
      }
    ]
  },

  "payload": {
    "text": "La pàtina del tempo rivelava i segreti di un'epoca lontana. Adso avanzò con cautela tra le rovine fumanti dell'abbazia.",

    "voice_ref_audio_path": "C:\\Users\\gemini\\aria\\data\\voices\\luca\\ref_padded.wav",
    "voice_ref_text": null,

    "language": "Italian",

    "instruct": "Warm Italian male voice, professional audiobook narrator, tense and restrained atmosphere, slow deliberate pace, pause naturally at ellipsis and commas.",

    "non_streaming_mode": true,
    "max_new_tokens": 4096,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.1,

    "output_sample_rate": 24000,

    "chunking": {
      "enabled": true,
      "max_words_per_chunk": 250,
      "gap_between_chunks_ms": 80
    },

    "scene_metadata": {
      "scene_id": "scene-ch003-01",
      "book_id": "7d9a3c1e-5f4a-4b8e-9c2d-1a3b5c7d9e0f",
      "primary_emotion": "suspense",
      "arousal": 0.75,
      "tension": 0.90,
      "pace_factor": 0.85
    }
  }
}
```

### Campi payload — Descrizione

| Campo | Tipo | Obbligatorio | Descrizione |
|---|---|---|---|
| `text` | string | ✅ | Testo da sintetizzare. Testo pulito, senza tag emotivi. |
| `voice_ref_audio_path` | string | ✅ | Path assoluto Windows al `ref_padded.wav` |
| `voice_ref_text` | string\|null | ❌ | Trascrizione sample. Passare `null` se non certa. |
| `language` | string | ✅ | Sempre `"Italian"` per DIAS |
| `instruct` | string | ✅ | Istruzione stile in inglese. Vedi §6 per la mappa. |
| `non_streaming_mode` | bool | ✅ | Sempre `true` per audiolibri |
| `max_new_tokens` | int | ❌ | Default 4096. Aumentare per scene molto lunghe. |
| `temperature` | float | ❌ | Default 0.7. Abbassare (0.5) per maggiore stabilità. |
| `top_p` | float | ❌ | Default 0.9 |
| `repetition_penalty` | float | ❌ | Default 1.1. Previene loop di ripetizione. |
| `output_sample_rate` | int | ❌ | 24000 o 16000. Default 24000. |
| `chunking.enabled` | bool | ❌ | Default true. Disabilitare solo per scene <100 parole. |
| `chunking.max_words_per_chunk` | int | ❌ | Default 250. |
| `chunking.gap_between_chunks_ms` | int | ❌ | Silenzio tra chunk. Default 80ms. |
| `scene_metadata` | object | ❌ | Metadati per logging e debug. Non usati per inferenza. |

### Risultato (scritto dal Server in Redis)

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440001",
  "status": "completed",
  "model_id": "qwen3-tts-1.7b",
  "completed_at": "2026-03-04T10:02:30Z",

  "output": {
    "audio_url": "http://192.168.1.139:8082/outputs/550e8400-e29b-41d4-a716-446655440001.wav",
    "local_path": "C:\\Users\\gemini\\aria\\outputs\\550e8400-e29b-41d4-a716-446655440001.wav",
    "duration_seconds": 18.4,
    "sample_rate": 24000,
    "chunks_count": 2
  },

  "metrics": {
    "inference_time_seconds": 28.6,
    "rtf": 1.55,
    "vram_peak_gb": 6.2,
    "chunks_processed": 2
  }
}
```

---

## 9. Implementazione Backend `qwen3_tts.py`

Seguendo il pattern **External HTTP Backend** già stabilito per Fish, il backend
Qwen3 gira come processo Python standalone con FastAPI nel suo ambiente conda,
ed ARIA lo chiama via HTTP sulla porta `8083`.

### Server FastAPI standalone (`qwen3_server.py`)

```python
# C:\Users\gemini\aria\qwen3_server.py
# Avviare con: conda activate qwen3-tts && python qwen3_server.py

import os
import time
import logging
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from qwen_tts import Qwen3TTSModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen3-tts-server")

MODEL_PATH = os.getenv("QWEN3_MODEL_PATH", "C:/models/qwen3-tts-1.7b")
OUTPUT_DIR = os.getenv("ARIA_OUTPUT_DIR", "C:/Users/gemini/aria/outputs")
HOST = os.getenv("QWEN3_HOST", "0.0.0.0")
PORT = int(os.getenv("QWEN3_PORT", "8083"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Qwen3-TTS Server", version="1.0.0")
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    global model
    logger.info(f"Caricamento Qwen3-TTS 1.7B su {device}...")
    try:
        model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device,
        )
        vram = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
        logger.info(f"Modello caricato (FlashAttention). VRAM: {vram:.2f} GB")
    except Exception as e:
        logger.warning(f"Flash attention non disponibile ({e}), caricamento standard.")
        model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            device_map=device,
        )


def chunk_text(text: str, max_words: int = 250) -> list[str]:
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
    chunks, current_chunk, current_words = [], [], 0
    for sentence in sentences:
        words = len(sentence.split())
        if current_words + words > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk, current_words = [sentence], words
        else:
            current_chunk.append(sentence)
            current_words += words
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def concatenate_wavs(wav_list: list, sr: int, gap_ms: int = 80) -> np.ndarray:
    gap = np.zeros(int(sr * gap_ms / 1000))
    result = []
    for i, wav in enumerate(wav_list):
        result.append(wav)
        if i < len(wav_list) - 1:
            result.append(gap)
    return np.concatenate(result)


class TTSRequest(BaseModel):
    text: str
    voice_ref_audio_path: str
    voice_ref_text: Optional[str] = None
    language: str = "Italian"
    instruct: str = "Warm Italian male voice, professional audiobook narrator, calm and measured."
    non_streaming_mode: bool = True
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    output_sample_rate: int = 24000
    max_words_per_chunk: int = 250
    gap_between_chunks_ms: int = 80
    output_filename: str = "output.wav"


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Modello non caricato")
    vram = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
    return {"status": "ok", "device": device, "vram_gb": round(vram, 2)}


@app.post("/tts")
def synthesize(req: TTSRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modello non caricato")

    t_start = time.time()

    # Carica audio di riferimento
    try:
        ref_audio, ref_sr = sf.read(req.voice_ref_audio_path)
        if ref_audio.ndim > 1:
            ref_audio = ref_audio[:, 0]  # mono
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore lettura ref audio: {e}")

    # Chunking
    chunks = chunk_text(req.text, req.max_words_per_chunk)
    logger.info(f"Testo suddiviso in {len(chunks)} chunk")

    wav_chunks = []
    output_sr = req.output_sample_rate

    for i, chunk_text_part in enumerate(chunks):
        logger.info(f"Chunk {i+1}/{len(chunks)}: {len(chunk_text_part.split())} parole")
        try:
            is_x_vector_only = (req.voice_ref_text is None)
            ref_input = (ref_audio, ref_sr)
            
            wavs, sr = model.generate_voice_clone(
                text=chunk_text_part,
                ref_audio=ref_input,
                ref_text=req.voice_ref_text,
                language=req.language,
                instruct=req.instruct,
                non_streaming_mode=req.non_streaming_mode,
                x_vector_only_mode=is_x_vector_only,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
            )
            output_sr = sr
            wav_chunks.append(wavs[0] if isinstance(wavs, list) else wavs)
        except Exception as e:
            logger.error(f"Errore chunk {i+1}: {e}")
            raise HTTPException(status_code=500, detail=f"Errore inferenza chunk {i+1}: {e}")

    # Concatenazione
    final_wav = concatenate_wavs(wav_chunks, output_sr, req.gap_between_chunks_ms)

    # Salvataggio
    out_path = os.path.join(OUTPUT_DIR, req.output_filename)
    sf.write(out_path, final_wav, output_sr)

    inference_time = time.time() - t_start
    duration = len(final_wav) / output_sr
    rtf = inference_time / duration if duration > 0 else 0

    logger.info(f"Completato: {duration:.1f}s audio in {inference_time:.1f}s (RTF {rtf:.2f}x)")

    return {
        "status": "ok",
        "output_path": out_path,
        "duration_seconds": round(duration, 2),
        "sample_rate": output_sr,
        "chunks_count": len(chunks),
        "inference_time_seconds": round(inference_time, 2),
        "rtf": round(rtf, 2),
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else 0
    }


@app.get("/outputs/{filename}")
def get_output(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File non trovato")
    return FileResponse(path, media_type="audio/wav")


if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host=HOST, port=PORT)
```

### Backend ARIA `qwen3_tts.py`

```python
# C:\Users\gemini\aria\aria_server\backends\qwen3_tts.py
# Seguendo il pattern External HTTP Backend (identico a fish_tts.py)

import uuid
import time
import logging
import requests
from .base_backend import BaseBackend

logger = logging.getLogger("aria.backend.qwen3_tts")

QWEN3_SERVER_URL = "http://127.0.0.1:8083"

EMOTION_TO_INSTRUCT = {
    "neutral":   "Warm male voice, Italian audiobook narrator, calm and measured, moderate pace.",
    "suspense":  "Warm male voice, Italian audiobook narrator, tense and restrained, slightly slower pace, hushed intensity.",
    "fear":      "Warm male voice, Italian audiobook narrator, anxious and cautious, slow deliberate pace, quiet.",
    "sadness":   "Warm male voice, Italian audiobook narrator, melancholic and subdued, slow pace, gentle.",
    "joy":       "Warm male voice, Italian audiobook narrator, warm and bright, energetic, slightly faster pace.",
    "anger":     "Warm male voice, Italian audiobook narrator, firm and intense, controlled anger, strong pace.",
    "curiosity": "Warm male voice, Italian audiobook narrator, inquisitive and engaged, moderate pace, slightly raised.",
}

DEFAULT_INSTRUCT = EMOTION_TO_INSTRUCT["neutral"]


class Qwen3TTSBackend(BaseBackend):
    model_id = "qwen3-tts-1.7b"
    model_type = "tts"

    def load(self, model_path: str, config: dict) -> None:
        """Health check: verifica che il server qwen3 sia raggiungibile."""
        try:
            r = requests.get(f"{QWEN3_SERVER_URL}/health", timeout=5)
            r.raise_for_status()
            logger.info(f"Qwen3-TTS server OK: {r.json()}")
        except Exception as e:
            raise RuntimeError(f"Qwen3-TTS server non raggiungibile su {QWEN3_SERVER_URL}: {e}")

    def unload(self) -> None:
        """No-op: il server è un processo esterno gestito dallo script .bat"""
        logger.info("Qwen3-TTS backend: unload (no-op, processo esterno)")

    def estimated_vram_gb(self) -> float:
        return 6.5

    def run(self, payload: dict) -> dict:
        text = payload.get("text", "")
        if not text:
            raise ValueError("Campo 'text' obbligatorio nel payload")

        # Risoluzione voice ref (ARIA-side, già iniettato dall'orchestratore)
        voice_ref_path = payload.get("voice_ref_audio_path", "")
        if not voice_ref_path:
            raise ValueError("Campo 'voice_ref_audio_path' obbligatorio")

        # Costruzione istruzione stile
        instruct = payload.get("instruct")
        if not instruct:
            emotion = payload.get("scene_metadata", {}).get("primary_emotion", "neutral")
            pace_factor = payload.get("scene_metadata", {}).get("pace_factor", 1.0)
            instruct = EMOTION_TO_INSTRUCT.get(emotion, DEFAULT_INSTRUCT)
            if pace_factor < 0.8:
                instruct += " Very slow and deliberate pace."
            elif pace_factor > 1.2:
                instruct += " Slightly faster pace."

        job_id = payload.get("job_id", str(uuid.uuid4()))
        output_filename = f"{job_id}.wav"

        # Chiamata al server
        request_body = {
            "text": text,
            "voice_ref_audio_path": voice_ref_path,
            "voice_ref_text": payload.get("voice_ref_text"),
            "language": payload.get("language", "Italian"),
            "instruct": instruct,
            "non_streaming_mode": payload.get("non_streaming_mode", True),
            "max_new_tokens": payload.get("max_new_tokens", 4096),
            "temperature": payload.get("temperature", 0.7),
            "top_p": payload.get("top_p", 0.9),
            "repetition_penalty": payload.get("repetition_penalty", 1.1),
            "output_sample_rate": payload.get("output_sample_rate", 24000),
            "max_words_per_chunk": payload.get("chunking", {}).get("max_words_per_chunk", 250),
            "gap_between_chunks_ms": payload.get("chunking", {}).get("gap_between_chunks_ms", 80),
            "output_filename": output_filename,
        }

        timeout = payload.get("timeout_seconds", 3600)

        try:
            response = requests.post(
                f"{QWEN3_SERVER_URL}/tts",
                json=request_body,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Timeout ({timeout}s) durante inferenza Qwen3-TTS")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Errore chiamata Qwen3-TTS server: {e}")

        output_path = result["output_path"]
        audio_url = f"http://192.168.1.139:8083/outputs/{output_filename}"

        return {
            "audio_url": audio_url,
            "local_path": output_path,
            "duration_seconds": result.get("duration_seconds"),
            "sample_rate": result.get("sample_rate"),
            "chunks_count": result.get("chunks_count"),
            "metrics": {
                "inference_time_seconds": result.get("inference_time_seconds"),
                "rtf": result.get("rtf"),
                "vram_peak_gb": result.get("vram_peak_gb"),
            }
        }

    def is_loaded(self) -> bool:
        try:
            r = requests.get(f"{QWEN3_SERVER_URL}/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False
```

---

## 10. Avvio Server HTTP Locale

### Script batch di avvio

```batch
:: start-qwen3-tts.bat
:: Posizionare in: C:\Users\gemini\aria\start-qwen3-tts.bat

@echo off
title Qwen3-TTS Server (porta 8083)

echo [%DATE% %TIME%] Avvio Qwen3-TTS Server...

:: Attendi che Redis e ARIA siano già in ascolto
timeout /t 30 /nobreak > nul

:: Attiva ambiente conda
call C:\Users\gemini\miniconda3\Scripts\activate.bat qwen3-tts

:: Variabili ambiente
set QWEN3_MODEL_PATH=C:\models\qwen3-tts-1.7b
set ARIA_OUTPUT_DIR=C:\Users\gemini\aria\outputs
set QWEN3_HOST=0.0.0.0
set QWEN3_PORT=8083

:: Avvia server
echo [%DATE% %TIME%] Caricamento modello (prima volta: ~30-60s)...
python C:\Users\gemini\aria\qwen3_server.py

echo [%DATE% %TIME%] Server terminato.
pause
```

### Task Scheduler Windows (avvio automatico)

Seguendo lo stesso pattern di Fish S1-mini (già configurato):

```
Nome task:         Qwen3-TTS Server
Trigger:           All'avvio del sistema
Azione:            Avvia programma: C:\Users\gemini\aria\start-qwen3-tts.bat
Ritardo avvio:     90 secondi (Fish parte a 60s, Qwen3 parte dopo)
Esegui come:       gemini (utente corrente)
Opzione:           Esegui che l'utente sia connesso o meno
```

### Porte utilizzate — Mappa aggiornata

| Servizio | Porta | Env Conda | Script Avvio |
|---|---|---|---|
| Fish S1-mini TTS | 8080 | `fish-speech` | `start-fish-api.bat` |
| Fish Voice Cloning (VQGAN) | 8081 | `fish-voice-cloning` | (esistente) |
| HTTP Asset Server | 8082 | (base) | (esistente) |
| **Qwen3-TTS 1.7B** | **8083** | **`qwen3-tts`** | **`start-qwen3-tts.bat`** ← nuovo |

---

## 11. Configurazione ARIA (`config.yaml`)

Aggiungere la sezione Qwen3 nel blocco `models.tts` esistente:

```yaml
models:
  tts:
    fish-s1-mini:
      enabled: true
      model_path: "aria/data/models/fish-s1-mini"
      estimated_vram_gb: 4.0
      server_url: "http://127.0.0.1:8080"
      max_retries: 2

    # NUOVO — Backend Qwen3-TTS per test comparativo
    qwen3-tts-1.7b:
      enabled: true
      model_path: "C:/models/qwen3-tts-1.7b"
      estimated_vram_gb: 6.5
      server_url: "http://127.0.0.1:8083"
      max_retries: 2
      notes: "Backend sperimentale. Test comparativo con Fish S1-mini."

    orpheus-3b:
      enabled: false
      model_path: "aria/data/models/orpheus-3b-q4"
      estimated_vram_gb: 7.0
      max_retries: 2
```

> ⚠️ **VRAM e coesistenza**: Fish usa ~4GB, Qwen3 usa ~6.5GB. Con 16GB VRAM
> non possono essere caricati contemporaneamente con MusicGen (4GB).
> Il BatchOptimizer dovrà scaricare un modello prima di caricare l'altro.
> Questo è il comportamento normale — un modello alla volta per design.

---

## 12. Aggiornamento DIAS — SceneDirector per Qwen3

### Quando DIAS invia a Fish vs quando invia a Qwen3

Durante la fase di test comparativo, il VoiceGenerator (Stage D) di DIAS
deve poter instradare lo stesso task verso entrambi i backend. Il modo più
semplice è un flag di configurazione in DIAS:

```yaml
# dias/config.yaml
voice_backend:
  primary: "fish-s1-mini"      # backend in produzione
  experimental: "qwen3-tts-1.7b"  # backend in test
  ab_test_mode: false           # se true: invia ogni task a entrambi i backend
  ab_test_ratio: 0.5            # 50% dei task al backend sperimentale
```

### Adattamento payload SceneDirector per Qwen3

Il SceneDirector genera già un payload con `voice_direction.emotion_description`
e `voice_direction.pace_factor`. Bisogna solo aggiungere la traduzione
nell'istruzione Qwen3. Questo avviene nel VoiceGenerator (Stage D), non
nel SceneDirector — il copione resta invariato.

```python
# In dias/stages/voice_generator.py

EMOTION_TO_QWEN3_INSTRUCT = {
    "neutral":   "Warm male voice, Italian audiobook narrator, calm and measured, moderate pace.",
    "suspense":  "Warm male voice, Italian audiobook narrator, tense and restrained, slightly slower pace, hushed intensity.",
    "fear":      "Warm male voice, Italian audiobook narrator, anxious and cautious, slow deliberate pace, quiet.",
    "sadness":   "Warm male voice, Italian audiobook narrator, melancholic and subdued, slow pace, gentle.",
    "joy":       "Warm male voice, Italian audiobook narrator, warm and bright, energetic, slightly faster pace.",
    "anger":     "Warm male voice, Italian audiobook narrator, firm and intense, controlled anger, strong pace.",
    "curiosity": "Warm male voice, Italian audiobook narrator, inquisitive and engaged, moderate pace, slightly raised.",
}

def build_qwen3_payload(scene_script: dict, voice_config: dict) -> dict:
    """
    Costruisce il payload ARIA per Qwen3-TTS a partire dal copione DIAS.
    Chiamato dal VoiceGenerator quando model_id == 'qwen3-tts-1.7b'.
    """
    emotion = scene_script["block_analysis"]["primary_emotion"]
    pace_factor = scene_script["voice_direction"]["pace_factor"]
    
    base_instruct = EMOTION_TO_QWEN3_INSTRUCT.get(emotion, EMOTION_TO_QWEN3_INSTRUCT["neutral"])
    
    if pace_factor < 0.8:
        base_instruct += " Very slow and deliberate pace."
    elif pace_factor > 1.2:
        base_instruct += " Slightly faster pace."

    return {
        "text": scene_script["text_content"],
        "voice_ref_audio_path": voice_config["ref_padded_wav_path"],
        "voice_ref_text": voice_config.get("ref_text"),  # None se non certo
        "language": "Italian",
        "instruct": base_instruct,
        "non_streaming_mode": True,
        "chunking": {
            "enabled": True,
            "max_words_per_chunk": 250,
            "gap_between_chunks_ms": 80
        },
        "scene_metadata": {
            "scene_id": scene_script["scene_id"],
            "book_id": scene_script.get("book_id"),
            "primary_emotion": emotion,
            "pace_factor": pace_factor
        }
    }
```

---

## 13. Setup Test Comparativo A/B con Fish

### Obiettivo del test

Prima di decidere se Qwen3 sostituisce Fish in produzione, vanno confrontati
direttamente sugli stessi testi. Il test deve valutare:

1. **Qualità timbrica del clone**: la voce del luca è riconoscibile?
2. **Accenti italiani**: parole rare, termini storici, nomi propri
3. **Prosodia e ritmo**: le pause sono naturali? Il ritmo segue l'emozione?
4. **Stabilità**: ci sono artefatti, silenzi anomali, ripetizioni?
5. **Velocità**: RTF su RTX 5060 Ti (obiettivo: RTF > 1.0x in entrambi)

### Script di test comparativo

```python
# test_ab_comparison.py
# Genera lo stesso testo con entrambi i backend e salva in cartelle separate

import json
import requests
import time
from pathlib import Path

TEST_TEXTS = [
    {
        "id": "test_neutral",
        "text": "Era il tramonto del quinto giorno quando Guglielmo di Baskerville giunse all'abbazia. Le torri si stagliavano scure contro un cielo color cenere.",
        "emotion": "neutral",
        "pace_factor": 1.0
    },
    {
        "id": "test_suspense",
        "text": "Adso avanzò nel corridoio buio. Qualcosa scricchiolò. Si fermò, trattenendo il respiro... poi il silenzio tornò, pesante come pietra.",
        "emotion": "suspense",
        "pace_factor": 0.8
    },
    {
        "id": "test_rare_words",
        "text": "La pàtina del manoscritto rivelava tracce di sangue. Il futòn sul quale giaceva il corpo era intriso di inchiostro e mistero.",
        "emotion": "suspense",
        "pace_factor": 0.85
    }
]

VOICE_REF_PATH = "C:/Users/gemini/aria/data/voices/luca/ref_padded.wav"
OUTPUT_DIR = Path("C:/Users/gemini/aria/test_ab_output")
OUTPUT_DIR.mkdir(exist_ok=True)

FISH_URL = "http://192.168.1.139:8080"
QWEN3_URL = "http://192.168.1.139:8083"

EMOTION_TO_QWEN3 = {
    "neutral":  "Warm male voice, Italian audiobook narrator, calm and measured, moderate pace.",
    "suspense": "Warm male voice, Italian audiobook narrator, tense and restrained, slightly slower pace, hushed intensity.",
}


def test_fish(text_item: dict) -> dict:
    """Chiama Fish S1-mini tramite la sua API nativa."""
    t_start = time.time()
    # Fish API — adatta alla struttura del tuo server Fish attuale
    response = requests.post(
        f"{FISH_URL}/v1/tts",
        json={
            "text": text_item["text"],
            "reference_id": "luca",
            "format": "wav",
            "normalize": False
        },
        timeout=600
    )
    duration = time.time() - t_start
    out_path = OUTPUT_DIR / f"fish_{text_item['id']}.wav"
    out_path.write_bytes(response.content)
    return {"path": str(out_path), "time": duration}


def test_qwen3(text_item: dict) -> dict:
    """Chiama Qwen3-TTS server."""
    t_start = time.time()
    instruct = EMOTION_TO_QWEN3.get(text_item["emotion"], EMOTION_TO_QWEN3["neutral"])
    response = requests.post(
        f"{QWEN3_URL}/tts",
        json={
            "text": text_item["text"],
            "voice_ref_audio_path": VOICE_REF_PATH,
            "language": "Italian",
            "instruct": instruct,
            "non_streaming_mode": True,
            "output_filename": f"qwen3_{text_item['id']}.wav"
        },
        timeout=600
    )
    duration = time.time() - t_start
    result = response.json()
    return {"path": result["output_path"], "time": duration, "rtf": result.get("rtf")}


if __name__ == "__main__":
    results = []
    for item in TEST_TEXTS:
        print(f"\n--- Test: {item['id']} ---")
        print(f"Testo ({len(item['text'].split())} parole): {item['text'][:60]}...")

        print("Fish S1-mini...")
        fish_result = test_fish(item)
        print(f"  → {fish_result['time']:.1f}s → {fish_result['path']}")

        print("Qwen3-TTS 1.7B...")
        qwen_result = test_qwen3(item)
        print(f"  → {qwen_result['time']:.1f}s (RTF {qwen_result.get('rtf', '?')}x) → {qwen_result['path']}")

        results.append({
            "id": item["id"],
            "fish": fish_result,
            "qwen3": qwen_result
        })

    # Salva report
    report_path = OUTPUT_DIR / "ab_test_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Test completato. Report: {report_path}")
    print(f"Audio in: {OUTPUT_DIR}")
    print("Ascolta i file fish_*.wav e qwen3_*.wav per il confronto soggettivo.")
```

---

## 14. Problemi Noti e Workaround

### P1 — Bleeding fonetico inizio audio
**Sintomo**: Artefatto sonoro nei primi 50-100ms dell'audio generato.  
**Causa**: Il primo token si condiziona sull'ultimo fonema del sample.  
**Fix**: Usare `ref_padded.wav` (sample + 0.5s silenzio finale). Già implementato nel setup §5.

### P2 — Loop infinito su sample > 30s
**Sintomo**: Il server non risponde, generazione non termina mai.  
**Causa**: `ref_audio_max_seconds` superato causa instabilità nell'attenzione.  
**Fix**: Il sample di riferimento deve essere 10-20 secondi. Se necessario, tagliare.

### P3 — Accento americano con voice cloning
**Sintomo**: La voce clonata parla italiano con accento americano o britannico.  
**Causa**: Il sample di riferimento è in inglese (cross-lingual cloning).  
**Fix**: Usare sempre un sample in italiano. Per DIAS il `luca.wav` è già in italiano: problema non applicabile.

### P4 — Qualità degradata modello 0.6B
**Sintomo**: Silenzi anomali fino a 27 secondi nell'audio, prosodia instabile.  
**Causa**: Bug noto del modello 0.6B su testi lunghi.  
**Fix**: Usare esclusivamente il modello `Qwen3-TTS-12Hz-1.7B-Base`. Mai il 0.6B per audiolibri.

### P5 — Primo avvio lento (torch.compile)
**Sintomo**: Il server impiega 2-5 minuti prima di rispondere alla prima richiesta.  
**Causa**: Compilazione JIT dei kernel CUDA per sm_120.  
**Fix**: Normale e atteso. Non interrompere. Le esecuzioni successive sono immediate.

### P6 — VRAM insufficiente se altri modelli sono caricati
**Sintomo**: `torch.cuda.OutOfMemoryError` al caricamento del modello.  
**Causa**: Qwen3 1.7B richiede ~6.5GB VRAM. Se Fish (~4GB) è caricato contemporaneamente → 10.5GB totali. Con MusicGen attivo → OOM.  
**Fix**: Il BatchOptimizer ARIA scarica il modello precedente prima di caricare Qwen3. Comportamento già gestito dal design ARIA "un modello alla volta".

---

## 15. Roadmap Sviluppo

### QW-0 — Test manuale standalone (nessuna modifica ad ARIA)

**Obiettivo**: validare Qwen3 su Windows prima di toccare qualsiasi codice ARIA.

- [ ] Setup ambiente conda `qwen3-tts` con Python 3.12 e cu128
- [ ] Download modello `Qwen3-TTS-12Hz-1.7B-Base`
- [ ] Creazione `ref_padded.wav` dal sample del luca
- [ ] Test inferenza diretta con `test_qwen3_direct.py`
- [ ] Ascolto output e verifica qualità timbrica + accenti
- [ ] Misurazione RTF su RTX 5060 Ti (obiettivo: RTF > 1.0x)

**Criterio di successo**: audio italiano riconoscibile come voce del luca,
accenti corretti su "pàtina" e "futòn", nessun artefatto evidente.  
**Stima**: 1 giorno

---

### QW-1 — Server FastAPI standalone

**Obiettivo**: `qwen3_server.py` funzionante su porta 8083.

- [ ] Implementare `qwen3_server.py` (codice in §9)
- [ ] Test health check: `GET http://localhost:8083/health`
- [ ] Test sintesi via `curl` o Postman
- [ ] Test chunking su testo da 300 parole
- [ ] Script `start-qwen3-tts.bat`
- [ ] Task Scheduler: avvio automatico con ritardo 90s

**Stima**: 1-2 giorni

---

### QW-2 — Backend ARIA `qwen3_tts.py`

**Obiettivo**: ARIA riconosce e usa `qwen3-tts-1.7b` come modello TTS.

- [ ] Implementare `aria_server/backends/qwen3_tts.py` (codice in §9)
- [ ] Aggiornare `config.yaml` con sezione `qwen3-tts-1.7b`
- [ ] Registrare `Qwen3TTSBackend` in `main.py`
- [ ] Test: push task manuale su `gpu:queue:tts:qwen3-tts-1.7b` via redis-cli
- [ ] Verifica risposta in `gpu:result:dias-minipc:{job_id}`

**File da modificare**:
```
aria_server/backends/qwen3_tts.py     ← NUOVO
aria_server/main.py                   ← registrazione backend
config.yaml                           ← sezione models.tts
```

**Stima**: 2-3 giorni

---

### QW-3 — Test comparativo A/B

**Obiettivo**: confronto oggettivo Fish vs Qwen3 sugli stessi testi DIAS.

- [ ] Eseguire `test_ab_comparison.py` con i 3 testi di test
- [ ] Ascolto cieco (senza sapere quale è quale) da parte di almeno 2 persone
- [ ] Valutazione su: timbro, accenti, prosodia, artefatti, naturalezza
- [ ] Misurazione RTF comparativo
- [ ] Decisione: Qwen3 sostituisce Fish o rimane come alternativa?

**Stima**: 2-3 giorni (incluso tempo di ascolto e valutazione)

---

### QW-4 — Integrazione DIAS completa (solo se QW-3 positivo)

**Obiettivo**: DIAS usa Qwen3 come backend TTS principale.

- [ ] Aggiornare `dias/stages/voice_generator.py` con `build_qwen3_payload()`
- [ ] Aggiornare `config.yaml` DIAS: `voice_backend.primary: "qwen3-tts-1.7b"`
- [ ] Test E2E: capitolo completo DIAS → ARIA → Qwen3 → WAV
- [ ] Verifica checkpointing e recovery su crash
- [ ] Aggiornare documentazione ARIA Blueprint (sezione 4b)

**Stima**: 1 settimana

---

## NOTE FINALI

Il copione generato da DIAS con tag Fish (`(scared)`, `(hesitating)`, ecc.)
**non è direttamente compatibile con Qwen3**, che usa istruzioni in linguaggio
naturale invece di tag discreti. Per questo il testo da passare a Qwen3 deve
essere il testo **pulito** del copione, con l'emozione tradotta nel campo
`instruct`. Questa traduzione avviene nel VoiceGenerator (Stage D) e non
richiede modifiche al SceneDirector o al prompt Gemini — il copione DIAS
resta identico.

Se Qwen3 non dovesse superare il test comparativo, il sistema resta su Fish
senza alcuna modifica: i due backend sono completamente indipendenti.

*Qwen3-TTS 1.7B Backend — ARIA v1.2 — Marzo 2026*  
*Documento di specifiche per sviluppo e test comparativo*
