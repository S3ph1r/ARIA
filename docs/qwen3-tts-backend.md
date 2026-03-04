# Qwen3-TTS 1.7B — Backend TTS per ARIA

> **Aggiornato**: 2026-03-04
> **Ambiente**: `%ARIA_ROOT%\envs\qwen3tts` (Python 3.12)
> **Porta**: 8083
> **Stato**: ✅ Funzionante
> **Spec storica completa**: `docs/archive/v1-pre-migration/qwen3-tts-backend.md`

---

## 1. Panoramica

Qwen3-TTS è il backend TTS "da audiolibro" di ARIA. Affianca Fish S1-mini con:
- **Voce calda e stabile**: ideale per narrazione lunga e continua
- **Controllo via linguaggio naturale**: istruzioni come "voce calda maschile, ritmo lento" invece di tag espliciti
- **Auto-Padding**: risolve automaticamente il problema del "bleeding fonetico"
- **12 Hz token rate**: alta qualità su testi lunghi senza allucinazioni

### Quando usare Qwen3 vs Fish

| Criterio | Qwen3-TTS | Fish S1-mini |
|----------|-----------|-------------|
| **Uso ideale** | Narrazione calda, audiolibri | Dialoghi emotivi, voci espressive |
| **Controllo stile** | Istruzioni naturali (instruct) | Emotion markers `(scared)` |
| **Qualità su testi lunghi** | ✅ Eccellente, stabile | ⚠️ Richiede chunking aggressivo |
| **Coda Redis** | `gpu:queue:tts:qwen3-tts-1.7b` | `gpu:queue:tts:fish-s1-mini` |

---

## 2. Architettura Tecnica

### LLM Transformer + DAC Codec

```
INPUT: testo + istruzione stile + ref audio
         │
         ▼
┌─────────────────────────────────┐
│  LLM TRANSFORMER (1.7B param)  │  ← interpreta testo + istruzione
│  "cosa dire e come dirlo"      │    genera speech codes a 12 token/s
└────────────────┬────────────────┘
                 │ speech tokens (12Hz)
                 ▼
┌─────────────────────────────────┐
│  DAC CODEC (decodifica)         │  ← trasforma token in forma d'onda
│  "audio finale"                 │
└─────────────────────────────────┘
                 │
                 ▼
             WAV OUTPUT (24 kHz)
```

### Variante modello

Per ARIA/DIAS usare **esclusivamente** `Qwen3-TTS-12Hz-1.7B-Base`:

| Variante | Param | VRAM | Uso |
|----------|-------|------|-----|
| `12Hz-1.7B-Base` | 1.7B | ~4-5 GB | ✅ Voice cloning (per DIAS) |
| `12Hz-0.6B-Base` | 0.6B | ~4 GB | ❌ Bug long-silence |
| `12Hz-1.7B-CustomVoice` | 1.7B | ~6 GB | ❌ Solo speaker predefiniti |

---

## 3. Setup Ambiente

> Setup dettagliato con variabili: `docs/environments-setup.md`

### Creazione ambiente

```cmd
:: Creare ambiente project-local (Python 3.12 — diverso da Fish che usa 3.10)
conda create --prefix %ARIA_ROOT%\envs\qwen3tts python=3.12 -y

:: PyTorch 2.6+cu124
%ARIA_ROOT%\envs\qwen3tts\python.exe -m pip install ^
    torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

:: Qwen-TTS + dipendenze
%ARIA_ROOT%\envs\qwen3tts\python.exe -m pip install ^
    qwen-tts fastapi uvicorn soundfile numpy ^
    transformers>=4.52.0 accelerate>=1.7.0 huggingface_hub

:: Flash Attention 2 (opzionale, riduce VRAM e latenza)
:: Scaricare wheel precompilato da https://github.com/Dao-AILab/flash-attention/releases
:: per Python 3.12 + cu128
```

### Download modello

```cmd
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base ^
    --local-dir %ARIA_ROOT%\data\models\qwen3-tts-1.7b
```

---

## 4. Server FastAPI (`qwen3_server.py`)

Il server Qwen3 è integrato nel codice ARIA:
`%ARIA_ROOT%\aria_node_controller\qwen3_server.py`

### Endpoints

| Endpoint | Metodo | Descrizione |
|----------|--------|-------------|
| `/health` | GET | Health check + info VRAM |
| `/tts` | POST | Sintesi vocale con voice cloning |

### Parametri per `/tts`

```json
{
  "text": "Il sole sorgeva lentamente all'orizzonte.",
  "voice_id": "luca",
  "instruct": "Warm Italian male voice, professional audiobook narrator, calm and measured pace."
}
```

Il server risolve automaticamente il `voice_id`:
1. Cerca `%ARIA_ROOT%\data\voices\luca\ref_padded.wav`
2. Se `ref_padded.wav` non esiste, lo genera da `ref.wav` (adds 0.5s silenzio)
3. Carica `ref.txt` se presente (ICL ad alta fedeltà)

---

## 5. Auto-Padding: il Problema del "Bleeding Fonetico"

**Problema**: Qwen3 è autoregressivo. Se il sample audio finisce bruscamente,
l'ultimo fonema si "propaga" ai token successivi, rovinando l'audio.

**Soluzione (automatica)**: il server cerca `ref_padded.wav`. Se non esiste,
invoca `ffmpeg` per aggiungere 0.5s di silenzio e lo salva per usi futuri.

```
ref.wav  →  [audio clips bruscamente]     → artefatti nel generato
ref_padded.wav → [audio + 0.5s silenzio]  → generato pulito ✅
```

---

## 6. Instruct: Controllo Stile via Linguaggio Naturale

A differenza di Fish (emotion markers espliciti), Qwen3 usa **istruzioni**
in linguaggio naturale per controllare tono, ritmo e atmosfera:

### Mappa emozioni DIAS → instruct Qwen3

| primary_emotion (DIAS) | Instruct Qwen3 |
|---|---|
| `neutral` | `"Warm male voice, Italian audiobook narrator, calm and measured, moderate pace."` |
| `suspense` | `"...tense and restrained, slightly slower pace, hushed intensity."` |
| `fear` | `"...anxious and cautious, slow deliberate pace, quiet."` |
| `sadness` | `"...melancholic and reflective, slow pace, soft voice."` |
| `anger` | `"...forceful and intense, faster pace, sharp articulation."` |
| `joy` | `"...warm and bright, moderate to fast pace, smile in the voice."` |

### Modificatori aggiuntivi

```python
# Per scene di suspense
"...dramatic pauses on ellipsis and em-dashes, hold breath before revelations."

# Per dialoghi
"...differentiate narrative voice from quoted speech, slight tonal shift."
```

---

## 7. Chunking Automatico

Per testi > 250 parole, il server splitta automaticamente:
1. Divide su confini di frase (`. ? !`)
2. Genera WAV separati per ogni chunk
3. Concatena con **80ms di silenzio** fisiologico tra i chunk
4. Restituisce un **unico file WAV** al chiamante

> Il client non deve preoccuparsi della lunghezza del testo.

---

## 8. Integrazione ARIA

Qwen3 è un **External HTTP Backend** avviato on-demand dall'Orchestratore:

```python
# In orchestrator.py → _build_cmd()
if model_id == "qwen3-tts-1.7b":
    python = str(self.aria_root / "envs" / "qwen3tts" / "python.exe")
    server = self.aria_root / "aria_node_controller" / "qwen3_server.py"
    return [python, str(server)]
```

### Schema Redis

```
INPUT:   gpu:queue:tts:qwen3-tts-1.7b
OUTPUT:  gpu:result:{client_id}:{job_id}
```

### Payload esempio

```json
{
  "job_id": "uuid-v4",
  "client_id": "dias-minipc",
  "model_type": "tts",
  "model_id": "qwen3-tts-1.7b",
  "payload": {
    "text": "Il sole sorgeva lentamente...",
    "voice_id": "angelo",
    "instruct": "Warm male voice, calm narrator."
  }
}
```

---

## 9. Voice Library

> Documentazione completa: `docs/hybrid-tts-architecture.md`

Qwen3 usa la Voice Library condivisa con Fish, con la differenza del padding:

```
%ARIA_ROOT%\data\voices\
├── luca/
│   ├── ref.wav             ← sample originale (NON usato da Qwen3)
│   ├── ref_padded.wav      ← sample + 0.5s silenzio (USATO da Qwen3)
│   └── ref.txt             ← trascrizione (ICL opzionale)
```

Per creare nuovi sample: `python scripts/voice_prepper.py "URL_YouTube" "nome_voce"`
Lo script genera automaticamente sia `ref.wav` che `ref_padded.wav`.

---

## 10. Problemi Noti

### Bleeding fonetico
Risolto con Auto-Padding (vedi sezione 5).

### Python 3.12 obbligatorio
Qwen3-TTS richiede Python 3.12+ per compatibilità con `qwen-tts` e flash-attention 2.
Non usare Python 3.10 (quello è per Fish).

### Primo caricamento lento
Il modello impiega ~30-60s per il primo caricamento in VRAM.
L'Orchestratore gestisce un `startup_wait` adeguato prima di inviare richieste.

---

*Documenti correlati:*
- *`docs/environments-setup.md` — guida ambienti Python*
- *`docs/hybrid-tts-architecture.md` — voice routing, ICL, Auto-Padding*
- *`docs/ARIA-blueprint.md` — architettura completa*
- *`docs/archive/v1-pre-migration/qwen3-tts-backend.md` — spec storica originale (49KB)*
