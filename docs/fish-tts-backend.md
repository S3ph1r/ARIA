# Fish Audio S1-mini — Backend TTS per ARIA

> **Aggiornato**: 2026-03-04
> **Ambiente**: `%ARIA_ROOT%\envs\fish-speech-env` (Python 3.10)
> **Porte**: 8080 (TTS), 8081 (Voice Cloning)
> **Stato**: 🔄 Ambiente da ricreare
> **Spec storica completa**: `docs/archive/v1-pre-migration/fish-tts-backend.md`

---

## 1. Panoramica

Fish Audio S1-mini è il backend TTS espressivo di ARIA. Sostituisce Orpheus 3B con:
- **Qualità**: Dual-AR + RLHF, #1 su TTS-Arena-V2, WER italiano <1%
- **Emotion markers**: 50+ tag (`(scared)`, `(sighing)`, `(whispering)`, ecc.)
- **Infrastruttura**: PyTorch nativo Windows, niente llama.cpp/GGUF
- **Italiano**: supporto nativo con prosodia naturale

### Quando usare Fish vs Qwen3

| Criterio | Fish S1-mini | Qwen3-TTS |
|----------|-------------|-----------|
| **Uso ideale** | Voci espressive, dialoghi emotivi | Narrazione calda, stile audiolibro |
| **Controllo emozione** | Emotion markers espliciti `(scared)` | Istruzioni in linguaggio naturale |
| **Concurrency** | TTS + Voice Cloning separati | Server unico |
| **Coda Redis** | `gpu:queue:tts:fish-s1-mini` | `gpu:queue:tts:qwen3-tts-1.7b` |

---

## 2. Architettura Tecnica

### Dual-AR (Semantic + Acoustic)

```
INPUT: testo + ref audio
         │
         ▼
┌─────────────────────────────────┐
│  AR-1: SEMANTIC                 │  ← comprende significato e prosodia
│  "cosa dire e come dirlo"       │
└────────────────┬────────────────┘
                 │ semantic tokens
                 ▼
┌─────────────────────────────────┐
│  AR-2: ACOUSTIC                 │  ← genera timbro e dettagli audio
│  "come suona fisicamente"       │
└────────────────┬────────────────┘
                 │ acoustic tokens
                 ▼
┌─────────────────────────────────┐
│  DECODER (DAC codec)            │  ← decodifica token → PCM audio
└─────────────────────────────────┘
                 │
                 ▼
             WAV OUTPUT
```

### Specifiche Tecniche

| Proprietà | Valore |
|-----------|--------|
| Modello | `fishaudio/openaudio-s1-mini` |
| VRAM | ~3-4 GB (GPU) |
| Voice Cloning | Zero-shot ICL, 10-30s ref audio |
| Chunking | Automatico, max 8192 token (~250 parole) |
| RTF su RTX 5060 Ti | ~1:5 (5x realtime) |
| Sample rate output | 44.1 kHz mono |

---

## 3. Setup Ambiente

> Setup dettagliato con variabili: `docs/environments-setup.md`

### Creazione ambiente

```cmd
:: Creare ambiente project-local
conda create --prefix %ARIA_ROOT%\envs\fish-speech-env python=3.10 -y

:: PyTorch 2.7+cu128 (OBBLIGATORIO per sm_120 Blackwell)
%ARIA_ROOT%\envs\fish-speech-env\python.exe -m pip install ^
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 ^
    --index-url https://download.pytorch.org/whl/cu128

:: Fish-Speech (dal repo clonato)
%ARIA_ROOT%\envs\fish-speech-env\python.exe -m pip install ^
    -e %ARIA_ROOT%\envs\fish-speech

:: Dipendenze aggiuntive
%ARIA_ROOT%\envs\fish-speech-env\python.exe -m pip install torchcodec
```

### Download modello

```cmd
huggingface-cli download fishaudio/openaudio-s1-mini ^
    --local-dir %ARIA_ROOT%\data\models\fish-s1-mini
```

---

## 4. Porte e Servizi

```
                   ┌─────────────────────────────────────┐
                   │  %ARIA_ROOT%\envs\fish-speech-env    │
                   │                                     │
Fish TTS (:8080)   │  python -m tools.api_server         │
                   │  --listen 0.0.0.0:8080              │
                   │  --llama-checkpoint-path <model>    │
                   │  --decoder-checkpoint-path <model>  │
                   │  --compile                          │
                   │                                     │
Voice Cloning      │  python voice_cloning_server.py     │
(:8081)            │  DEVICE=cuda (se PyTorch sm_120 ok) │
                   └─────────────────────────────────────┘
```

> ⚠️ **sm_120 Warning**: Con PyTorch < 2.7, il VQGAN (Voice Cloning) crasha
> con `CUDA error: no kernel image`. La soluzione è PyTorch 2.7+cu128.

---

## 5. Emotion Markers

Fish S1-mini supporta 50+ tag emotivi posizionati **prima** della parola target:

```
(scared)La porta era socchiusa... (whispering)qualcuno ci osservava.
(serious)"Non possiamo restare qui," (nervous)disse guardandosi alle spalle.
```

### Tag principali per audiolibri italiani

| Tag | Uso | Effetto |
|-----|-----|---------|
| `(scared)` | Paura, terrore | Voce tremante, accelerata |
| `(whispering)` | Sussurri | Volume basso, intimità |
| `(serious)` | Gravità | Tono profondo, misurato |
| `(nervous)` | Ansia | Leggero tremito, esitazione |
| `(sighing)` | Rassegnazione | Respiro udibile |
| `(excited)` | Entusiasmo | Energia, velocità |
| `(sad)` | Tristezza | Tono basso, lento |
| `(laughing)` | Risata | Effetto diretto |
| `(panicked)` | Panico | Voce alta, concitata |
| `(hesitating)` | Esitazione | Pause, incertezza |

### Parametri API consigliati per audiolibri

```python
payload = {
    "text": "(nervous)La porta era socchiusa...",
    "references": [{"audio": ref_bytes, "text": ref_text}],
    "normalize": False,       # NON strippare i tag emotivi
    "mp3_bitrate": 0,         # Output WAV, non MP3
    "format": "wav",
    "streaming": False,
    "max_new_tokens": 2048,
    "top_p": 0.8,
    "temperature": 0.7,
    "repetition_penalty": 1.2
}
```

---

## 6. Voice Library

> Documentazione completa: `docs/hybrid-tts-architecture.md`

Fish usa la Voice Library condivisa con Qwen3:

```
%ARIA_ROOT%\data\voices\
├── angelo/                 ← Disponibile (ref.wav present)
├── luca/
│   ├── ref.wav             ← Usato da Fish TTS
│   ├── ref_padded.wav      ← Usato da Qwen3 TTS
│   └── ref.txt             ← Trascrizione (ICL ad alta fedeltà)
```

Quando il backend riceve `voice_id: "luca"`:
1. Carica `voices/luca/ref.wav` e `voices/luca/ref.txt`
2. Li converte nel formato payload HTTP di Fish
3. Invia la richiesta al server Fish su `:8080`

---

## 7. Integrazione ARIA

Fish è un **External HTTP Backend** avviato on-demand:

```python
# In orchestrator.py → _build_cmd()
if model_id == "fish-s1-mini":
    python = str(self.aria_root / "envs" / "fish-speech-env" / "python.exe")
    fish_dir = self.aria_root / "envs" / "fish-speech"
    return [python, "-m", "tools.api_server", "--listen", "0.0.0.0:8080", ...]
```

### Schema Redis

```
INPUT:   gpu:queue:tts:fish-s1-mini
OUTPUT:  gpu:result:{client_id}:{job_id}
```

### Chunking automatico

Per testi > 250 parole, l'Orchestratore splitta su confini di frase e
concatena i WAV con 80ms di silenzio tra i chunk.

---

## 8. Workaround Noti

### Glitch primo token
Fish S1-mini ha un artefatto sonoro sul primo token generato.
**Soluzione**: l'Orchestratore inietta `(break)` all'inizio di ogni chunk.
Il glitch "mangia" il silenzio del break, salvando le parole reali.

### VQGAN su CPU
Se PyTorch non supporta sm_120, il Voice Cloning (VQGAN) va forzato su CPU:
impostare `DEVICE="cpu"` in `voice_cloning_server.py`. Più lento ma funzionante.

---

*Documenti correlati:*
- *`docs/environments-setup.md` — guida ambienti Python*
- *`docs/hybrid-tts-architecture.md` — voice routing e ICL*
- *`docs/ARIA-blueprint.md` — architettura completa*
- *`docs/archive/v1-pre-migration/fish-tts-backend.md` — spec storica originale*
