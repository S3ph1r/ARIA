# ARIA — Roadmap Backend
## Sviluppo e Test dei Backend di Inferenza

> **File**: `docs/backends-roadmap.md`
> **Aggiornato**: 2026-03-03
> **Riferimento**: ARIA Blueprint v2.0, `docs/fish-tts-backend.md`

Questo documento traccia i prossimi passi operativi per la componente **backend** di ARIA:
ambienti conda, server di inferenza, integrazione nell'orchestratore.

---

## Stato Attuale degli Ambienti (PC Gaming Windows — 192.168.1.139)

```
C:\Users\gemini\miniconda3\envs\
├── fish-voice-cloning\   # VQGAN Voice Cloning Server (porta 8081)
├── fish-speech\          # Fish S1-mini TTS Server (porta 8080)
└── (base)                # ARIA Node Controller (main_tray.py)
```

Lo script di avvio unificato è `Avvia_Tutti_Server_ARIA.bat`.

---

## FASE BK-1 — Verifica e Possibile Unificazione Ambienti Fish

> **Priorità**: Alta — semplifica la manutenzione prima di aggiungere nuovi env
> **Stima**: 1 giorno
> **Blocco**: Richiede accesso al PC Gaming per eseguire i comandi

### Contesto

I due ambienti `fish-speech` e `fish-voice-cloning` nascono separati durante
il debugging del crash VQGAN su `sm_120`. Entrambi sono stati allineati alla
stessa soluzione: `torch==2.7.0+cu128`. Se i pacchetti installati risultano
identici, è possibile unirli in un unico ambiente senza alcuna perdita funzionale.

### BK-1.1 — Analisi Differenze tra i Due Ambienti

Eseguire sul PC Gaming (cmd con conda attivato):

```cmd
conda run -n fish-speech pip list > C:\temp\pkgs-fish-speech.txt
conda run -n fish-voice-cloning pip list > C:\temp\pkgs-fish-voice-cloning.txt

:: Confronto — cerca differenze significative
fc C:\temp\pkgs-fish-speech.txt C:\temp\pkgs-fish-voice-cloning.txt
```

**Criteri di decisione**:

| Risultato diff | Decisione |
|---|---|
| Nessuna differenza o differenze solo in pacchetti di debug/test | ✅ Unificare — usa `fish-speech` come env unico |
| Versioni PyTorch diverse o dipendenze fish-speech incompatibili | ❌ Tenere separati — documentare il motivo |
| Differenze in pacchetti minori (es. versioni patch) | ⚠️ Allineare i pacchetti, poi unificare |

- [ ] Eseguire i comandi di confronto
- [ ] Documentare il risultato del diff qui sotto

**Risultato diff** _(da compilare dopo l'analisi)_:
```
[Incollare qui l'output del diff o annotare le differenze trovate]
```

### BK-1.2 — Se Unificazione Approvata: Aggiornare il .bat

Se il diff è favorevole, aggiornare `Avvia_Tutti_Server_ARIA.bat`:

```bat
:: PRIMA (2 env separati):
%MINICONDA%\envs\fish-voice-cloning\python.exe voice_cloning_server.py
%MINICONDA%\envs\fish-speech\python.exe tools\api_server.py

:: DOPO (1 env unificato):
%MINICONDA%\envs\fish-speech\python.exe voice_cloning_server.py
%MINICONDA%\envs\fish-speech\python.exe tools\api_server.py
```

Test post-unificazione:
- [ ] Voice Cloning Server risponde su `http://localhost:8081/health`
- [ ] Fish TTS Server risponde su `http://localhost:8080/health`
- [ ] Test end-to-end: task voice cloning → TTS → WAV generato correttamente
- [ ] Eliminare l'ambiente `fish-voice-cloning` se tutto funziona

### BK-1.3 — Se Unificazione Non Approvata: Documentare

- [ ] Annotare le dipendenze incompatibili trovate
- [ ] Aggiornare `docs/ARIA-blueprint.md` sezione "Ambienti Python" con la spiegazione

---

## FASE BK-2 — Backend LLM: Llama 3.1 8B

> **Priorità**: Media — prerequisito per autonomizzare DIAS dalla dipendenza Gemini
> **Stima**: 3-5 giorni
> **Dipendenza**: BK-1 completata (sappiamo quanti env gestire)

### Architettura Scelta

**Pattern: External HTTP Backend** — identico a Fish S1-mini.

Un server FastAPI dedicato gira nativo Windows nel suo conda env (`llm-backend`),
espone un endpoint `POST /v1/chat/completions` compatibile OpenAI.
Il broker ARIA lo chiama via HTTP, esattamente come chiama Fish su `:8080`.

```
ARIA Node Controller (orchestratore)
    │
    │ HTTP POST http://localhost:8085/v1/chat/completions
    ▼
LLM Server (conda env llm-backend, porta 8085)
    │ torch 2.7+cu128 + transformers + bitsandbytes
    ▼
Llama 3.1 8B (Q4 bitsandbytes) — ~4.5-5.0 GB VRAM
```

**Perché non `llama-server` (llama.cpp)**:
- Su Blackwell sm_120, llama.cpp richiede compilazione da sorgente con flag speciali
  (documentata in `docs/Dockerfile.llama-blackwell.md`) — complessità non necessaria
- PyTorch 2.7+cu128 è già testato e funzionante sul PC Gaming
- `bitsandbytes >= 0.44` supporta Windows nativo con PyTorch ≥ 2.7
- Con transformers + bitsandbytes si ottiene lo stesso footprint VRAM di GGUF Q4

### BK-2.1 — Setup Ambiente Conda `llm-backend`

```cmd
:: 1. Crea ambiente (Python 3.11 — meglio di 3.10 per HuggingFace recenti)
conda create -n llm-backend python=3.11 -y
conda activate llm-backend

:: 2. PyTorch — STESSA versione testata per Fish (fondamentale per sm_120)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 ^
    --index-url https://download.pytorch.org/whl/cu128

:: 3. HuggingFace stack
pip install transformers>=4.47.0
pip install accelerate>=0.34.0
pip install bitsandbytes>=0.44.0

:: 4. Server API
pip install fastapi uvicorn[standard]
pip install huggingface-hub

:: 5. Verifica CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

```cmd
:: 6. Download modello Llama 3.1 8B (Instruct, formato safetensors)
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct ^
    --local-dir C:\models\llama-3.1-8b-instruct
```

> **Nota modello**: Richiede accettare la licenza Meta su HuggingFace e token HF.
> Alternativa senza licenza: `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`
> con bitsandbytes non necessario (ma torna il problema llama.cpp-sm_120).
> **Consiglio**: usare il modello ufficiale Meta con bitsandbytes Q4.

- [ ] Ambiente conda `llm-backend` creato
- [ ] CUDA verificato (output: `CUDA: True, Device: NVIDIA GeForce RTX 5060 Ti`)
- [ ] bitsandbytes funzionante: `python -c "import bitsandbytes; print(bitsandbytes.__version__)"`
- [ ] Modello scaricato in `C:\models\llama-3.1-8b-instruct\`

### BK-2.2 — LLM Server (`llm_server.py`)

Creare `C:\Users\Roberto\aria\envs\llm-backend\llm_server.py`:

```python
"""
LLM Inference Server per ARIA — Llama 3.1 8B (bitsandbytes Q4)
Porta: 8085
Endpoint: POST /v1/chat/completions (compatibile OpenAI)
"""
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_PATH = os.getenv("LLM_MODEL_PATH", r"C:\models\llama-3.1-8b-instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="ARIA LLM Server", version="1.0.0")

# Configurazione quantizzazione 4-bit (bitsandbytes)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Modello caricato all'avvio
tokenizer = None
model = None

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    print(f"[LLM Server] Caricamento modello da {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print(f"[LLM Server] Modello caricato. VRAM usata: "
          f"{torch.cuda.memory_allocated() / 1e9:.1f} GB")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "cuda": torch.cuda.is_available()}

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    chat = tokenizer.apply_chat_template(
        [m.model_dump() for m in req.messages],
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(chat, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            do_sample=req.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return {
        "choices": [{"message": {"role": "assistant", "content": response_text}}],
        "model": "llama-3.1-8b-instruct"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8085)
```

- [ ] `llm_server.py` creato nel path corretto
- [ ] Test avvio manuale: `python llm_server.py` → log "Modello caricato"
- [ ] Test health: `curl http://localhost:8085/health`
- [ ] Test inferenza: chiamata POST con prompt italiano → risposta coerente
- [ ] Misurare VRAM usata con modello caricato (obiettivo: < 6GB)

### BK-2.3 — Test Qualità LLM

Prima di integrare in ARIA, validare la qualità per i casi d'uso DIAS:

```python
# Test prompt tipico DIAS — scene director
test_messages = [
    {"role": "system", "content": "Sei un direttore creativo per audiolibri italiani."},
    {"role": "user", "content": "Analizza questa scena e assegna emozioni ai dialoghi: 'Il monaco aprì la porta lentamente. Dentro, silenzio.'"}
]
```

- [ ] Risposta in italiano coerente e utile
- [ ] Latenza prima risposta: < 30s (obiettivo)
- [ ] RTF stimato per contesti tipici DIAS (800-1200 token)

### BK-2.4 — Backend `llm_backend.py` in ARIA

Creare `aria_server/backends/llm_backend.py` seguendo il pattern `FishTTSBackend`:

```python
# Pattern — External HTTP Backend (identico a fish_tts.py)
class LLMBackend(BaseBackend):
    model_id = "llama-3.1-8b"
    model_type = "llm"

    def load(self) -> None:
        """Health check sul llm-server. Fallisce se non risponde."""

    def unload(self) -> None:
        """No-op: il processo LLM gira su Windows, non lo gestiamo noi."""

    def estimated_vram_gb(self) -> float:
        return 5.0  # Llama 3.1 8B Q4 su sm_120

    def run(self, task: AriaTaskPayload) -> AriaTaskResult:
        """POST /v1/chat/completions → risposta testuale."""
```

- [ ] `aria_server/backends/llm_backend.py` creato
- [ ] Aggiornare `config.yaml`: `llama-3.1-8b: enabled: true, api_url: http://localhost:8085`
- [ ] Registrare backend in `aria_node_controller/core/orchestrator.py`
- [ ] Test integrazione: task Redis `gpu:queue:llm:llama-3.1-8b` → risposta su `gpu:result:*`

### BK-2.5 — Avvio Automatico LLM Server

Aggiungere al `Avvia_Tutti_Server_ARIA.bat`:

```bat
echo Avvio del Server LLM (Porta 8085) [VRAM ~5GB]...
start "ARIA LLM SERVER" cmd /k "echo ===== LLM SERVER 8085 ===== & ^
    %MINICONDA_ROOT%\envs\llm-backend\python.exe ^
    %ARIA_ROOT%\envs\llm-backend\llm_server.py"
timeout /t 90 /nobreak >nul
:: 90s di attesa — il modello ci mette ~60s a caricarsi la prima volta
```

> ⚠️ **Nota avvio**: Llama 3.1 8B in Q4 richiede ~60s per il caricamento iniziale.
> Il sistema di semaforo ARIA garantisce che nessun task LLM venga eseguito
> finché l'health check su `:8085/health` non risponde.

- [ ] `.bat` aggiornato con il server LLM
- [ ] Test riavvio Windows → tutti e 3 i server attivi entro 3 minuti
- [ ] ARIA BatchOptimizer riconosce modello `llama-3.1-8b` dalla coda

---

## Matrice Ambienti Finale (Obiettivo Post BK-1 + BK-2)

| Env Conda | Python | Scopo | Porta | VRAM |
|---|---|---|---|---|
| `fish-speech` | 3.10 | TTS Fish S1-mini + Voice Cloning (se BK-1 unifica) | 8080 + 8081 | ~3-4 GB |
| `fish-voice-cloning` | 3.10 | VQGAN encode (solo se BK-1 NON unifica) | 8081 | CPU |
| `llm-backend` | 3.11 | Llama 3.1 8B inferenza | 8085 | ~5 GB |
| `(base)` | 3.x | ARIA Node Controller (orchestratore + tray) | — | — |

**Footprint VRAM massimo simultaneo** (un modello alla volta — regola ARIA):
- Fish TTS attivo: 3-4 GB → 12-13 GB liberi per gioco/altro
- LLM attivo: 5 GB → 11 GB liberi

---

## Log Decisioni

| Data | Decisione | Motivazione |
|---|---|---|
| 2026-03-03 | Scelto conda env su llama-server per LLM | PyTorch cu128 già testato; llama.cpp richiede build custom su sm_120 |
| 2026-03-03 | Mantenuto pattern External HTTP Backend | Coerenza con Fish — orchestratore chiama via HTTP, process isolation |
| 2026-03-03 | BK-1 (unificazione) prima di BK-2 | Ridurre env prima di aggiungerne uno nuovo |

---

*ARIA Backends Roadmap — Marzo 2026*
*Prossimo documento da consultare: `docs/fish-tts-backend.md`, `docs/ARIA-blueprint.md`*
