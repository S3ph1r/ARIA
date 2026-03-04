# LLM Backend — Llama 3.1 8B per ARIA
## Backend Testuale: Filosofia, Dettagli Tecnici e Setup

> **File**: `docs/llm-backend.md`
> **Contesto**: Questo documento descrive il backend LLM di ARIA.
> Stessa filosofia di `docs/fish-tts-backend.md`: un server nativo Windows
> nel suo ambiente conda isolato, che ARIA chiama via HTTP.

---

## 1. PERCHÉ UN LLM LOCALE

### Il contesto attuale

DIAS dipende attualmente da Gemini (Google API) per le fasi cognitive della pipeline:
- **Stage C (Scene Director)**: analisi del testo, assegnazione emozioni ai dialoghi
- **Stage D (Text Director)**: annotazione con emotion markers Fish

Questa dipendenza è funzionale ma introduce:
- Costo per token (API a consumo)
- Latenza di rete
- Dipendenza da API esterna (rate limits, downtime)
- Privacy: il contenuto del libro passa per server Google

### Perché Llama 3.1 8B risolve il problema

Llama 3.1 8B Instruct è il modello open source che meglio bilanciamento qualità/VRAM
per compiti di editing testuale strutturato in italiano:
- **Italiano**: supporto nativo, output coerente senza fine-tuning
- **Instruction following**: addestrato con RLHF per seguire istruzioni strutturate
- **Contesto**: 128k token di contesto — sufficiente per capitoli interi
- **VRAM**: ~4.5-5.0 GB in Q4 bitsandbytes → compatibile con la politica "un modello alla volta"

### Confronto con le alternative

| Modello | VRAM Q4 | Italiano | Instruction | Note |
|---------|---------|----------|-------------|------|
| Llama 3.1 8B Instruct | ~5 GB | ✅ Nativo | ✅ Eccellente | **Scelta principale** |
| Qwen 2.5 7B Instruct | ~4.5 GB | ✅ Buono | ✅ Buono | Alternativa valida |
| Mistral 7B Instruct v0.3 | ~4.5 GB | ⚠️ Discreto | ✅ Buono | Meno capace in italiano |
| Gemma 2 9B Instruct | ~6 GB | ✅ Buono | ✅ Buono | VRAM leggermente superiore |

---

## 2. ARCHITETTURA TECNICA

### Pattern: External HTTP Backend

Il backend LLM segue lo stesso pattern di Fish S1-mini:
un processo Python standalone gira nativo Windows nel suo env conda,
espone un'API HTTP, e ARIA lo chiama come External Backend.

```
ARIA Node Controller (orchestratore Windows)
    │
    │  HTTP POST http://localhost:8085/v1/chat/completions
    │  (formato compatibile OpenAI)
    ▼
┌──────────────────────────────────────────────┐
│  LLM Server (conda env: llm-backend)         │
│  python llm_server.py --port 8085           │
│  transformers + bitsandbytes Q4              │
│  Accesso diretto GPU RTX 5060 Ti            │
└──────────────────────────────────────────────┘
    │
    ▼
Llama 3.1 8B Instruct (Q4 bitsandbytes)
~4.5-5.0 GB VRAM
```

### Perché Python/transformers invece di llama.cpp

- `llama.cpp` (llama-server) su Blackwell `sm_120` richiede compilazione da sorgente
  con flag speciali (documentata in `docs/Dockerfile.llama-blackwell.md`)
- `torch==2.7.0+cu128` è già testato e funzionante su questo PC Gaming per Fish
- `bitsandbytes >= 0.44.0` supporta Windows nativo quando PyTorch ha sm_120
- Manutenzione: `pip install --upgrade` vs gestione build C++

---

## 3. SETUP AMBIENTE — Windows Nativo

### Prerequisiti

- Miniconda installato (vedi `docs/environments-setup.md`)
- HuggingFace token con accesso accettato a `meta-llama/Meta-Llama-3.1-8B-Instruct`
  ([Richiedi accesso qui](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct))

### Creazione ambiente conda

```cmd
:: 1. Crea ambiente (Python 3.11) — project-local in %ARIA_ROOT%/envs/
conda create --prefix %ARIA_ROOT%\envs\llm python=3.11 -y

:: Installa le dipendenze usando il python dell'ambiente
%ARIA_ROOT%\envs\llm\python.exe -m pip install ^
    --index-url https://download.pytorch.org/whl/cu128

:: 3. Stack HuggingFace
pip install transformers>=4.47.0
pip install accelerate>=0.34.0
pip install bitsandbytes>=0.44.0

:: 4. Server API
pip install fastapi uvicorn[standard] pydantic>=2.0

:: 5. Utility
pip install huggingface-hub

:: 6. Verifica CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, sm: {torch.cuda.get_device_capability()}')"
:: Output atteso: CUDA: True, sm: (12, 0)

:: 7. Verifica bitsandbytes
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

### Download modello

```cmd
:: Login HuggingFace (una tantum)
huggingface-cli login

:: Download Llama 3.1 8B Instruct (~16GB su disco, safetensors)
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct ^
    --local-dir C:\models\llama-3.1-8b-instruct ^
    --exclude "original/*"
:: L'opzione --exclude "original/*" salta i pesi nel formato originale Meta
:: e scarica solo i safetensors (più veloci da caricare)
```

> ⚠️ **Disco**: ~16 GB non compressi. Conta circa 20 minuti su connessione a 100 Mbps.

---

## 4. LLM SERVER (`llm_server.py`)

Da posizionare in `%ARIA_ROOT%\aria_node_controller\llm_server.py`:

```python
"""
ARIA LLM Server — Llama 3.1 8B Instruct (bitsandbytes Q4)
Porta: 8085
Compatibile OpenAI: POST /v1/chat/completions
"""
import os
import time
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL_PATH = os.getenv("LLM_MODEL_PATH", r"C:\models\llama-3.1-8b-instruct")
PORT = int(os.getenv("LLM_SERVER_PORT", "8085"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="ARIA LLM Server — Llama 3.1 8B", version="1.0.0")

# ─── Quantizzazione 4-bit ─────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ─── Modello globale (caricato a startup) ─────────────────────────────────────
_tokenizer = None
_model = None
_load_time = None

@app.on_event("startup")
async def load_model():
    global _tokenizer, _model, _load_time
    t0 = time.time()
    print(f"[LLM Server] Caricamento modello da {MODEL_PATH}...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    _load_time = time.time() - t0
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"[LLM Server] ✅ Pronto in {_load_time:.1f}s — VRAM: {vram_gb:.1f} GB")

# ─── Modelli Pydantic ─────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    return {
        "status": "ok",
        "model": os.path.basename(MODEL_PATH),
        "cuda": torch.cuda.is_available(),
        "vram_used_gb": round(vram_gb, 2),
        "load_time_s": round(_load_time, 1) if _load_time else None,
    }

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    t0 = time.time()

    # Prepara il prompt con il template Llama 3.1
    chat = _tokenizer.apply_chat_template(
        [m.model_dump() for m in req.messages],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _tokenizer(chat, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.temperature > 0,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Estrae solo i token generati (non il prompt)
    response_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = _tokenizer.decode(response_ids, skip_special_tokens=True)
    duration = time.time() - t0

    return {
        "choices": [{"message": {"role": "assistant", "content": response_text}}],
        "model": "llama-3.1-8b-instruct",
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": len(response_ids),
        },
        "duration_seconds": round(duration, 2),
    }

if __name__ == "__main__":
    print(f"[LLM Server] Avvio su 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
```

---

## 5. INTEGRAZIONE CON ARIA

### Configurazione `config.yaml`

```yaml
models:
  llm:
    llama-3.1-8b:
      enabled: true
      api_url: "http://localhost:8085"
      request_timeout_seconds: 120
      estimated_vram_gb: 5.0
      max_retries: 1
```

### Backend `aria_server/backends/llm_backend.py`

```python
import time
import requests
import os
from .base import BaseBackend
from ..models import AriaTaskPayload, AriaTaskResult

LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8085")

class LLMBackend(BaseBackend):
    """
    Backend LLM per Llama 3.1 8B.
    Chiama il llm_server.py su Windows nativo via HTTP.
    load()   = health check sull'API server
    unload() = no-op (il processo è gestito da Windows)
    run()    = POST /v1/chat/completions → risposta testuale
    """

    model_id = "llama-3.1-8b"
    model_type = "llm"

    def __init__(self, config: dict):
        self.api_url = config.get("api_url", LLM_API_URL)
        self.timeout = config.get("request_timeout_seconds", 120)
        self._loaded = False

    def load(self) -> None:
        """Health check sul llm_server. Fallisce se non risponde."""
        try:
            r = requests.get(f"{self.api_url}/health", timeout=10)
            r.raise_for_status()
            self._loaded = True
            data = r.json()
            print(f"[llm_backend] Server LLM pronto — VRAM: {data.get('vram_used_gb')} GB")
        except Exception as e:
            raise RuntimeError(f"LLM server non raggiungibile: {e}")

    def unload(self) -> None:
        """No-op: il processo LLM gira su Windows, non lo gestiamo noi."""
        self._loaded = False

    def estimated_vram_gb(self) -> float:
        return 5.0

    def is_loaded(self) -> bool:
        return self._loaded

    def run(self, task: AriaTaskPayload) -> AriaTaskResult:
        t0 = time.time()
        payload = task.payload

        # Supporta sia formato "messages" diretto che "intent" risolto
        messages = payload.get("messages", [])
        if not messages:
            raise ValueError("Payload LLM mancante di 'messages'")

        body = {
            "messages": messages,
            "max_tokens": payload.get("max_tokens", 1000),
            "temperature": payload.get("temperature", 0.7),
        }

        r = requests.post(
            f"{self.api_url}/v1/chat/completions",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        result_data = r.json()

        response_text = result_data["choices"][0]["message"]["content"]
        duration = time.time() - t0

        return AriaTaskResult(
            job_id=task.job_id,
            client_id=task.client_id,
            model_type=self.model_type,
            model_id=self.model_id,
            status="done",
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            processing_time_seconds=round(duration, 2),
            output={
                "text": response_text,
                "tokens_generated": result_data.get("usage", {}).get("completion_tokens"),
            }
        )
```

### Schema Redis per task LLM

```
INPUT:   gpu:queue:llm:llama-3.1-8b
OUTPUT:  gpu:result:{client_id}:{job_id}
```

### Payload task LLM da DIAS

```json
{
  "job_id": "uuid-v4",
  "client_id": "dias-minipc",
  "model_type": "llm",
  "model_id": "llama-3.1-8b",
  "queued_at": "2026-03-03T10:00:00Z",
  "priority": 1,
  "timeout_seconds": 300,
  "callback_key": "gpu:result:dias-minipc:uuid-v4",
  "payload": {
    "messages": [
      {
        "role": "system",
        "content": "Sei un direttore creativo per audiolibri italiani..."
      },
      {
        "role": "user",
        "content": "Testo della scena da analizzare..."
      }
    ],
    "max_tokens": 1000,
    "temperature": 0.2
  }
}
```

---

## 6. AVVIO AUTOMATICO (Aggiornamento .bat)

Aggiungere a `Avvia_Tutti_Server_ARIA.bat`:

```bat
echo Avvio del Server LLM Llama 3.1 8B (Porta 8085) [VRAM ~5GB]...
start "ARIA LLM SERVER" cmd /k "cd /d %ARIA_ROOT%\envs\llm-backend & ^
    echo ===== LLM SERVER 8085 ===== & ^
    %MINICONDA_ROOT%\envs\llm-backend\python.exe llm_server.py"
timeout /t 90 /nobreak >nul
:: Attesa 90s — il modello impiega ~60s per caricarsi la prima volta
```

---

## 7. NOTE SU VRAM E COESISTENZA

Con La politica ARIA "un modello alla volta":

| Scenario | VRAM occupata | VRAM libera |
|---|---|---|
| LLM Llama 3.1 8B attivo | ~5 GB | ~11 GB |
| Fish S1-mini attivo | ~3-4 GB | ~12-13 GB |
| Nessun modello attivo | ~0.5 GB (CUDA context) | ~15.5 GB |

Il `BatchOptimizer` garantisce che non vengano caricati contemporaneamente.
Se in futuro i task LLM e TTS si alternano frequentemente, il BatchOptimizer
deve essere configurato con `batch_wait_seconds` adeguato per minimizzare
i cicli di caricamento/scaricamento.

---

## 8. ROADMAP LLM

Vedi `docs/backends-roadmap.md` per la checklist operativa.

| Fase | Contenuto | Stato |
|---|---|---|
| BK-2.1 | Setup conda `llm-backend` + verifica CUDA | 🔲 Da fare |
| BK-2.2 | `llm_server.py` + test manuale | 🔲 Da fare |
| BK-2.3 | Test qualità output italiano per DIAS | 🔲 Da fare |
| BK-2.4 | `llm_backend.py` in ARIA + Redis integration | 🔲 Da fare |
| BK-2.5 | Avvio automatico nel `.bat` | 🔲 Da fare |

---

*ARIA LLM Backend — Marzo 2026*
*Documenti correlati: `docs/backends-roadmap.md`, `docs/fish-tts-backend.md`, `docs/ARIA-blueprint.md`*
