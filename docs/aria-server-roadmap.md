# 🖥️ ARIA Server — Roadmap Sviluppo
## GPU Inference Broker — PC Gaming (Windows)

> **Riferimento**: ARIA Blueprint v1.0
> **Ambiente**: Windows 11, RTX 5060 Ti 16GB, Docker Desktop + GPU passthrough
> **Obiettivo MVP**: ARIA riceve task TTS Orpheus da DIAS, genera audio, restituisce risultati

---

## 🎯 CONSIGLI PER PROSSIMI AGENT

### Pattern Redis Bridge (Fase AS-2)
**Struttura Task Obbligatoria**:
```json
{
  "job_id": "job-xxx",
  "client_id": "client-xxx",
  "model_type": "tts|music|llm", 
  "model_id": "orpheus-3b|musicgen-small|llama-3b",
  "queued_at": 1740250000,
  "timeout_seconds": 300,
  "callback_key": "gpu:result:dias-minipc:job-xxx",
  "payload": { /* dati specifici modello */ }
}
```

**Code GPU Monitorate**:
- `gpu:queue:tts:orpheus-3b`
- `gpu:queue:music:musicgen-small` 
- `gpu:queue:llm:llama-3b`

**Comandi Utili**:
```bash
# Test lettura task
docker exec -it aria-aria-server-1 python3 -c "from aria_server import QueueManager; qm = QueueManager(); print(qm.next_task())"

# Test scrittura risultato
docker exec -it aria-aria-server-1 python3 -c "from aria_server import ResultWriter; rw = ResultWriter(); rw.write_result('job-xxx', 'client-xxx', {'status': 'success'})"

# Installa dipendenze runtime
docker exec -it aria-aria-server-1 pip install -r requirements-light.txt
```

### Build Ottimizzate
**Problema comune**: Build lente per ogni modifica
**Soluzione**: Usare sempre `docker-compose.dev.yml` per sviluppo
**Comando**: `docker-compose -f docker-compose.dev.yml up`
**Beneficio**: Modifiche immediate, PyTorch cached

### Debug Redis Bridge
**Task non letti**: Verificare struttura task con campi richiesti
**Connection refused**: Controllare Redis host in config.py
**Module not found**: Installare runtime con `docker exec pip install`

---

## 📋 SOMMARIO PROGRESSI

### ✅ COMPLETATO
- ✅ **Fase AS-1**: Setup Docker + struttura progetto (COMPLETATA)
- ✅ Repository GitHub creato: https://github.com/NH-Mini/ARIA
- ✅ Docker container GPU funzionante su PC Gaming 192.168.1.139
- ✅ RTX 5060 Ti 16GB rilevata correttamente con CUDA 12.1
- ✅ FastAPI server operativo su porta 8000
- ✅ Health check endpoint: `/health` con GPU status
- ✅ TTS API placeholders: `/tts/voices` e `/tts/synthesize`
- ✅ Workflow Git: sviluppo LXC 190 → push GitHub → pull PC Gaming
- ✅ **Setup sviluppo rapido**: docker-compose.dev.yml con volume mapping e reload automatico
- ✅ **Build ottimizzata**: Multi-stage build per separare PyTorch (pesante) da codice (leggero)

### ✅ COMPLETATO
- ✅ **Fase AS-2**: Redis Bridge — connessione e comunicazione (COMPLETATA 22/02/2026)
  - ✅ QueueManager con BRPOP non-bloccante e reconnect automatico
  - ✅ ResultWriter con TTL e circuit breaker
  - ✅ Validazione task con schema strutturato
  - ✅ Pattern code GPU: `gpu:queue:{type}:{model}`
  - ✅ Test end-to-end con DIAS verificato
  - ✅ Build ottimizzata: 65s → 2s con volume mapping

### 🔄 IN CORSO
- [ ] **Fase AS-3**: Logging — console colorata + JSON su file

### 📅 TUTTE LE FASI
- [ ] **Fase AS-1**: Setup Docker + struttura progetto
- [ ] **Fase AS-2**: Redis Bridge — connessione e comunicazione
- [ ] **Fase AS-3**: Logging — console colorata + JSON su file
- [ ] **Fase AS-4**: VRAM Manager — load/unload modelli
- [ ] **Fase AS-5**: Batch Optimizer — priority-first + greedy
- [ ] **Fase AS-6**: Backend TTS Orpheus — primo backend, MVP
- [ ] **Fase AS-7**: Result Writer + Crash Recovery
- [ ] **Fase AS-8**: API HTTP — semaforo e status
- [ ] **Fase AS-9**: Dashboard Web — monitoring dettagliato
- [ ] **Fase AS-10**: Tray Icon — controllo semaforo rapido
- [ ] **Fase AS-11**: `aria download` — gestione modelli
- [ ] **Fase AS-12**: `aria update` — script aggiornamento
- [ ] **Fase AS-13**: Backend MusicGen — secondo backend
- [ ] **Fase AS-14**: Backend LLM — terzo backend
- [ ] **Fase AS-15**: Samba — configurazione cartella condivisa

---

## 🏗️ ARCHITETTURA SCELTA

```
PC GAMING (Windows 11)
├── Docker Desktop (WSL2 backend + nvidia-container-toolkit)
│   └── container: aria-server
│       ├── main.py                  # orchestratore principale
│       ├── queue_manager.py         # BRPOP da Redis minipc
│       ├── batch_optimizer.py       # priority-first → greedy
│       ├── vram_manager.py          # load/unload + OOM handling
│       ├── result_writer.py         # scrive risultati + circuit breaker
│       ├── heartbeat.py             # pubblica stato ogni 10s
│       ├── backends/
│       │   ├── base_backend.py      # interfaccia astratta
│       │   ├── tts_backend.py       # Orpheus TTS (MVP)
│       │   ├── music_backend.py     # MusicGen (fase 13)
│       │   └── llm_backend.py       # LLM testuali (fase 14)
│       └── api/
│           └── http_api.py          # FastAPI: semaforo + status + dashboard
│
├── aria-tray/                       # processo separato (fuori Docker)
│   └── tray_icon.py                 # pystray, chiama API HTTP locale
│
├── C:\models\                       # modelli scaricati con aria download
│   ├── orpheus-3b-q4/
│   └── musicgen-small/
│
├── Z:\  (← \\minipc\aria-shared)   # cartella Samba montata
│   ├── voices/
│   ├── input/
│   └── output/
│
└── scripts/
    ├── aria-download.bat            # scarica modelli via HF Hub
    └── aria-update.bat              # aggiorna ARIA Server

```

---

## 🚀 PATTERN BUILD OTTIMIZZATI

### 🚀 PATTERN BUILD OTTIMIZZATI — AGGIORNATI CON ESPERIENZA

### Problema Riscontrato & Soluzione
**Problema**: Ogni modifica richiedeva rebuild completo (65s) con reinstallazione PyTorch
**Soluzione**: Multi-stage + volume mapping → 2s per modifiche codice

### Sviluppo Rapido (Consigliato - TESTATO)
```bash
# Setup una tantum
git pull origin master
docker-compose -f docker-compose.dev.yml up

# Le modifiche al codice sono immediate (hot-reload)
# PyTorch cached nel container base
# Requirements leggeri installati runtime
# Tempo effettivo: 2 secondi vs 65+ secondi
```

### Installazione Runtime Dependencies (Senza Rebuild)
```bash
# Se manca un modulo (es: redis)
docker exec -it aria-aria-server-1 pip install redis==5.0.1

# Per requirements completi
docker exec -it aria-aria-server-1 pip install -r requirements-light.txt
```

### Produzione Stabile
```bash
# Solo quando cambiano requirements pesanti
docker-compose build --no-cache
docker-compose up

# Build multi-stage separa:
# - Stage 1: PyTorch CUDA (cached permanentemente)
# - Stage 2: Requirements leggeri (aggiornabili runtime)
# - Stage 3: Codice applicazione (hot-reload in dev)
```

### File di Configurazione Aggiornati
- `docker-compose.dev.yml` → Sviluppo con volume mapping + hot-reload
- `docker-compose.yml` → Produzione stabile
- `Dockerfile.dev` → Base leggera per sviluppo rapido
- `Dockerfile` → Produzione completa con tutte le dipendenze
- `requirements-heavy.txt` → PyTorch, CUDA (installate una volta)
- `requirements-light.txt` → FastAPI, Redis, etc (aggiornabili runtime)

**Pattern riutilizzabile**: Separare dipendenze pesanti da codice leggero + volume mapping per sviluppo rapido senza rebuild.

---

### Scelte tecniche chiave

| Aspetto | Scelta | Motivazione |
|---|---|---|
| Runtime | Docker + nvidia-container-toolkit | Isolamento, portabilità |
| Modelli | `aria download` → `C:\models\` | Controllo esplicito, nessun download a runtime |
| File binari | Samba share montato come `Z:\` | Nativo Windows, zero software aggiuntivo |
| Test | Mock unit + GPU integration | Sviluppo offline sempre possibile |
| Scheduling | Priority-first (3→2→1) poi greedy | Flessibile, fairness con priorità |
| UI | Tray icon + dashboard web | Semaforo rapido + monitoring dettagliato |
| Crash recovery | Auto + circuit breaker | Robusto su crash innocui, sicuro su task patologici |
| Logging | Console colorata + JSON file | Debug comodo + analisi produzione |
| MVP | TTS Orpheus | End-to-end con DIAS il prima possibile |
| Update | `aria update` script | Un comando, controllo esplicito |
| Build sviluppo | Volume mapping + reload automatico | Modifiche immediate senza rebuild (65s → 2s) |
| Build produzione | Multi-stage + requirements separati | PyTorch cached, solo codice ricostruito |
| Runtime install | `docker exec pip install` | Aggiunta moduli senza rebuild (es: redis) |

---

## 📊 DETTAGLIO FASI

---

### 🔧 FASE AS-1: Setup Docker + Struttura Progetto

**🎯 Obiettivo**: Ambiente Docker funzionante con GPU passthrough, struttura progetto definita, verifica CUDA dal container

**📁 Struttura Progetto**:
```
aria-server/
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── main.py
├── config.yaml
├── aria_server/
│   ├── __init__.py
│   ├── queue_manager.py
│   ├── batch_optimizer.py
│   ├── vram_manager.py
│   ├── result_writer.py
│   ├── heartbeat.py
│   ├── semaphore.py
│   └── backends/
│       ├── __init__.py
│       └── base_backend.py
├── api/
│   ├── __init__.py
│   └── http_api.py
├── tests/
│   ├── unit/
│   └── integration/
└── scripts/
    ├── aria-download.bat
    └── aria-update.bat
```

**🔧 Implementazione**:
- [ ] Installare Docker Desktop su Windows con WSL2 backend
- [ ] Installare nvidia-container-toolkit in WSL2
- [ ] Verificare GPU dal container: `docker run --gpus all nvidia/cuda:12.1-base nvidia-smi`
- [ ] Scrivere `Dockerfile`:
  - Base image: `nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04`
  - Python 3.11, torch cu121, dipendenze base
  - Working dir `/app`, copia codice
- [ ] Scrivere `docker-compose.yml`:
  - GPU passthrough: `deploy.resources.reservations.devices`
  - Volume `C:\models` → `/models` (read-only)
  - Volume `Z:\` → `/aria-shared` (read-write)
  - Porta `7860:7860` per API HTTP
  - env_file `.env`
- [ ] Scrivere `.env.example` con tutte le variabili configurabili
- [ ] `config.yaml` base con Redis host, modelli abilitati, path
- [ ] `main.py` stub: verifica connessione Redis + `torch.cuda.is_available()` → log + exit
- [ ] `base_backend.py`: classe astratta con metodi `load()`, `unload()`, `run()`, `estimated_vram_gb()`
- [ ] `MockBackend`: implementazione mock che ritorna output fittizi in 1s
- [ ] Primo test unit: `MockBackend.run()` ritorna schema risultato valido

**📋 docker-compose.yml**:
```yaml
version: "3.8"
services:
  aria-server:
    build: .
    restart: unless-stopped
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    env_file: .env
    ports:
      - "7860:7860"
    volumes:
      - C:\models:/models:ro
      - Z:\:/aria-shared:rw
      - C:\logs\aria:/app/logs:rw
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

**✅ Criteri di Successo**:
- [ ] `docker compose up` avvia senza errori
- [ ] `nvidia-smi` visibile dall'interno del container
- [ ] `torch.cuda.is_available()` → True dal container
- [ ] `main.py` logga connessione Redis OK + CUDA OK

**✅ Criteri di Successo — COMPLETATI**:
- ✅ `docker compose up` avvia senza errori
- ✅ `nvidia-smi` visibile dall'interno del container
- ✅ `torch.cuda.is_available()` → True dal container
- ✅ FastAPI server raggiungibile su http://localhost:8000
- ✅ Health check ritorna GPU disponibile e CUDA 12.1
- ✅ Repository GitHub privato creato e codice pushato
- ✅ Struttura progetto completa con test unitari
- ✅ **Setup sviluppo rapido**: Volume mapping per modifiche immediate senza rebuild
- ✅ **Build ottimizzata**: Multi-stage build per separare dipendenze pesanti da codice leggero

**📅 Stima**: COMPLETATA (22/02/2026)

**🔧 Pattern Build Ottimizzati — AGGIORNATI**:
- **Problema risolto**: Build da 65s → 2s con volume mapping e requirements separati
- **docker-compose.dev.yml**: Hot-reload per modifiche immediate
- **Dockerfile.dev**: Base leggera senza PyTorch (installato runtime)
- **requirements-heavy.txt**: PyTorch, CUDA (cached)
- **requirements-light.txt**: FastAPI, Redis, etc (aggiornabili runtime)
- **Comando runtime**: `docker exec -it aria-aria-server-1 pip install redis==5.0.1` (senza rebuild)

---

### ✅ FASE AS-2: Redis Bridge — COMPLETATA

**🎯 Obiettivo**: Connessione stabile al Redis del minipc, lettura code, scrittura risultati, reconnect automatico

**📁 File Creati**:
```
aria_server/queue_manager.py        # BRPOP con timeout, reconnect automatico
aria_server/result_writer.py        # TTL + circuit breaker
aria_server/config.py               # Centralizzazione configurazioni
aria_server/__init__.py              # Esporta moduli Redis Bridge
tests/unit/test_redis_bridge.py     # 9/9 test passanti
```

**🔧 Implementazione Completata**:
- ✅ `QueueManager`:
  - Connessione Redis con retry infinito e backoff esponenziale
  - `next_task()`: BRPOP con timeout 2s su tutte le code configurate
  - `queue_lengths()`: LLEN su tutte le code → dict `{model_key: count}`
  - Riconnessione automatica trasparente se Redis cade
- ✅ `ResultWriter`:
  - `write_result()`: SET con TTL `result_ttl_seconds`
  - `write_processing()`: HSET su `gpu:processing:{job_id}` con TTL
  - `clear_processing()`: DEL dopo completamento
  - `get_all_processing()`: KEYS per crash recovery
- ✅ Validazione schema task: campi obbligatori e configurazione modello
- ✅ Task scaduti scartati silenziosamente con log WARNING
- ✅ Pattern code GPU: `gpu:queue:{type}:{model}` (es: `gpu:queue:tts:orpheus-3b`)

**📋 Schema Task Validato**:
```json
{
  "job_id": "job-006",
  "client_id": "test-client-001", 
  "model_type": "tts",
  "model_id": "orpheus-3b",
  "queued_at": 1740250000,
  "timeout_seconds": 300,
  "callback_key": "test-callback-001",
  "payload": {
    "text": "Ciao mondo GPU strutturato",
    "voice": "it-IT",
    "language": "it",
    "speed": 1.0
  }
}
```

**✅ Test End-to-End Verificati**:
- ✅ Lettura task da Redis minipc: `docker exec -it aria-aria-server-1 python3 -c "from aria_server import QueueManager; qm = QueueManager(); task = qm.next_task()"`
- ✅ Scrittura risultato: `GET gpu:result:*` su Redis
- ✅ Reconnect automatico dopo `redis-cli shutdown`
- ✅ Task scaduti scartati correttamente
- ✅ **9/9 test unit passanti** con pytest

**🚀 Pattern Ottimizzazione Build**:
- **Sviluppo**: `docker-compose.dev.yml` con volume mapping + hot-reload
- **Produzione**: Multi-stage build per separare PyTorch da codice
- **Tempo build**: 65s → 2s (solo codice ricostruito)
- **Runtime deps**: Installabili via `docker exec` senza rebuild

**📅 Completamento**: 22/02/2026 (1 giorno effettivo)

---

### 🔧 FASE AS-3: Logging

**🎯 Obiettivo**: Sistema di logging ibrido — console colorata per sviluppo, JSON strutturato su file per produzione

**📁 File**:
```
aria_server/logger.py
tests/unit/test_logger.py
```

**🔧 Implementazione**:
- [ ] Logger centralizzato basato su `structlog` + `logging` stdlib
- [ ] **Handler console**: output colorato con `rich` o `colorlog`
  - Format: `[HH:MM:SS] [LEVEL] [component] message {key=value}`
  - Colori: DEBUG=grigio, INFO=bianco, WARNING=giallo, ERROR=rosso
- [ ] **Handler file**: JSON strutturato con rotazione giornaliera
  - Campi fissi: `timestamp`, `level`, `component`, `job_id`, `model_id`, `message`, `duration_ms`, `error`
  - Path: `/app/logs/aria-{YYYY-MM-DD}.jsonl`
  - Rotazione: giornaliera, retention 7 giorni
- [ ] Ogni componente ha il proprio logger nominato: `aria.queue`, `aria.vram`, `aria.batch`, `aria.backend.tts`, ecc.
- [ ] Context logging: quando un task è in processing, `job_id` e `model_id` si propagano automaticamente a tutti i log del thread
- [ ] Test unit: verifica che un log produca sia output console che riga JSON corretta

**📋 Esempio output console**:
```
[10:23:45] [INFO ] [aria.queue    ] Task ricevuto job_id=abc123 model=tts:orpheus-3b priority=1
[10:23:45] [INFO ] [aria.vram     ] Orpheus-3b già in VRAM, skip load
[10:23:46] [INFO ] [aria.backend  ] Inferenza avviata text_length=320 words
[10:25:58] [INFO ] [aria.backend  ] Inferenza completata duration_s=142.1
[10:25:58] [INFO ] [aria.result   ] Risultato scritto callback_key=gpu:result:dias-minipc:abc123
```

**📋 Esempio riga JSON su file**:
```json
{"ts":"2026-02-20T10:25:58Z","level":"INFO","component":"aria.backend","job_id":"abc123","model_id":"orpheus-3b","event":"inference_completed","duration_ms":142100}
```

**✅ Criteri di Successo**:
- [ ] Console mostra log colorati leggibili durante `docker compose up`
- [ ] File JSONL creato e scritto correttamente in `/app/logs/`
- [ ] Rotazione giornaliera funzionante
- [ ] `job_id` propagato correttamente in tutti i log di un task

**📅 Stima**: 2-3 giorni

---

### 🔧 FASE AS-4: VRAM Manager

**🎯 Obiettivo**: Load/unload modelli VRAM in modo sicuro, monitoraggio utilizzo, gestione OOM

**📁 File**:
```
aria_server/vram_manager.py
tests/unit/test_vram_manager.py
tests/integration/test_vram_real.py
```

**🔧 Implementazione**:
- [ ] `VRAMManager`:
  - `load(backend: BaseBackend)`: verifica VRAM disponibile, chiama `backend.load()`, registra stato
  - `unload(backend: BaseBackend)`: chiama `backend.unload()`, `torch.cuda.empty_cache()`, `gc.collect()`
  - `current_model()` → `BaseBackend | None`
  - `vram_stats()` → `{used_gb, free_gb, total_gb}`
  - `can_load(backend)` → bool: `free_gb > backend.estimated_vram_gb() + 1.0` (margine 1GB)
- [ ] Gestione `torch.cuda.OutOfMemoryError`:
  - Cattura eccezione durante `backend.run()`
  - Tenta unload + reload con payload ridotto (strategia per tipo: TTS → chunk più piccoli, Image → risoluzione dimezzata)
  - Se OOM persiste dopo retry → scrive risultato con `status=error, error_code=OOM`
- [ ] Stato VRAM pubblicato su Redis ogni 30s: `gpu:server:vram`
- [ ] Unit test con mock torch (no GPU reale)
- [ ] Integration test: carica Orpheus, verifica VRAM usata, unload, verifica VRAM liberata

**✅ Criteri di Successo**:
- [ ] Load/unload Orpheus senza memory leak (VRAM identica prima e dopo)
- [ ] OOM gestito senza crash del broker
- [ ] `can_load()` impedisce caricamento se VRAM insufficiente
- [ ] VRAM stats corrette su Redis

**📅 Stima**: 3-5 giorni

---

### 🔧 FASE AS-5: Batch Optimizer

**🎯 Obiettivo**: Scheduling intelligente priority-first poi greedy — decide quale modello caricare in base a priorità task e numero task in coda

**📁 File**:
```
aria_server/batch_optimizer.py
tests/unit/test_batch_optimizer.py
```

**🔧 Implementazione**:
- [ ] `BatchOptimizer.next_model(queue_lengths, current_model)` → `model_key | None`:
  1. Legge tutte le code non vuote
  2. Trova la priorità massima presente tra tutti i task in testa a ogni coda
     (legge solo il primo elemento di ogni coda senza consumarlo: LINDEX ... 0)
  3. Filtra le code che hanno task a quella priorità massima
  4. Tra le code filtrate: sceglie quella con più task (greedy)
  5. Se il modello scelto è già in VRAM → nessun cambio (preferenza per modello corrente a parità)
  6. Se tutte le code sono vuote → ritorna None (attendi)
- [ ] `BatchOptimizer.should_switch(current_model, queue_lengths)` → bool:
  - False se la coda del modello corrente non è vuota (continua a drenare)
  - True se la coda del modello corrente è vuota e altre code hanno task
- [ ] `batch_wait_seconds`: aspetta N secondi dopo che una coda si svuota prima di cambiare modello (potrebbero arrivare nuovi task dello stesso tipo)
- [ ] Log decisione scheduler a ogni cambio di modello: quale coda, quanti task, perché
- [ ] Test con scenari simulati:
  - 3 task priority=1 orpheus + 1 task priority=3 llm → llm eseguito prima
  - 5 task orpheus + 2 task musicgen → orpheus eseguito prima (greedy)
  - Coda orpheus si svuota → attende `batch_wait_seconds` → passa a musicgen

**📋 Logica priority-first + greedy**:
```
CODE IN CODA:
  tts:orpheus-3b    → [task(p=1), task(p=1), task(p=1)]   3 task, max_priority=1
  music:musicgen    → [task(p=2), task(p=1)]               2 task, max_priority=2
  llm:llama-3b      → [task(p=3)]                          1 task, max_priority=3

STEP 1: max priority globale = 3
STEP 2: code con task a priority=3 → solo llm:llama-3b
STEP 3: greedy su quelle code → llm:llama-3b (unica)
DECISIONE: carica LLM

Dopo llm terminato:
STEP 1: max priority globale = 2
STEP 2: code con task a priority=2 → music:musicgen
DECISIONE: carica MusicGen

Dopo musicgen terminato:
STEP 1: max priority globale = 1
STEP 2: code con priority=1 → tts:orpheus (3 task), music:musicgen (1 task)
STEP 3: greedy → tts:orpheus (3 > 1)
DECISIONE: carica Orpheus
```

**✅ Criteri di Successo**:
- [ ] Task priority=3 sempre eseguiti prima di priority=1 indipendentemente dal modello
- [ ] A parità di priorità: modello con più task vince
- [ ] Modello già in VRAM preferito a parità di tutto
- [ ] `batch_wait_seconds` rispettato prima di cambio modello
- [ ] 100% test unit passano

**📅 Stima**: 1 settimana

---

### 🔧 FASE AS-6: Backend TTS Orpheus — MVP

**🎯 Obiettivo**: Primo backend reale — inferenza Orpheus TTS, produzione WAV, integrazione completa con la pipeline. Questo è il punto in cui ARIA diventa funzionante end-to-end con DIAS.

**📁 File**:
```
aria_server/backends/tts_backend.py
tests/unit/test_tts_backend_mock.py
tests/integration/test_tts_orpheus_real.py
```

**🔧 Prerequisiti** (da completare prima di sviluppare):
- [ ] Scaricare modello: `aria-download.bat orpheus-3b` (implementato in AS-11, o manuale HF Hub)
- [ ] Verificare modello in `C:\models\orpheus-3b-q4\`
- [ ] Campione voce narratore: `Z:\voices\narrator_it.wav` (WAV 48kHz mono, 3-30s)
- [ ] Samba share montata come `Z:\` (implementato in AS-15, o montata manualmente)

**🔧 Implementazione**:
- [ ] `OrpheusBackend(BaseBackend)`:
  - `load()`: carica modello da `/models/orpheus-3b-q4/` con `transformers`, device=cuda
  - `unload()`: `self._model = None`, `self._tokenizer = None`, `torch.cuda.empty_cache()`, `gc.collect()`
  - `estimated_vram_gb()` → 7.0
  - `run(payload)`:
    - Legge `payload.text` (già annotato con tag Orpheus da DIAS TextDirector)
    - Legge campione voce da `payload.voice_sample_path` (path su `/aria-shared/`)
    - Applica `pace_factor`: aggiunge `<slow>` o `<fast>` prefix se fuori range 0.85-1.15
    - Esegue inferenza Orpheus
    - Se testo >280 parole: chunking per frase + crossfade 80ms con FFmpeg
    - Salva WAV 48kHz mono in `payload.output_path` (path su `/aria-shared/`)
    - Ritorna `{output_path, duration_seconds, sample_rate: 48000}`
- [ ] `MockTTSBackend`: ritorna WAV silenzioso di durata proporzionale alla lunghezza testo
- [ ] Gestione `torch.cuda.OutOfMemoryError`: dimezza chunk size, retry automatico
- [ ] Unit test con MockTTSBackend (no GPU)
- [ ] Integration test su GPU reale con paragrafo di esempio italiano

**📋 Payload Orpheus atteso**:
```json
{
  "text": "leah: Aprì la porta lentamente. <gasp> Non c'era nessuno.",
  "voice_name": "leah",
  "voice_sample_path": "/aria-shared/voices/narrator_it.wav",
  "pace_factor": 0.82,
  "output_path": "/aria-shared/output/dias-minipc/book_123/scene_001.wav",
  "output_format": "wav",
  "sample_rate": 48000,
  "channels": 1
}
```

**✅ Criteri di Successo — MVP**:
- [ ] WAV generato correttamente in `/aria-shared/output/`
- [ ] Tag emotivi (`<gasp>`, `<sigh>`) presenti nell'audio
- [ ] Chunking corretto per scene >280 parole
- [ ] Voice cloning fedele al campione fornito
- [ ] Processing time < 2x realtime su RTX 5060 Ti
- [ ] **Test end-to-end con DIAS**: Stage C invia task → ARIA genera WAV → Watcher aggiorna stato ✅

**📅 Stima**: 2 settimane

---

### 🔧 FASE AS-7: Result Writer + Crash Recovery

**🎯 Obiettivo**: Scrittura risultati robusta con circuit breaker, recovery automatico dei task in processing al riavvio

**📁 File**:
```
aria_server/result_writer.py      (esteso)
aria_server/crash_recovery.py
tests/unit/test_crash_recovery.py
```

**🔧 Implementazione**:
- [ ] **Visibility timeout pattern**:
  - Prima di `backend.run()`: `HSET gpu:processing:{job_id} {...task} EX processing_timeout_s`
  - Dopo `backend.run()` con successo: `DEL gpu:processing:{job_id}`
  - Se eccezione: `DEL gpu:processing:{job_id}` + scrivi errore in risultato
- [ ] **Crash recovery all'avvio** (`crash_recovery.py`):
  - `KEYS gpu:processing:*` → lista task interrotti
  - Per ogni task: legge `retry_count` dal task
  - Se `retry_count < max_retries`: `LPUSH` sulla coda originale con `retry_count + 1`
  - Se `retry_count >= max_retries` (circuit breaker): `HSET gpu:dead:{client_id}:{job_id}` con motivo `CRASH_CIRCUIT_BREAKER`
  - Log chiaro per ogni decisione
- [ ] **Scrittura risultato**:
  - Successo: `SET gpu:result:{client_id}:{job_id} {json} EX result_ttl_seconds`
  - Errore: stesso schema con `status=error`, `error_code`, `error_message`
- [ ] Test unit crash recovery con scenari:
  - Task a `retry_count=0` → riaccodato
  - Task a `retry_count=max_retries` → dead letter
  - Nessun task in processing → avvio normale

**✅ Criteri di Successo**:
- [ ] Crash simulato (kill -9 container) → task riaccodato al riavvio
- [ ] Task patologico (crash 3 volte) → finisce in dead letter al 4° avvio
- [ ] Risultati scritti con TTL corretto (verificato con `TTL gpu:result:*`)

**📅 Stima**: 3-5 giorni

---

### 🔧 FASE AS-8: API HTTP — Semaforo e Status

**🎯 Obiettivo**: API FastAPI minimale per controllo semaforo e monitoring, raggiungibile dalla LAN

**📁 File**:
```
api/http_api.py
tests/unit/test_http_api.py
```

**🔧 Implementazione**:
- [ ] FastAPI app con uvicorn su `0.0.0.0:7860`
- [ ] Middleware CORS per accesso da minipc e altri device LAN
- [ ] API Key header `X-API-Key` per endpoint privilegiati
- [ ] Endpoints:
  - `GET /health` → `{status: ok, uptime_seconds}` — no auth
  - `GET /status` → stato completo (semaforo, modello caricato, VRAM, code, task corrente) — no auth
  - `POST /semaphore` body `{state: green|red}` → cambia semaforo — **auth richiesta**
  - `GET /queue/{model_type}/{model_id}` → `{length, oldest_age_seconds}` — no auth
  - `DELETE /queue/{model_type}/{model_id}/{job_id}` → cancella task da coda — **auth richiesta**
  - `GET /models` → lista modelli configurati con stato loaded/unloaded
  - `GET /logs/recent` → ultime 50 righe log JSON — no auth
- [ ] Semaphore state persistito su Redis (`gpu:server:semaphore`) — sopravvive a restart API
- [ ] Test unit con FastAPI TestClient

**✅ Criteri di Successo**:
- [ ] `GET /health` risponde in <100ms
- [ ] `POST /semaphore` senza API key → 401
- [ ] `POST /semaphore {"state":"red"}` → semaforo cambia su Redis verificato
- [ ] API raggiungibile da minipc via `curl http://192.168.1.20:7860/status`

**📅 Stima**: 3-5 giorni

---

### 🔧 FASE AS-9: Dashboard Web

**🎯 Obiettivo**: Pagina HTML servita da ARIA su `localhost:7860/dashboard` con monitoring dettagliato e controllo semaforo

**📁 File**:
```
api/dashboard.html         (o template Jinja2)
api/http_api.py            (aggiunge route /dashboard)
```

**🔧 Implementazione**:
- [ ] Pagina HTML single-file (no framework JS pesanti — vanilla JS + CSS)
- [ ] Auto-refresh ogni 5s via `fetch /status`
- [ ] Sezioni:
  - **Stato server**: semaforo con colore (verde/rosso/grigio offline), uptime, versione
  - **VRAM**: barra di utilizzo visuale, modello caricato, GB usati/liberi/totali
  - **Code**: tabella per ogni modello con numero task in attesa e task più vecchio
  - **Task corrente**: job_id, client, modello, tempo trascorso, progress stimato
  - **Storico**: ultimi 20 task completati con durata e stato
  - **Log live**: ultime 20 righe log in tempo reale
- [ ] Pulsanti semaforo: "🟢 GPU Disponibile" / "🔴 GPU Occupata" con conferma
- [ ] Responsive: leggibile anche da mobile sulla stessa LAN
- [ ] Nessun login richiesto (LAN privata) — API key solo per azioni privilegiate

**✅ Criteri di Successo**:
- [ ] Dashboard aperta da browser su `http://localhost:7860/dashboard`
- [ ] Stato aggiornato ogni 5s senza refresh manuale
- [ ] Click "GPU Occupata" → semaforo diventa rosso su Redis
- [ ] Raggiungibile da minipc su `http://192.168.1.20:7860/dashboard`

**📅 Stima**: 1 settimana

---

### 🔧 FASE AS-10: Tray Icon

**🎯 Obiettivo**: Icona system tray Windows per controllo semaforo rapido senza aprire browser — processo separato dal container Docker

**📁 File**:
```
aria-tray/
├── tray_icon.py
├── requirements-tray.txt    (pystray, Pillow, requests)
└── install-tray-service.bat
```

> **Nota**: La tray icon gira come processo Python nativo Windows (fuori Docker)
> perché Docker non ha accesso alla GUI di Windows. Chiama l'API HTTP di ARIA Server.

**🔧 Implementazione**:
- [ ] `pystray` + `Pillow` per icona dinamica nella system tray
- [ ] Icona con colore dinamico generata con Pillow:
  - 🟢 Verde: semaforo green
  - 🔴 Rosso: semaforo red
  - 🟡 Giallo: task in esecuzione (busy)
  - ⚫ Grigio: server offline
- [ ] Aggiornamento stato ogni 5s via `GET /status`
- [ ] Menu click destro:
  - "🟢 GPU Disponibile" (→ `POST /semaphore green`)
  - "🔴 GPU Occupata" (→ `POST /semaphore red`)
  - Separatore
  - "📊 Apri Dashboard" (→ apre browser su `localhost:7860/dashboard`)
  - "ℹ️ Stato: {modello caricato, VRAM usata}"
  - Separatore
  - "❌ Esci"
- [ ] Tooltip hover: "{semaphore} — {n} task in coda — VRAM {x}GB/{y}GB"
- [ ] Si avvia automaticamente con Windows (registro startup o Task Scheduler)
- [ ] `install-tray-service.bat`: configura avvio automatico

**✅ Criteri di Successo**:
- [ ] Icona visibile nella system tray al boot Windows
- [ ] Colore cambia entro 5s quando semaforo cambia
- [ ] Click "GPU Occupata" → semaforo red verificato su Redis
- [ ] "Apri Dashboard" apre il browser correttamente

**📅 Stima**: 1 settimana

---

### 🔧 FASE AS-11: `aria download` — Gestione Modelli

**🎯 Obiettivo**: Comando CLI per scaricare modelli da HuggingFace Hub in `C:\models\`, con verifica integrità

**📁 File**:
```
scripts/aria-download.bat
aria_server/model_downloader.py
tests/unit/test_model_downloader.py
```

**🔧 Implementazione**:
- [ ] `model_downloader.py`:
  - Legge `config.yaml` per trovare `model_id` HF e path destinazione
  - Usa `huggingface_hub.snapshot_download()` per scaricare in `C:\models\{model_id}\`
  - Mostra progress bar durante download
  - Verifica checksum dopo download (sha256 dei file principali)
  - Scrive `C:\models\{model_id}\.aria-manifest.json` con versione, data download, checksum
- [ ] `aria-download.bat`:
  ```bat
  @echo off
  python aria_server/model_downloader.py %1
  ```
- [ ] Modelli supportati in config con loro HF repo ID:
  ```yaml
  downloadable_models:
    orpheus-3b:
      repo_id: "canopylabs/orpheus-3b-0.1-ft"
      local_path: "C:/models/orpheus-3b-q4"
    musicgen-small:
      repo_id: "facebook/musicgen-small"
      local_path: "C:/models/musicgen-small"
  ```
- [ ] Comando `aria-download.bat list` → mostra modelli disponibili e stato (scaricato/non scaricato)
- [ ] Comando `aria-download.bat orpheus-3b` → scarica solo quel modello

**✅ Criteri di Successo**:
- [ ] `aria-download.bat orpheus-3b` scarica modello in `C:\models\orpheus-3b-q4\`
- [ ] Re-esecuzione su modello già scaricato → skip con messaggio "già presente"
- [ ] Download corrotto rilevato dal checksum → re-download automatico
- [ ] `aria-download.bat list` mostra stato corretto

**📅 Stima**: 3-5 giorni

---

### 🔧 FASE AS-12: `aria update` — Script Aggiornamento

**🎯 Obiettivo**: Un comando che aggiorna ARIA Server in modo sicuro e riproducibile

**📁 File**:
```
scripts/aria-update.bat
```

**🔧 Implementazione**:
- [ ] `aria-update.bat`:
  ```bat
  @echo off
  echo [ARIA Update] Fermo il container...
  docker compose down
  echo [ARIA Update] Aggiorno il codice...
  git pull origin main
  echo [ARIA Update] Rebuild immagine Docker...
  docker compose build --no-cache
  echo [ARIA Update] Riavvio...
  docker compose up -d
  echo [ARIA Update] Fatto! Versione:
  docker compose exec aria-server python -c "import aria_server; print(aria_server.__version__)"
  ```
- [ ] Verifica pre-update: controlla che nessun task sia in `gpu:processing:*` prima di fermare (se sì, avverte e chiede conferma)
- [ ] Backup `config.yaml` prima dell'update (non viene sovrascritto da git se in `.gitignore`)
- [ ] Log dell'update con timestamp in `C:\logs\aria\updates.log`

**✅ Criteri di Successo**:
- [ ] `aria-update.bat` completa senza intervento manuale
- [ ] Config.yaml non sovrascritto
- [ ] Container riavviato con nuova versione verificata

**📅 Stima**: 1-2 giorni

---

### 🔧 FASE AS-13: Backend MusicGen

**🎯 Obiettivo**: Secondo backend reale — AudioCraft MusicGen Small, produzione WAV stereo adattivo alla scena

**📁 File**:
```
aria_server/backends/music_backend.py
tests/unit/test_music_backend_mock.py
tests/integration/test_music_real.py
```

**🔧 Implementazione**:
- [ ] `MusicGenBackend(BaseBackend)`:
  - `load()`: carica AudioCraft MusicGen Small da `/models/musicgen-small/`
  - `estimated_vram_gb()` → 4.0
  - `run(payload)`:
    - Legge `payload.prompt` (prompt testuale musica)
    - Legge `payload.duration_seconds` (durata target)
    - Genera audio con MusicGen
    - Se `duration_seconds > 30` e `loop_seamless=true`: genera con continuation
    - Verifica seamless loop: energia ultimi 0.5s ≈ energia primi 0.5s (soglia 10%)
    - Salva WAV stereo 48kHz in `payload.output_path`
    - Ritorna `{output_path, duration_seconds, loop_verified}`
- [ ] `MockMusicBackend`: ritorna WAV silenzioso stereo di durata corretta

**✅ Criteri di Successo**:
- [ ] Musica generata coerente col prompt
- [ ] Loop seamless per durate >30s (verificato ad ascolto)
- [ ] WAV stereo 48kHz in output

**📅 Stima**: 1 settimana

---

### 🔧 FASE AS-14: Backend LLM Testuale

**🎯 Obiettivo**: Terzo backend — LLM testuale (Qwen 2.5 7B o Llama 3.1 8B) con transformers + bitsandbytes

**📁 File**:
```
aria_server/backends/llm_backend.py
tests/unit/test_llm_backend_mock.py
tests/integration/test_llm_real.py
```

**🔧 Implementazione**:
- [ ] `LLMBackend(BaseBackend)`:
  - `load()`: carica modello con `transformers` + `bitsandbytes` (quantizzazione 4-bit)
  - `estimated_vram_gb()` → 5.5 (4-bit)
  - `run(payload)`:
    - Legge `payload.messages` (formato chat OpenAI-compatible)
    - Applica chat template del modello
    - Genera risposta con `model.generate()`
    - Se `response_format=json`: valida output come JSON, retry se non valido (max 2)
    - Ritorna `{content, usage: {prompt_tokens, completion_tokens}}`
- [ ] Nota: questo backend rende possibile usare LLM locali al posto di Gemini per DIAS in modalità fully-offline

**✅ Criteri di Successo**:
- [ ] Risposta coerente in italiano su prompt di test
- [ ] `response_format=json` produce JSON valido
- [ ] VRAM < 6GB con quantizzazione 4-bit

**📅 Stima**: 1 settimana

---

### 🔧 FASE AS-15: Samba — Cartella Condivisa

**🎯 Obiettivo**: Configurare la condivisione SMB sul minipc e il mount sul gaming PC, documentare il processo

> **Nota**: Questa fase può essere eseguita in parallelo con AS-1/AS-2 —
> non dipende dal codice ARIA ma è prerequisito per AS-6 (Orpheus).

**🔧 Implementazione**:
- [ ] **Sul minipc (Linux)**:
  ```bash
  sudo apt install samba
  sudo mkdir -p /mnt/aria-shared/{voices,input,output}
  sudo chown -R dias:dias /mnt/aria-shared
  # Aggiungere a /etc/samba/smb.conf:
  # [aria-shared]
  #   path = /mnt/aria-shared
  #   read only = no
  #   guest ok = yes
  sudo systemctl restart smbd
  ```
- [ ] **Sul gaming PC (Windows)**:
  - Esplora risorse → `\\minipc\aria-shared` → Mappa unità di rete → `Z:`
  - Oppure via PowerShell: `New-PSDrive -Name Z -PSProvider FileSystem -Root \\minipc\aria-shared -Persist`
- [ ] Verificare accesso da Docker container via `/aria-shared/` (volume mount in docker-compose)
- [ ] Copiare campione voce narratore in `Z:\voices\narrator_it.wav`
- [ ] Test write/read da entrambi i lati

**✅ Criteri di Successo**:
- [ ] Gaming PC legge/scrive su `Z:\` senza credenziali
- [ ] Container Docker legge/scrive su `/aria-shared/`
- [ ] Minipc legge output generato da ARIA in `/mnt/aria-shared/output/`

**📅 Stima**: 1-2 giorni

---

## 🎯 MILESTONE MVP

**MVP raggiunto dopo AS-6** quando:
- ARIA Server gira in Docker con GPU
- Riceve task TTS Orpheus da Redis minipc
- Genera WAV in `/aria-shared/output/`
- Scrive risultato su Redis
- DIAS Watcher lo trova e aggiorna stato pipeline

Tutto il resto (dashboard, tray icon, MusicGen, LLM, script update) è
miglioramento progressivo su una base già funzionante.

---

## 📈 METRICHE DI SUCCESSO GLOBALI

- [ ] TTS Orpheus: processing < 2x realtime su RTX 5060 Ti
- [ ] MusicGen: 60s audio generato in <30s
- [ ] Broker overhead: <500ms tra ricezione task e inizio inferenza
- [ ] Semaforo red: broker in pausa entro 60s (finisce task corrente)
- [ ] Zero task persi su crash (verificato con test crash simulato)
- [ ] Recovery automatico entro 30s dal riavvio
- [ ] Dashboard aggiornata ogni 5s senza lag percepibile
- [ ] Tray icon colore corretto entro 5s da cambio semaforo

---

## 📝 NOTE PER L'AGENT

### Ordine sviluppo obbligatorio
AS-1 → AS-2 → AS-3 → AS-4 → AS-5 → AS-6 (**MVP**)
→ AS-7 → AS-8 → AS-9 → AS-10 → AS-11 → AS-12
→ AS-13 → AS-14

AS-15 (Samba) può partire in parallelo da subito — è infrastruttura, non codice.

### Sviluppo offline
Tutto fino ad AS-5 sviluppabile e testabile senza GPU reale usando MockBackend.
AS-6 richiede GPU per i test di integrazione — i test unit restano con mock.

### Pattern mock consistente
Ogni backend ha `Mock{Backend}` nella stessa directory.
`MOCK_GPU=true` in `.env` → tutti i backend usano mock automaticamente.
Stessa filosofia di `MOCK_SERVICES=true` in DIAS.

### Tray icon fuori Docker
La tray icon (`aria-tray/`) è un progetto Python separato che gira su
Windows nativo (non nel container). Ha il suo `requirements-tray.txt`
e si installa con `pip install -r requirements-tray.txt` nell'ambiente
Windows base, non nel venv del container.

### Dipendenze container (requirements.txt)
```
torch>=2.3.0+cu121
transformers>=4.40.0
accelerate>=0.30.0
bitsandbytes>=0.43.0
audiocraft>=1.0.0
soundfile>=0.12.1
fastapi>=0.110.0
uvicorn>=0.29.0
redis>=5.0.0
pyyaml>=6.0
structlog>=24.0.0
rich>=13.0.0
huggingface-hub>=0.22.0
ffmpeg-python>=0.2.0
```

### Dipendenze tray (requirements-tray.txt)
```
pystray>=0.19.5
Pillow>=10.0.0
requests>=2.31.0
```
