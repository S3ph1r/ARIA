# ARIA — Adaptive Resource for Inference and AI
## Blueprint Funzionale v1.0

> Piattaforma di inferenza AI privata e distribuita per reti domestiche e locali.
> ARIA trasforma qualsiasi PC con GPU in un servizio di inferenza condiviso,
> accessibile da qualsiasi device sulla stessa rete come se fosse un'API cloud —
> con la differenza che gira a casa tua, senza costi ricorrenti e senza privacy compromessa.

---

## INDICE

1. [Visione e Principi](#1-visione-e-principi)
2. [Architettura di Sistema](#2-architettura-di-sistema)
3. [Componenti](#3-componenti)
   - 3.1 ARIA Server (GPU Host)
   - 3.2 ARIA Client (Library)
   - 3.3 ARIA Redis Bus (Shared)
4. [Modelli Supportati e Backend](#4-modelli-supportati-e-backend)
5. [Schema Task — Specifiche Complete](#5-schema-task--specifiche-complete)
6. [Ciclo di Vita di un Task](#6-ciclo-di-vita-di-un-task)
7. [Gestione Semaforo e Disponibilità GPU](#7-gestione-semaforo-e-disponibilitgpu)
8. [Scenari di Utilizzo e Comportamento](#8-scenari-di-utilizzo-e-comportamento)
9. [Gestione File Binari](#9-gestione-file-binari)
10. [Multi-Client: Routing e Isolamento](#10-multi-client-routing-e-isolamento)
11. [API HTTP di ARIA Server](#11-api-http-di-aria-server)
12. [Configurazione](#12-configurazione)
13. [Sicurezza](#13-sicurezza)
14. [Limiti e Vincoli Noti](#14-limiti-e-vincoli-noti)

---

## 1. Visione e Principi

### La metafora

ARIA funziona come una **stampante di rete intelligente** per l'AI generativa.
Esattamente come una stampante di rete:
- È sempre disponibile per chi è sulla rete (quando accesa)
- Accoda i lavori da più dispositivi simultaneamente
- Esegue i lavori quando la risorsa è libera
- Notifica il mittente quando il lavoro è completato
- Non richiede che il mittente resti in attesa

La differenza rispetto a una stampante: ARIA è **asíncrona per design**. Il
client invia il task e continua a lavorare. Il risultato arriva quando la GPU
ha finito — potrebbe essere tra 30 secondi, potrebbe essere dopo che l'utente
ha finito di giocare. Il client è progettato per gestire questa latenza variabile.

### Principi fondamentali

**1. Agnosticismo e AI-as-a-Service (AIaaS)**
ARIA Server non conosce DIAS, non conosce nessun progetto specifico.
L'interazione avviene tramite **Intenti**: il client non dice ad ARIA *come* 
lavorare (path, file specifici, prompt tecnici), ma *cosa* desidera ottenere.
Qualsiasi client può usare ARIA per qualsiasi scopo tramite un'interfaccia 
standardizzata e disaccoppiata.

**2. Autonomia degli Asset**
ARIA è il proprietario della propria "Libreria di Intenzioni". Gestisce 
internamente le voci (Voice Library), i modelli (Model Registry) e i 
template di prompt. Il client invia un ID astratto (es. `voice: "narratore"`), 
e ARIA risolve autonomamente i file necessari (`ref.wav`, `ref.txt`).

**2. Non-blocking sempre**
`submit_task()` ritorna in <100ms in qualsiasi scenario — GPU occupata,
PC spento, semaforo rosso. Il task viene accodato o rifiutato con un codice
chiaro, mai con un timeout sospeso.

**3. Zero perdita di task**
Un task scritto su Redis è persistente. Se il Server crasha durante l'esecuzione,
il task viene rimesso in coda al riavvio. Se il PC è spento, il task aspetta.
L'unico modo in cui un task sparisce è: completamento, scadenza esplicita (TTL),
o cancellazione esplicita dal client.

**4. Un modello alla volta in VRAM**
La RTX 5060 Ti ha 16GB. Caricare due modelli grandi contemporaneamente
causa OOM o degradazione. ARIA carica un modello, esegue tutti i task
disponibili per quel modello, poi decide se cambiare. La decisione è del
Batch Optimizer, non del client.

**5. Intercambiabilità dei backend**
Ogni tipo di modello ha un backend Python. L'interfaccia è identica per tutti:
`load()`, `unload()`, `run(payload) → result`. Aggiungere supporto a un nuovo
modello significa scrivere un nuovo backend — niente altro cambia.

**6. Privacy totale**
Nessun dato lascia la rete locale. Nessun log remoto. Nessuna telemetria.
Il codice è open source e ispezionabile.

---

## 2. Architettura di Sistema

### Topologia fisica

```
╔══════════════════════════════════════════════════════════════╗
║                     RETE LOCALE (LAN)                        ║
║                                                              ║
║  ┌─────────────────────┐      ┌──────────────────────────┐  ║
║  │   BRAIN NODE        │      │   WORKER NODE (GPU)      │  ║
║  │                     │      │                          │  ║
║  │  Narrative Engine   │◄────►│  ARIA SERVER             │  ║
║  │  (DIAS, etc.)       │      │  (Inference Service)     │  ║
║  │                     │      │                          │  ║
║  │  INFRASTRUCTURE ◄───┼──────┼─── legge/scrive code     │  ║
║  │  (Redis Store)      │      │                          │  ║
║  └─────────────────────┘      │  Hardware Accelerato     │  ║
║             │                 └──────────────────────────┘  ║
║             │                               ▲                ║
║             └──────── SSH (Management) ──────┘                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

**Filosofia Agnostica**: ARIA non è legata a un IP specifico. La scoperta dei nodi avviene tramite il registro degli heartbeat su Redis. Per le specifiche tecniche di comunicazione, consultare [ARIA-network-interface.md](ARIA-network-interface.md).

### Flusso dati ad alto livello

```
CLIENT                    REDIS (Infrastructure)      ARIA WORKER (GPU Node)
  │                           │                               │
  │── submit_task() ─────────►│                               │
  │   (vedi Interface Spec)   │                               │
  │◄─ job_id ─────────────────│                               │
  │                           │◄──── fetch_task() ────────────│
  │                           │                               │
  │                           │                               │── carica modello
  │                           │                               │── esegue inferenza
  │                           │                               │── salva output
  │                           │◄──── post_result() ───────────│
  │                           │                               │
  │── get_result() ──────────►│                               │
  │◄─ result ─────────────────│                               │
```

### Componenti software

```
ARIA NODE CONTROLLER (Windows — Nodo GPU — %ARIA_ROOT%)
├── aria_node_controller/              # Orchestratore e logica di controllo
│   ├── main_tray.py                   # Entry point + Tray Icon (systray semaforo)
│   ├── settings_gui.py                # GUI impostazioni (CustomTkinter)
│   ├── qwen3_server.py                # Server FastAPI Qwen3-TTS (porta 8083)
│   ├── core/
│   │   ├── orchestrator.py            # Loop principale, dispatch task, process manager
│   │   ├── cloud_manager.py           # Gestore sequenziale task Cloud (Gemini) [v2.0]
│   │   ├── rate_limiter.py            # Centralized Gemini Quota & Pacing [v2.0]
│   │   ├── queue_manager.py           # BRPOP da Redis, routing code
│   │   ├── batch_optimizer.py         # Decide quale modello caricare
│   │   ├── models.py                  # Pydantic models (AriaTaskResult, ecc.)
│   │   ├── config_manager.py          # Lettura node_settings.json
│   │   └── logger.py                  # Structured logging
│   └── backends/
│       ├── qwen3_tts.py               # Backend HTTP per Qwen3-TTS
│       └── cloud/                     # Backends per modelli remoti [NEW]
│           └── gemini_worker.py       # Worker isolato per Google GenAI
├── envs/                              # Ambienti Python isolati (project-local)
│   ├── qwen3tts/                      # Python 3.12 + PyTorch + qwen-tts
│   └── fish-speech/                   # Repo Fish + (futuro) env Python 3.10
├── data/
│   ├── models/                        # Pesi dei modelli
│   ├── voices/                        # Voice Library
│   └── outputs/                       # WAV generati (serviti via HTTP :8082)
├── Avvia_Tutti_Server_ARIA.bat        # Script avvio principale
└── node_settings.json                 # Configurazione nodo (Network role: Worker)
```
Per i dettagli sulla configurazione e l'accesso a Redis, consultare [ARIA-network-interface.md](ARIA-network-interface.md).

---

## 3. Componenti

### 3.1 ARIA Server

ARIA Server è il processo che gira sul PC con GPU. Ha una sola responsabilità:
**ricevere task da Redis, eseguirli sulla GPU, scrivere i risultati su Redis**.

Non espone direttamente i modelli. Non conosce i client. Non ha stato applicativo
oltre alla coda corrente. È stateless rispetto ai progetti — tutto lo stato
vive su Redis.

Il loop principale:

```
    1. Leggi stato semaforo → se RED: attendi, non consumare task
    2. Chiedi a BatchOptimizer: quale modello caricare?
    3. Se modello "cloud-gemini":
       a. Chiama RateLimiter.wait_for_slot() (Smart Pacing/Quota)
       b. Esegui Task via CloudManager
       c. Se errore 429: RateLimiter.report_429() (Global Lockout)
    4. Se modello locale (es. `llm` via Qwen 3.5):
       a. BatchOptimizer avvia `llama-server.exe` (se non attivo)
       b. Consuma task dalla coda `global:queue:llm:local:{model_id}`
    5. Se modello locale diverso da quello in VRAM: unload → load nuovo
    6. Consuma task dalla coda del modello scelto (BRPOP)
    7. **Risoluzione Intent (Aria-side)**:
       - L'Orchestratore analizza il payload.
       - Se presente `voice_id`, `intent_id` o `theme_id`, consulta il registro locale.
       - Inietta i path fisici (`ref.wav`, `ref.txt`, `system_prompt`) nel payload.
    8. Esegui inferenza con backend appropriato
    9. Scrivi risultato in `global:callback:{client_id}:{job_id}` (v2.1) o `gpu:result:{client_id}:{job_id}` (legacy)
    10. Torna a 2
```

### 3.2 ARIA Client

ARIA Client è una **libreria Python** che si installa su qualsiasi device
sulla rete. Non è un servizio — è importata dal codice applicativo.

Interfaccia pubblica:

```python
client = ARIAClient(redis_host="192.168.1.10", client_id="dias-minipc")

# Invia task — ritorna sempre in <100ms
job_id = client.submit_task(
    model_type="tts",
    model_id="orpheus-3b",
    payload={...},
    priority=1,
    timeout_seconds=1800
)

# Controlla risultato senza bloccare
result = client.get_result(job_id)  # → result dict | None

# Aspetta risultato (blocca — usare con cautela)
result = client.wait_result(job_id, timeout_seconds=3600)

# Stato del server
status = client.server_status()     # → {semaphore, vram, queues, ...}
available = client.is_available()   # → bool
```

Il Watcher è un componente opzionale del client — un thread background che
monitora continuamente i risultati e chiama callback registrate:

```python
def on_voice_ready(job_id, result):
    # aggiorna stato DIAS quando Orpheus ha finito
    update_scene_state(job_id, result)

client.watcher.register_callback("tts", on_voice_ready)
client.watcher.start()
```

### 3.3 ARIA Redis Bus

Redis è il **sistema nervoso** di ARIA. Non è un componente di ARIA — è
l'infrastruttura su cui ARIA opera. Deve essere sempre accessibile.

Struttura chiavi completa:

```
# Code di input (scritte dal Client, lette dal Server)
# Pattern Locale: gpu:queue:{model_type}:{model_id}
gpu:queue:tts:orpheus-3b
gpu:queue:music:musicgen-small

# Pattern Cloud Gateway (v2.0): global:queue:cloud:{provider}:{model_id}:{client_id}
global:queue:cloud:google:gemini-flash-lite-latest:dias

# Task in esecuzione (visibility timeout, prevenzione perdita su crash)
gpu:processing:{job_id}               # Hash, TTL = timeout_seconds task

# Risultati (scritti dal Server, letti dal Client)
gpu:result:{client_id}:{job_id}       # String JSON, TTL configurabile
  es: gpu:result:dias-minipc:uuid-123

# Stato Server (scritto dal Server ogni 10s)
gpu:server:status                     # Hash: {status, active_backends, available_voices, vram}
gpu:server:semaphore                  # String: "green" | "red" | "busy"
gpu:server:heartbeat                  # Timestamp ultimo heartbeat

**Heartbeat Dinamico**: Il registro dello stato (`gpu:server:status`) include un array `available_voices` che scansiona in tempo reale la cartella locale `%ARIA_ROOT%\data\voices\`. Questo permette ai client (es. Dashboard DIAS) di mostrare solo le voci effettivamente pronte all'uso senza configurazioni statiche.

# Task scaduti (scritti dal Client Watcher)
gpu:dead:{client_id}:{job_id}         # Hash con motivo scadenza

# Metriche (opzionale, per monitoring)
gpu:metrics:completed_count           # Contatore totale task completati
gpu:metrics:avg_duration:{model_id}   # Media mobile durata per modello
```

---

## 4. Modelli Supportati e Backend

### Architettura backend

Ogni backend implementa l'interfaccia `BaseBackend`:

```python
class BaseBackend(ABC):
    model_id: str
    model_type: str

    @abstractmethod
    def load(self, model_path: str, config: dict) -> None:
        """Carica il modello in VRAM. Chiamato da VRAMManager."""

    @abstractmethod
    def unload(self) -> None:
        """Scarica il modello. Libera VRAM completamente."""

    @abstractmethod
    def run(self, payload: dict) -> dict:
        """Esegue inferenza. Input e output sono dict validati dallo schema."""

    @abstractmethod
    def estimated_vram_gb(self) -> float:
        """VRAM stimata per questo modello. Usata da VRAMManager pre-load."""

    def is_loaded(self) -> bool:
        return self._model is not None
```

### Tabella modelli supportati

| model_type | model_id (esempi) | VRAM est. | Framework | Output |
|---|---|---|---|---|
| `tts` | `fish-s1-mini` | 4GB | fish-speech (nativo) | WAV mono 44.1kHz |
| `tts` | `qwen3-tts-1.7b` | 5GB | transformers (nativo) | WAV mono 24kHz |
| `tts` | `qwen3-tts-custom` | 6GB | transformers (nativo) | WAV mono 24kHz |
| `llm` | `qwen3.5-35b-moe-q3ks` | 13GB (q3) | llama-server.exe | text (thinking) |
| `llm` | `gemini-flash-lite` | Cloud | Google Gateway | text |
| `vision` | `qwen-vl-7b` | 8GB | transformers | text |
| `stt` | `whisper-large-v3` | 3GB | faster-whisper | text + timestamps |

### Nota su VRAM e coesistenza

Con 16GB VRAM, non si possono caricare due modelli grandi contemporaneamente.
Il VRAMManager verifica sempre `estimated_vram_gb()` prima di caricare.
Se VRAM insufficiente: unload del modello corrente prima di caricare il nuovo.

L'unica eccezione: modelli molto piccoli (Kokoro 2GB + Whisper 3GB = 5GB)
potrebbero coesistere. Questa ottimizzazione è futura — il comportamento
di default è sempre "un modello alla volta".

> ⚠️ **Limiti Hardware Voice Cloning (Fish S1-mini + VQGAN) su RTX 5000 (`sm_120`)**
> Su macchine Windows dotate di GPU architettura `sm_120` (es. RTX 5060 Ti), le librerie PyTorch precompilate (versioni standard <2.7 pip/conda) crasheranno sul modulo VQGAN fallendo il lookup del kernel CUDA, rendendo obbligatorio il fallback in CPU.
> **La soluzione approvata per mantenere l'accelerazione CUDA** consiste nell'installare esplicitamente PyTorch stabile 2.7+ e i pacchetti collegati puntando all'indice PyTorch CUDA 12.8 (`--index-url https://download.pytorch.org/whl/cu128`). Questa soluzione è stata verificata su entrambi gli ambienti Fish (`fish-speech` e `fish-voice-cloning`) ed è il riferimento per tutti i nuovi backend.

---

## 4b. Backend Inventory — Stato Reale di Deploy

Questa sezione documenta lo **stato operativo effettivo** dei backend sul PC Gaming.
Per i dettagli tecnici di ogni backend, consultare il documento dedicato.
Per la guida completa agli ambienti Python, vedere `docs/environments-setup.md`.

### Architettura Ambienti (PC Gaming — Marzo 2026)

**Filosofia**: Miniconda globale (`%MINICONDA_ROOT%`) come gestore +
ambienti Python isolati project-local (`%ARIA_ROOT%\envs\`).
Per la tabella variabili e i valori concreti, vedere `docs/environments-setup.md`.

```
C:\Users\%USERNAME%\
├── miniconda3\                    ← Python "base" per l'Orchestratore
│   └── python.exe                 (pystray, redis, PIL — niente AI)
└── aria\envs\
    ├── qwen3tts\                  ← Python 3.12 + PyTorch + qwen-tts
    │   └── python.exe
    └── fish-speech-env\           ← Python 3.10 + PyTorch + fish-speech (da ricreare)
        └── python.exe
```

| Ambiente | Path | Python | Backend | Porta | Stato | VRAM |
|---|---|---|---|---|---|---|
| Orchestratore | `miniconda3\` | 3.12 | main_tray.py | — | 🔄 Da installare | — |
| Qwen3-TTS | `aria\envs\qwen3tts\` | 3.12 | qwen3_server.py | 8083 | ✅ Operativo | ~4-5 GB |
| Fish S1-mini | `aria\envs\fish-speech-env\` | 3.10 | tools/api_server.py | 8080 | 🔄 Da ricreare | ~3-4 GB |
| Voice Cloning | `aria\envs\fish-speech-env\` | 3.10 | voice_cloning_server.py | 8081 | 🔄 Da ricreare | CPU |
| LLM (futuro) | `aria\envs\llm\` | 3.11 | llm_server.py | 8085 | 🔲 In sviluppo | ~5 GB |

### Tabella Backend per Tipo

| model_type | model_id | Backend Class | Ambiente | Documento | Stato |
|---|---|---|---|---|---|
| `tts` | `qwen3-tts-1.7b` | `Qwen3TTSBackend` | `envs/qwen3tts` | `docs/qwen3-tts-backend.md` | ✅ |
| `tts` | `fish-s1-mini` | `FishTTSBackend` | `envs/fish-speech-env` | `docs/fish-tts-backend.md` | 🔄 |
| `tts` | `voice-cloning` | (companion Fish) | `envs/fish-speech-env` | `docs/fish-tts-backend.md` | 🔄 |
| `llm` | `llama-3.1-8b` | `LLMBackend` | `envs/llm` | `docs/llm-backend.md` | 🔲 |
| `music` | `musicgen-small` | — | da creare | — | 🔲 Futuro |
| `stt` | `whisper-large-v3` | — | da creare | — | 🔲 Futuro |

### Pattern comune a tutti i backend

Tutti i backend seguono il pattern **External HTTP Backend** con **avvio on-demand**:
- Processo Python standalone nel suo ambiente project-local
- Avviato dall'Orchestratore (`ModelProcessManager`) quando arriva un task
- Espone un'API HTTP su una porta dedicata
- Spento automaticamente dopo 45 min di inattività (`IDLE_TIMEOUT_S`)
- Apre una finestra CMD visibile con log in tempo reale

L'eccezione sono i backend futuri basati su `diffusers` (image) che potrebbero
essere integrati direttamente nel Node Controller se l'ambiente conda lo permette.

---

## 5. Schema Task — Specifiche Complete

### Task in ingresso (Client → Redis → Server)

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "client_id": "dias-minipc",
  "client_ip": "192.168.1.10",
  "client_version": "1.0.0",

  "model_type": "tts",
  "model_id": "orpheus-3b",

  "queued_at": "2026-02-20T10:00:00Z",
  "priority": 1,
  "timeout_seconds": 1800,

  "callback_key": "gpu:result:dias-minipc:550e8400-...",

  "file_refs": {
    "input": [
      {
        "ref_id": "voice_sample",
        "local_path": "%ARIA_ROOT%\\data\\voices\\narrator.wav",
        "size_bytes": 441000
      }
    ],
    "output": [
      {
        "ref_id": "audio_output",
        "expected_filename": "{job_id}.wav",
        "server_delivery": "http"
      }
    ]
  },

  "payload": {
    // Specifico per model_type — vedi sezione payload per tipo
  }
}
```

**Campi obbligatori**: `job_id`, `client_id`, `model_type`, `model_id`,
`queued_at`, `timeout_seconds`, `callback_key`, `payload`

**Campi opzionali**: `client_ip`, `client_version`, `file_refs`, `priority`

**Priority**: 1=normale (default), 2=alta, 3=urgente.
Il BatchOptimizer considera la priorità all'interno della stessa coda modello.

### Payload per tipo modello

**TTS (model_type: tts)**:
L'interfaccia preferita è quella **basata su intenti** (Voice ID). ARIA 
risolve i parametri dalla sua libreria interna.

```json
{
  "text": "(serious) Il cammino dell'uomo timorato...",
  "voice_id": "narratore",       // Intent: ARIA risolve wav e txt locali
  "pace_factor": 1.0,
  "output_format": "wav"
}
```

*Nota Architetturale sulle Pause*: ARIA si occupa unicamente di generare l'audio "raw". La gestione delle pause drammatiche e della spaziatura strutturale tra le battute/micro-scene è delegata in toto al client (es. DIAS Stage F Audio Mixer) tramite l'assemblaggio dei WAV. ARIA non inietta silenzi artificiali per rispettare il principio di disaccoppiamento.

*Nota: La versione legacy con `file_refs` e `voice_sample_ref` resta supportata 
solo per campioni temporanei "one-shot" non presenti in libreria.*

**Music Generation (model_type: music)**:
```json
{
  "prompt": "Gregorian choir ambient, cold stone, minor key, 70bpm",
  "duration_seconds": 147,
  "output_ref": "audio_output",
  "output_format": "wav",
  "sample_rate": 48000,
  "channels": 2,
  "loop_seamless": true
}
```

**LLM Testuale (model_type: llm)**:
ARIA può risolvere il modello e il sistema di prompt tramite un ID intento.

```json
{
  "intent": "scene_director",    // Resolves to specific model and system prompt
  "messages": [
    {"role": "user", "content": "Testo del capitolo da analizzare..."}
  ],
  "max_tokens": 1000,
  "temperature": 0.2
}
```

**Image Generation (model_type: image)**:
```json
{
  "theme": "monastic_dark",      // ARIA resolves style, negative prompt, etc.
  "prompt": "A dark monastery library, candlelight",
  "width": 1024,
  "height": 1024
}
```

**Speech-to-Text (model_type: stt)**:
```json
{
  "audio_url": "http://...",     // O ID asset registrato
  "language": "it"
}
```

### Risultato (Server → Redis → Client)

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "client_id": "dias-minipc",
  "model_type": "tts",
  "model_id": "orpheus-3b",

  "status": "done",
  "completed_at": "2026-02-20T10:02:30Z",
  "duration_seconds": 142.5,
  "processing_time_seconds": 68.2,

  "output": {
    // Specifico per tipo modello
    "audio_ref": "audio_output",
    "duration_seconds": 142.5,
    "sample_rate": 48000
  },

  "error": null,
  "error_code": null,
  "retry_count": 0
}
```

**status** può essere: `done` | `error` | `timeout` | `cancelled`

**error_code** in caso di errore: `OOM` | `MODEL_LOAD_FAILED` |
`INFERENCE_FAILED` | `INVALID_PAYLOAD` | `FILE_NOT_FOUND` | `TIMEOUT`

---

## 6. Ciclo di Vita di un Task

```
                    ┌─────────┐
                    │ CREATED │  client.submit_task()
                    └────┬────┘
                         │ LPUSH su gpu:queue:{type}:{model}
                    ┌────▼────┐
                    │ QUEUED  │  task in coda Redis, job_id al client
                    └────┬────┘
         semaforo green  │  BatchOptimizer sceglie questo modello
                    ┌────▼──────────┐
                    │  PROCESSING   │  task in gpu:processing:{job_id}
                    └────┬──────────┘
              ┌──────────┴──────────┐
         successo                 errore
              │                     │
        ┌─────▼────┐          ┌─────▼──────┐
        │   DONE   │          │   FAILED   │
        └─────┬────┘          └─────┬──────┘
              │                     │ retry < max_retries?
    risultato su Redis         ┌────▼────┐
    TTL result_ttl_s           │  RETRY  │──► torna a QUEUED
              │                └─────────┘
         ┌────▼────┐                │ retry >= max_retries
         │  READ   │          ┌─────▼──────────┐
         └────┬────┘          │  DEAD LETTER   │
              │               └────────────────┘
         ┌────▼────────┐
         │  CONSUMED   │  client ha letto il risultato
         └─────────────┘
```

### Transizioni speciali

**QUEUED → DEAD LETTER (timeout)**:
Il Client Watcher controlla periodicamente i task in stato QUEUED.
Se `now - queued_at > timeout_seconds` → sposta in dead letter.
Ragione: il PC è probabilmente spento da troppo tempo e il task non ha
più senso (es. DIAS ha già fallito su quel capitolo per altri motivi).

**PROCESSING → QUEUED (crash recovery)**:
All'avvio di ARIA Server, controlla `gpu:processing:*`.
Se trova task in processing (da un crash precedente) → li rimette in coda.
Questo garantisce zero perdita di task su crash server.

**QUEUED → QUEUED (semaforo red)**:
Il task resta in coda. Il Server non consuma. Nessuna transizione di stato.
Il task aspetta semaforo green — anche per giorni se necessario.

---

## 7. Gestione Semaforo e Disponibilità GPU

### Stati del semaforo

```
GREEN  → Server attivo, GPU disponibile per inferenza
RED    → Server attivo, GPU riservata all'utente (gioco, lavoro)
BUSY   → Server sta eseguendo un task (transizione automatica)
OFFLINE → Server non raggiungibile (heartbeat mancante da >30s)
```

### Chi imposta cosa

**Server imposta automaticamente**:
- `BUSY` quando inizia un task
- `GREEN` quando termina un task (se non era stato impostato RED manualmente)
- Scrive `heartbeat` ogni 10 secondi

**Utente imposta manualmente**:
- `RED` tramite tray icon Windows o chiamata API
- `GREEN` per riprendere dopo pausa manuale

**Client legge ma non scrive** il semaforo (sicurezza: solo il server
e l'utente locale controllano la GPU).

### Comportamento con semaforo RED

```
Server:
  - Finisce il task corrente (non interrompe l'inferenza a metà)
  - Non consuma nuovi task dalla coda
  - Mantiene il modello in VRAM (non fa unload — potrebbe tornare GREEN presto)
  - Continua a scrivere heartbeat (è online, solo in pausa)
  - Se RED dura >30 min: fa unload modello per liberare VRAM

Client:
  - submit_task() funziona normalmente — task va in coda
  - is_available() ritorna False
  - I task si accumulano in coda silenziosamente
  - Il Watcher continua a fare polling (i task completati prima del RED
    potrebbero ancora arrivare)
  - Nessun timeout prematuro — il timeout del task è relativo alla creazione,
    non all'inizio dell'esecuzione
```

### Comportamento con server OFFLINE

```
Server:
  - Non c'è (PC spento o crash)
  - Nessun heartbeat scritto

Client:
  - submit_task() funziona — task va in coda (Redis è ancora attivo sul minipc)
  - is_available() ritorna False
  - server_status() ritorna {"status": "offline", "last_seen": "..."}
  - I task si accumulano silenziosamente
  - Dead Letter Handler monitora timeout — se task troppo vecchio → dead letter
```

---

## 8. Scenari di Utilizzo e Comportamento

### Scenario A — Flusso normale, server disponibile

```
t=0s   DIAS Stage C: submit_task(tts, orpheus, payload_scena_1) → job_id_1
t=0.1s DIAS Stage C: submit_task(tts, orpheus, payload_scena_2) → job_id_2
t=0.2s DIAS Stage C: submit_task(tts, orpheus, payload_scena_3) → job_id_3

t=1s   ARIA Server: vede 3 task in gpu:queue:tts:orpheus
       BatchOptimizer: carica Orpheus (già in VRAM → skip load)
       Esegue scena_1...

t=70s  ARIA Server: scena_1 completata
       Scrive gpu:result:dias-minipc:job_id_1
       Watcher minipc: trova risultato → aggiorna dias:state:scene:001

t=140s ARIA Server: scena_2 completata
       Watcher: aggiorna dias:state:scene:002
       DIAS Pipeline: capitolo ha tutte voci → avvia Stage E (music)
```

### Scenario B — Utente inizia a giocare durante elaborazione

```
t=0s   ARIA Server: sta eseguendo scena_4 (task in PROCESSING)
t=30s  Utente: click su tray icon → "GPU Occupata"
       Semaforo → RED
t=95s  ARIA Server: termina scena_4 (non interrompe)
       Scrive risultato, poi vede semaforo RED
       Non consuma scena_5 dalla coda
       Mantiene Orpheus in VRAM

t+2h   Utente: finisce di giocare → "GPU Disponibile"
       Semaforo → GREEN
t+2h+1s ARIA Server: vede GREEN, riprende da scena_5
        Esegue scena_5, scena_6, ... senza perdere nulla
```

### Scenario C — PC gaming spento, minipc invia task

```
t=0s   DIAS Stage C: submit_task(tts, orpheus, payload) → job_id_A
       Task in gpu:queue:tts:orpheus — Redis scrive OK
       job_id_A ritorna al client immediatamente

t=0s..t+8h  PC gaming spento, nessun heartbeat
            Watcher minipc: server OFFLINE, log warning ogni 60s
            Task in coda, nessun timeout (timeout_seconds=1800 non scattato)

t+8h   PC gaming si accende, ARIA Server avvia
       Startup: controlla gpu:processing:* → niente (nessun crash)
       BatchOptimizer: vede task in gpu:queue:tts:orpheus
       Carica Orpheus, esegue task
       Scrive risultato

t+8h+70s  Watcher minipc: trova gpu:result:dias-minipc:job_id_A
          Aggiorna stato DIAS, pipeline continua
```

### Scenario D — Task con timeout scaduto

```
timeout_seconds del task: 3600 (1 ora)
t=0s   submit_task() con queued_at=now

t+1h   Dead Letter Handler (minipc): controlla pending tasks
       now - queued_at = 3600s = timeout_seconds
       Task non ancora risultato → sposta in gpu:dead:dias-minipc:job_id
       Motivo: "TIMEOUT_QUEUED — server non disponibile per 1h"

       DIAS Pipeline: riceve notifica da Watcher → scena marcata come fallita
       Brain Coordinator: decide se ritentare o skippare la scena
```

### Scenario E — Due client simultanei (DIAS + laptop)

```
t=0s   DIAS (minipc): submit_task(tts, orpheus, ...) → job_id_DIAS
t=1s   Laptop: submit_task(llm, llama-3.1-8b, ...) → job_id_LAPTOP

       Redis:
         gpu:queue:tts:orpheus  → [task_DIAS]
         gpu:queue:llm:llama    → [task_LAPTOP]

t=2s   ARIA Server BatchOptimizer:
         tts:orpheus ha 1 task, llm:llama ha 1 task
         Orpheus già in VRAM → esegue TTS prima

t=72s  ARIA Server: TTS completato
         Scrive gpu:result:dias-minipc:job_id_DIAS
         DIAS Watcher: trova risultato → aggiorna pipeline

       ARIA Server: unload Orpheus, load Llama
       Esegue task laptop

t=95s  ARIA Server: LLM completato
         Scrive gpu:result:laptop-client:job_id_LAPTOP
         Laptop Watcher: trova risultato
```

**Nota**: i risultati sono in chiavi separate per `client_id`. Il DIAS Watcher
non vede mai i risultati del laptop e viceversa.

### Scenario F — OOM durante inferenza

```
t=0s   ARIA Server: carica Flux-dev (12GB stima)
       VRAM disponibile: 14GB → OK, carica
       Inizia inferenza immagine 2048x2048

t=5s   torch.cuda.OutOfMemoryError
       Backend: cattura eccezione
       Scrive risultato con status="error", error_code="OOM"
       VRAMManager: torch.cuda.empty_cache() + gc.collect()

       ARIA Server: legge config max_retries=2
       Rimette task in coda con payload modificato:
         steps: 30 → 20 (riduzione qualità)
         width/height: 2048 → 1024 (riduzione risoluzione)
         retry_count: 1
       Riprende dal prossimo task in coda
```

---

## 9. Gestione File Binari (Nuova Architettura HTTP Content Server)

### Il problema originale

Redis è ottimizzato per dati piccoli (chiavi, metadati, JSON).
Inizialmente, ARIA utilizzava una condivisione Samba Linux montata su Windows, ma questo approccio generava gravi problemi di permessi (UNC paths, `WinError 3`, `PermissionError`) e un inutile collo di bottiglia di I/O. 

**Soluzione Architetturale: L'HTTP Asset Server Isolato**
L'architettura aggiornata implementa un sistema self-contained ("On-Demand"):
1. **Conservazione Locale:** Il nodo GPU (`aria_node_controller.py`) salva i risultati pesanti generati sul proprio SSD NVMe locale ad altissima velocità, all'interno della sua cartella di competenza isolata (`%ARIA_ROOT%\data\outputs`).
2. **Accessibilità in Rete:** Il Controller avvia un micro-server HTTP nativo sulla porta 8082, garantendo l'accesso via link simbolico/URL all'interno della rete locale.
3. **Payload a URL:** Invece di far combattere il Client LXC con path condivisi remoti Unix-To-Windows, Redis riceverà in output il direct link: `{"output_url": "http://192.168.1.139:8082/...wav"}` che il Client potrà scaricare o trasmettere nativamente.

Per i file immessi dall'utente per task specifici (come referenze audio per Voice Cloning), i path andranno definiti direttamente come `local_path` residenti nella radice del progetto host per la GPU (`%ARIA_ROOT%\data\voices`).

1.  **Storage Locale su Windows**: Quando l'Orchestratore ARIA su Windows completa un'inferenza, salva l'asset finale direttamente nel suo file system locale (es. `%ARIA_ROOT%\outputs\`). Non ci sono più Virtual Drives, permessi incrociati o condivisioni di cartelle.
2.  **HTTP Static Server**: Lo stesso Orchestratore Windows solleva un leggerissimo server HTTP locale (es. sulla porta `8082`) dedicato esclusivamente a servire in *sola lettura* la cartella `outputs`.

```
GAMING PC (Windows 11): 
├── Esegue inferenza GPU
├── Salva fisicamente su %ARIA_ROOT%\outputs\test-book-123.wav
└── Serve all'IP http://192.168.1.139:8082/outputs/test-book-123.wav

MINIPC (LXC 120 / DIAS):
└── Consuma la URL via Redis e la passa al layer applicativo / browser client.
```

### Path nei task e Payload di Ritorno

L'orchestratore non risponde più passando un `path` Unix o Windows nei risultati Redis, ma una URL universale assoluta. Questo slega completamente i client (DIAS, WebApp, app mobile) dalla conoscenza della topologia del file system.

```json
{
  "job_id": "test-book-123",
  "client_id": "dias-minipc",
  "status": "done",
  "output": {
    "audio_url": "http://192.168.1.139:8082/outputs/test-book-123.wav",
    "duration_seconds": 142.5
  }
}
```

### Gestione degli Input (File References)

Per i file necessari in ingresso all'inferenza (ad esempio i campioni voce per il Voice Cloning), l'architettura supporta due approcci:
1.  **Trasferimento Inline Base64**: Per file piccoli (come reference audio di 10 secondi), il payload inietta direttamente i token convertiti in base64 string.
2.  **URL HTTP Retrieval**: Per file più grandi, DIAS fornirà una URL esposta dal proprio content-server e l'Orchestratore Windows si occuperà di scaricarla localmente prima dell'inferenza. Niente più passaggi di `shared_path`.

---

## 10. Multi-Client: Routing e Isolamento

### Identificazione client

Ogni istanza di ARIAClient ha un `client_id` configurato dall'utente:

```python
# Su DIAS minipc
client = ARIAClient(redis_host="192.168.1.10", client_id="dias-minipc")

# Su laptop personale
client = ARIAClient(redis_host="192.168.1.10", client_id="laptop-personal")

# Su secondo minipc (futuro)
client = ARIAClient(redis_host="192.168.1.10", client_id="minipc-2")
```

Il `client_id` deve essere unico sulla rete. Non c'è un registro centrale —
è responsabilità dell'utente configurarlo correttamente. Se due client usano
lo stesso `client_id`, i risultati potrebbero essere letti dal client sbagliato.

### Isolamento dei risultati

I risultati sono in chiavi separate per client:
```
gpu:result:dias-minipc:job_id_1      ← solo DIAS legge questa
gpu:result:laptop-personal:job_id_2  ← solo laptop legge questa
```

Il Watcher di ogni client fa GET solo sulle chiavi con il suo `client_id`.
Non è necessario autenticazione per questo — l'isolamento è per convenzione
di naming, non per sicurezza crittografica (siamo su LAN privata).

### Priorità tra client

Non esiste priorità tra client diversi — tutti competono sulla stessa coda
per tipo modello. Se DIAS e il laptop inviano task TTS contemporaneamente,
vengono eseguiti in ordine FIFO (salvo campo `priority` del task).

Se in futuro si vuole dare priorità a un client specifico, si possono
creare code separate per priorità:
```
gpu:queue:tts:orpheus:high    # priority 3
gpu:queue:tts:orpheus:normal  # priority 1-2
```
Il BatchOptimizer drena prima la coda high, poi normal. Questa è
un'ottimizzazione futura — non necessaria per la v1.

---

## 11. API HTTP di ARIA Server

L'API HTTP è **minimale per design**. Non gestisce task — quelli passano
sempre via Redis. L'API serve solo per:
- Controllo semaforo
- Monitoring stato
- Health check da script e dashboard

**Base URL**: `http://{gaming_pc_ip}:7860`

### Endpoints

```
GET  /health
     → 200 OK {"status": "ok", "uptime_seconds": 3642}
     → Risponde sempre, anche con semaforo RED

GET  /status
     → {
         "semaphore": "green",
         "loaded_model": {"type": "tts", "id": "orpheus-3b"},
         "vram": {"used_gb": 7.2, "free_gb": 8.8, "total_gb": 16.0},
         "queues": {
           "tts:orpheus-3b": 3,
           "music:musicgen-small": 1,
           "llm:llama-3.1-8b": 0
         },
         "current_task": {"job_id": "...", "client_id": "...", "started_at": "..."},
         "stats": {
           "completed_today": 47,
           "avg_duration_tts_s": 68.2
         }
       }

POST /semaphore
     Body: {"state": "green" | "red"}
     Headers: X-API-Key: {api_key}
     → 200 OK {"previous": "green", "current": "red"}
     → 401 Unauthorized se API key errata

GET  /queue/{model_type}/{model_id}
     → {"length": 3, "oldest_task_age_seconds": 142}

DELETE /queue/{model_type}/{model_id}/{job_id}
     Headers: X-API-Key: {api_key}
     → 200 OK {"cancelled": true}
     → 404 se task non trovato in coda (potrebbe essere già in processing)

GET  /models
     → Lista modelli configurati con stato (loaded/unloaded) e VRAM
```

---

## 12. Configurazione

### config.yaml — ARIA Server

```yaml
aria:
  version: "1.0.0"
  server_id: "gaming-pc-main"

redis:
  host: "192.168.1.10"          # IP minipc
  port: 6379
  password: ""                   # consigliato impostare in produzione
  db: 0
  reconnect_interval_seconds: 5
  max_reconnect_attempts: 0      # 0 = infinito

api:
  host: "0.0.0.0"
  port: 7860
  api_key: "cambia_questa_chiave"

broker:
  poll_interval_seconds: 2
  result_ttl_seconds: 86400      # 24h
  batch_wait_seconds: 5          # attesa prima di caricare modello
  heartbeat_interval_seconds: 10
  processing_timeout_seconds: 3600  # task in processing da >1h = crash recovery

semaphore:
  default_state: "green"
  red_vram_unload_after_minutes: 30

file_sharing:
  server_base_path: "%ARIA_ROOT%\\data\\outputs"
  http_asset_server_port: 8082
  inline_max_bytes: 5242880      # 5MB

models:
  tts:
    fish-s1-mini:
      enabled: true
      model_path: "aria/data/models/fish-s1-mini"
      estimated_vram_gb: 4.0
      max_retries: 2
    orpheus-3b:
      enabled: false
      model_path: "aria/data/models/orpheus-3b-q4"
      estimated_vram_gb: 7.0
      max_retries: 2
  music:
    musicgen-small:
      enabled: false
      model_path: "aria/data/models/musicgen-small"
      estimated_vram_gb: 4.0
      max_retries: 1
  llm:
    llama-3.1-8b:
      enabled: false
      model_path: "aria/data/models/llama-3.1-8b-q4"
      estimated_vram_gb: 5.5
      max_retries: 1

logging:
  level: "INFO"
  file: "C:/logs/aria-server/aria.log"
  rotation: "daily"
  retention_days: 7
```

### Configurazione ARIAClient (Python)

```python
# Minima (obbligatoria)
client = ARIAClient(
    redis_host="192.168.1.10",
    client_id="dias-minipc"
)

# Completa (con tutti i parametri)
client = ARIAClient(
    redis_host="192.168.1.10",
    redis_port=6379,
    redis_password="",
    client_id="dias-minipc",
    default_timeout_seconds=1800,
    default_priority=1,
    result_poll_interval_seconds=5,
    shared_path_local="/mnt/aria-shared",
    api_base_url="http://192.168.1.20:7860",  # opzionale, solo per semaforo
    api_key="cambia_questa_chiave"             # opzionale
)
```

---

## 13. Sicurezza

### Modello di sicurezza

ARIA è progettato per **reti locali fidate** (casa, piccolo ufficio).
Non è progettato per essere esposto su internet. Il modello di sicurezza
è proporzionale a questo contesto.

### Misure implementate

**API Key per operazioni privilegiate**: cambiare semaforo, cancellare task,
ricaricare modelli. Non richiesta per lettura stato e submit task (Redis gestisce già questo).

**Redis senza accesso esterno**: il firewall del minipc deve bloccare
la porta 6379 per IP esterni alla LAN. Configurazione Redis:
`bind 127.0.0.1 192.168.1.10` (solo localhost e LAN).

**Validazione payload**: ogni task è validato contro lo schema prima
di essere accodato. Payload malformati sono rifiutati con log.

**client_id non autenticato**: su LAN privata, l'isolamento per client_id
è sufficiente. Non serve autenticazione crittografica per il routing risultati.

### Cosa NON è protetto

- Un client malevolo sulla LAN potrebbe leggere task di altri client
  (conosce il naming schema delle chiavi Redis)
- Un client potrebbe inviare task con modelli non configurati (verranno
  rifiutati ma il tentativo non è autenticato)
- L'API HTTP è protetta da API key ma non da TLS (HTTP, non HTTPS)

Queste limitazioni sono accettabili su LAN domestica. Per ambienti con
requisiti di sicurezza maggiori, aggiungere: Redis AUTH, TLS su Redis,
HTTPS sull'API, autenticazione client_id con token.

---

## 14. Limiti e Vincoli Noti

| Limite | Valore | Motivo |
|--------|--------|--------|
| Modelli in VRAM simultanei | 1 (default) | RTX 5060 Ti 16GB |
| Dimensione file inline Redis | 5MB | Performance Redis |
| Task in coda per modello | illimitato | Redis list, no cap |
| Client simultanei | illimitato | isolamento per naming |
| Formati modello supportati | safetensors, pytorch_model.bin | no GGUF nativo v1 |
| Sistema operativo server | Windows 10/11 | CUDA su Windows |
| Sistema operativo client | qualsiasi con Python 3.10+ | solo Redis e stdlib |
| Comunicazione server-client | Redis su LAN | no internet, no VPN |

### Limitazioni v1 da risolvere in futuro

- **No streaming**: i risultati sono restituiti interamente al completamento.
  Per LLM, lo streaming token-by-token richiederebbe un canale dedicato
  (Redis pub/sub o WebSocket). Non implementato in v1.

- **No multi-GPU**: con più GPU il BatchOptimizer dovrebbe gestire
  assegnazione task per GPU. Non necessario con setup attuale.

- **No modelli GGUF**: richiede llama-cpp-python come dipendenza aggiuntiva.
  Aggiungibile come backend opzionale in v2.

- **No priorità inter-client**: tutti i client competono alla pari sulla
  stessa coda FIFO. Priorità intra-task (campo priority) è supportata,
  priorità tra client non lo è.

---

## 15. Evoluzione AI-as-a-Service (SOA)

### Visione: Disaccoppiamento Client-Portante
Nella versione v2.0, ARIA si evolve da semplice "worker di PDF/WAV" a un vero provider di servizi AI. 

1. **Agnosticismo del Client**: Un'app (come DIAS) non deve sapere che ARIA sta usando Fish S1-mini o Llama-3. Deve solo richiedere una "Azione" (es: `generate_speech`) con dei parametri semantici (`voice: "narratore"`).
2. **Internalizzazione degli Asset**: ARIA gestisce un proprio file system strutturato per gli asset:
   - `/aria/data/voices/{voice_id}/ref.wav`
   - `/aria/data/voices/{voice_id}/ref.txt`
   - `/aria/data/prompts/{task_type}/system_prompt.txt`

### Flusso di Risoluzione Interna (ARIA-side)
Quando ARIA riceve un task con intent `voice_id: "narratore"`:
1. **Lookup**: L'Orchestratore consulta la `VoiceLibrary`.
2. **Risoluzione**: Se l'ID esiste, ARIA inietta automaticamente i path locali (`%ARIA_ROOT%\...`) nel payload che invia al backend specifico.
3. **Execution**: Il backend riceve i file già pronti senza che il client remoto (DIAS su Linux) abbia mai saputo della loro esistenza.

### Futuro: Pipeline Interamente Locali
In questa visione, DIAS potrebbe delegare ad ARIA non solo la voce, ma anche la generazione del copione:
- **Task 1 (LLM)**: Client → ARIA (LLM Backend) → JSON Scena.
- **Task 2 (TTS)**: Client → ARIA (TTS Backend) → WAV Scena.
Questo riduce la latenza, azzera la dipendenza da API esterne (Google) e centralizza la "conoscenza" AI in un unico nodo GPU potente.

---

*ARIA Blueprint v2.0 — Marzo 2026*
*Documento di evoluzione architetturale verso AI-as-a-Service*
## 15. Deployment & Bootstrap (NH-Mini Philosophy)

Come risposto alla domanda: "Basta Git per accendere un nuovo PC?", la risposta è **Sì per la logica, No per gli asset**.

### 15.1 Logica (Git / LXC 190)
Il repository Git contiene tutto ciò che definisce il "cervello" e i "muscoli" di ARIA:
- Il codice dei backend (`aria_server/backends/`).
- L'orchestratore del nodo (`aria_node_controller/`).
- Gli script di avvio (`Avvia_Tutti_Server_ARIA.bat`).
- La documentazione e i blueprint.

### 15.2 Assets (Local / Runtime)
Gli asset pesanti e specifici del nodo **non sono in Git** per design. Devono essere replicati o scaricati sul nuovo PC:
1. **Modelle AI (`data/models/`)**: Giga di pesi (GGUF, Safetensors) scaricabili via `aria-download.bat`.
2. **Voice Library (`data/voices/`)**: La libreria delle voci clonate. È un database locale. Si consiglia di sincronizzarla separatamente se si vuole consistenza tra nodi.
3. **Ambienti Python (`envs/` o Conda)**: Vanno ricreati seguendo `SETUP_PC_GAMING.md` per garantire la compatibilità con l'hardware specifico (driver CUDA, architettura sm_xxx).

### 15.3 Piano di "Cold Boot" (Nuovo Nodo)
1. `git clone` del repo.
2. Installazione Conda + PyTorch sm_120 (vedi `SETUP_PC_GAMING.md`).
3. Esecuzione `aria-download.bat` per scaricare `fish-s1-mini`.
4. Copia manuale (o via rete) della cartella `data/voices` se si desidera ereditare le voci esistenti.
5. Avvio via `Avvia_Tutti_Server_ARIA.bat`.
