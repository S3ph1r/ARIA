# ARIA — Guida Ambienti Python
## Architettura, Setup e Manutenzione degli Ambienti di Inferenza

> **Aggiornato**: 2026-03-04
> **Riferimento**: ARIA Blueprint v2.0 (`docs/ARIA-blueprint.md`, sezione 4b)

---

## Variabili di Riferimento

Questa guida utilizza variabili generalizzate. Per i valori concreti del deploy
corrente, vedere la sezione [Il Nostro Deploy](#il-nostro-deploy) in fondo.

| Variabile | Significato |
|-----------|-------------|
| `%ARIA_ROOT%` | Directory root del progetto ARIA sul nodo GPU |
| `%MINICONDA_ROOT%` | Installazione Miniconda globale |
| `%REDIS_HOST%` | IP o hostname del server Redis |
| `%GPU_NODE_IP%` | IP del PC Gaming (nodo GPU) |
| `%DEV_HOST_IP%` | IP del dev server (LXC/Linux) |
| `%USERNAME%` | Utente Windows sul nodo GPU |

---

## 1. Filosofia: Miniconda Globale + Ambienti Project-Local

ARIA utilizza una **architettura a 3 livelli Python** che garantisce isolamento
totale tra i backend, portabilità e manutenzione semplice.

### Il principio

```
┌──────────────────────────────────────────────────────────────┐
│  LIVELLO 0: Miniconda Globale                                │
│  %MINICONDA_ROOT%\python.exe                                 │
│  Solo librerie leggere: pystray, redis, PIL                  │
│  ► Esegue l'Orchestratore (main_tray.py)                     │
│  ► NON carica modelli AI in VRAM                             │
├──────────────────────────────────────────────────────────────┤
│  LIVELLO 1: Ambiente Backend A (Qwen3-TTS)                   │
│  %ARIA_ROOT%\envs\qwen3tts\python.exe                        │
│  Python 3.12 + PyTorch 2.6+cu124 + qwen-tts + FastAPI        │
│  ► Esegue qwen3_server.py (porta 8083)                       │
│  ► ~4-5 GB VRAM quando attivo                                │
├──────────────────────────────────────────────────────────────┤
│  LIVELLO 1: Ambiente Backend B (Fish-Speech)                 │
│  %ARIA_ROOT%\envs\fish-speech-env\python.exe                  │
│  Python 3.10 + PyTorch 2.7+cu128 + fish-speech               │
│  ► Esegue tools/api_server.py (porta 8080)                   │
│  ► Esegue voice_cloning_server.py (porta 8081)               │
│  ► ~3-4 GB VRAM quando attivo                                │
└──────────────────────────────────────────────────────────────┘
```

### Perché questa architettura

1. **Isolamento**: ogni `python.exe` cerca le librerie nella propria cartella
   `Lib\site-packages\`. Non serve `conda activate` — basta chiamare
   direttamente il python.exe dell'ambiente desiderato.

2. **Nessun conflitto**: Qwen3 usa PyTorch 2.6+cu124, Fish usa PyTorch 2.7+cu128.
   Versioni diverse coesistono senza problemi perché vivono in cartelle separate.

3. **Portabilità**: tutti gli ambienti risiedono dentro `%ARIA_ROOT%\envs\`,
   rendendo il progetto autocontenuto e spostabile.

4. **Avvio on-demand**: l'Orchestratore (Livello 0) usa `subprocess.Popen()` per
   avviare il `python.exe` specifico del backend necessario, solo quando arriva
   un task dalla coda Redis corrispondente.

---

## 2. Mappa Directory su Nodo GPU

```
C:\Users\%USERNAME%\
├── miniconda3\                          ← Miniconda globale (Livello 0)
│   ├── python.exe
│   ├── conda.exe
│   ├── Lib\site-packages\
│   │   ├── pystray\
│   │   ├── redis\
│   │   ├── PIL\
│   │   └── customtkinter\
│   └── Scripts\
│
└── aria\                                ← %ARIA_ROOT%
    ├── aria_node_controller\            ← Codice sorgente (da dev server)
    │   ├── main_tray.py                 # Entry point Orchestratore
    │   ├── qwen3_server.py              # Server Qwen3 (porta 8083)
    │   ├── core\orchestrator.py         # Loop principale + ModelProcessManager
    │   └── backends\qwen3_tts.py        # Backend HTTP Qwen3
    │
    ├── envs\                            ← Ambienti Python isolati (Livello 1)
    │   ├── qwen3tts\                    ← Creato con: conda create --prefix ... python=3.12
    │   │   ├── python.exe               # IL suo Python 3.12
    │   │   ├── Lib\site-packages\
    │   │   │   ├── torch\               # IL suo PyTorch (2.6+cu124)
    │   │   │   ├── qwen_tts\            # Qwen-TTS
    │   │   │   ├── transformers\
    │   │   │   └── fastapi\
    │   │   └── Scripts\
    │   │
    │   ├── fish-speech\                 ← Clone del REPO (codice sorgente Fish)
    │   │   ├── tools\api_server.py      # Server Fish TTS
    │   │   ├── voice_cloning_server.py  # Server Voice Cloning
    │   │   └── ...
    │   │
    │   └── fish-speech-env\             ← DA CREARE — Ambiente Python Fish
    │       ├── python.exe               # Python 3.10
    │       └── Lib\site-packages\
    │           ├── torch\               # PyTorch 2.7+cu128
    │           ├── fish_speech\
    │           └── torchcodec\
    │
    ├── data\
    │   ├── models\
    │   │   ├── fish-s1-mini\            ← Pesi modello Fish (~1.5GB)
    │   │   └── qwen3-tts-1.7b\         ← Pesi modello Qwen3 (~3.8GB)
    │   ├── voices\
    │   │   ├── angelo\                  ← ref.wav + ref_padded.wav + ref.txt
    │   │   └── luca\
    │   └── outputs\                     ← WAV generati (serviti via HTTP :8082)
    │
    ├── Avvia_Tutti_Server_ARIA.bat       ← Script avvio
    └── node_settings.json                ← Config nodo (Redis host, IP)
```

---

## 3. Come l'Orchestratore Avvia i Backend

L'Orchestratore (`ModelProcessManager` in `orchestrator.py`) non usa mai
`conda activate`. Chiama direttamente il `python.exe` del backend:

```python
def _build_cmd(self, model_id: str) -> list:
    if model_id == "qwen3-tts-1.7b":
        python = str(self.aria_root / "envs" / "qwen3tts" / "python.exe")
        server = self.aria_root / "aria_node_controller" / "qwen3_server.py"
        return [python, str(server)]

    elif model_id == "fish-s1-mini":
        python = str(self.aria_root / "envs" / "fish-speech-env" / "python.exe")
        fish_dir = self.aria_root / "envs" / "fish-speech"
        return [python, str(fish_dir / "tools" / "api_server.py"), ...]

    elif model_id == "voice-cloning":
        python = str(self.aria_root / "envs" / "fish-speech-env" / "python.exe")
        fish_dir = self.aria_root / "envs" / "fish-speech"
        return [python, str(fish_dir / "voice_cloning_server.py")]
```

Il processo viene aperto in una **finestra CMD dedicata** visibile sul desktop:
```python
subprocess.Popen(
    f'start "{title}" cmd.exe /k "{cmd_str}"',
    shell=True, cwd=str(self.aria_root)
)
```

---

## 4. Setup da Zero (Guida Operativa)

### 4.1 Installare Miniconda Globale

```cmd
:: Scaricare l'installer da https://docs.conda.io/en/latest/miniconda.html
:: Installare in %MINICONDA_ROOT% (es. C:\Users\<USERNAME>\miniconda3)
:: Opzioni: solo per l'utente corrente, aggiungere al PATH

:: Verificare
%MINICONDA_ROOT%\python.exe --version
```

### 4.2 Installare dipendenze per l'Orchestratore

```cmd
%MINICONDA_ROOT%\python.exe -m pip install ^
    pystray Pillow redis customtkinter requests python-dotenv
```

### 4.3 Creare un nuovo ambiente backend (esempio Qwen3)

```cmd
:: Creare con --prefix (project-local, non globale)
%MINICONDA_ROOT%\Scripts\conda.exe create ^
    --prefix %ARIA_ROOT%\envs\qwen3tts ^
    python=3.12 -y

:: Installare i pacchetti dentro l'ambiente
%ARIA_ROOT%\envs\qwen3tts\python.exe -m pip install ^
    torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
%ARIA_ROOT%\envs\qwen3tts\python.exe -m pip install ^
    qwen-tts fastapi uvicorn soundfile numpy transformers accelerate
```

### 4.4 Aggiungere un nuovo backend (template)

Per aggiungere un backend completamente nuovo (es. LLM):

1. Creare l'ambiente: `conda create --prefix %ARIA_ROOT%\envs\llm python=3.11 -y`
2. Installare le dipendenze: `%ARIA_ROOT%\envs\llm\python.exe -m pip install ...`
3. Creare il server FastAPI: `aria_node_controller\llm_server.py`
4. Aggiornare `_build_cmd()` in `orchestrator.py` con il nuovo model_id
5. Aggiungere `MODEL_CONFIGS[model_id]` con health_url e startup_wait
6. Testare: `redis-cli LPUSH gpu:queue:llm:llama-3.1-8b '{...}'`

---

## 5. Porte Backend

| Porta | Backend | Protocollo |
|-------|---------|-----------|
| 8080 | Fish S1-mini TTS | HTTP (FastAPI) |
| 8081 | Fish Voice Cloning (VQGAN) | HTTP (FastAPI) |
| 8082 | Asset Server (WAV output) | HTTP (integrato in Orchestratore) |
| 8083 | Qwen3-TTS | HTTP (FastAPI) |
| 8085 | LLM Llama 3.1 8B (futuro) | HTTP (FastAPI) |

---

## 6. Note Tecniche

### CUDA e sm_120 (Blackwell)
La RTX 5060 Ti usa l'architettura sm_120. Le build PyTorch standard (< 2.7)
**non supportano** sm_120. Tutti gli ambienti devono usare:
- `torch >= 2.6.0` con `--index-url https://download.pytorch.org/whl/cu124` (Qwen3)
- `torch >= 2.7.0` con `--index-url https://download.pytorch.org/whl/cu128` (Fish, LLM)

### Perché `--prefix` invece di `--name`
`conda create --name env_name` crea l'ambiente dentro `miniconda3\envs\`.
`conda create --prefix /path/to/env` lo crea dove vuoi tu.
Usiamo `--prefix` per tenere tutto dentro `%ARIA_ROOT%\envs\` — portabilità e ordine.

---

## Il Nostro Deploy

> Valori concreti per il deploy Homelab NH-Mini (Marzo 2026).

| Variabile | Valore |
|-----------|--------|
| `%USERNAME%` | `Roberto` |
| `%ARIA_ROOT%` | `C:\Users\Roberto\aria` |
| `%MINICONDA_ROOT%` | `C:\Users\Roberto\miniconda3` |
| `%REDIS_HOST%` | `192.168.1.10` (CT120 su MiniPC) |
| `%GPU_NODE_IP%` | `192.168.1.139` (PC Gaming "INFINITY", RTX 5060 Ti 16GB) |
| `%DEV_HOST_IP%` | `192.168.1.190` (LXC 190 su MiniPC) |

---

*ARIA Environments Setup — Marzo 2026*
*Documenti correlati: `docs/ARIA-blueprint.md` (sezione 4b), `docs/hybrid-tts-architecture.md`*
