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
│  LIVELLO 1: Ambiente Backend C (High-Fidelity Sound Engine)  │
│  %ARIA_ROOT%\envs\dias-sound-engine\python.exe               │
│  Python 3.11 + PyTorch 2.11.0+cu128 + sm_120                 │
│  ► Esegue ace_step_server.py (porta 8085)                    │
│  ► Esegue HTDemucs + TorchCodec (Sound Factory v4.5)         │
│  ► ~10-12 GB VRAM quando attivo (ACE-Step XL SFT)            │
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
    ### 1.1 Conda Environments (Worker Node)

| Nome Ambiente | Scopo | Installazione |
|---|---|---|
| `fish-speech-env` | TTS Fish S1-mini | Basato su `torch 2.7+cu128` |
| `nh-qwen35-llm` | LLM locale Qwen 3.5 | `pip install llama-cpp-python` (CUDA) |
| `aria-cloud` | Gateway Cloud Gemini | `pip install google-generativeai` |
| `qwen3tts` | TTS Qwen3-Audio | Ambiente isolato per Transformers |
| `dias-sound-engine` | **Sound Factory v4.5** | `torch 2.11.0+cu128` (Blackwell Native) |
    │   └── sox\                         ← Utilità audio per i backend TTS
    │
    ├── data\
    │   ├── models\
    │   │   ├── fish-s1-mini\            ← Pesi modello Fish (~1.5GB)
    │   │   ├── qwen3-tts-1.7b\         ← Pesi modello Qwen3 (~3.8GB)
    │   │   └── qwen3-tts-1.7b-customvoice ← Pesi modello Qwen3 Custom
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

## 8. Sound Factory v4.5 (dias-sound-engine)

L'ambiente `dias-sound-engine` è il backend di ARIA per la produzione audio di alta fedeltà, ottimizzato per **NVIDIA Blackwell (sm_120)**.

### Specifiche Reali (Verificate Aprile 2026)
- **Python**: 3.11.15
- **PyTorch**: `2.11.0+cu128` (Supporto nativo sm_120)
- **Quantizzazione**: `torchao 0.12.0` (Mandatoria per stabilità XL su 16GB)
- **Audio Decoding**: `torchcodec 0.11.0` (Integrazione con HTDemucs)

### Componenti AI
1. **Generazione Musicale**: ACE-Step 1.5 XL SFT (4B)
   - Richiede `transformers >= 4.55`, `accelerate >= 1.12`.
   - **Blackwell Fidelity**: Su GPU da 16GB (Tier 6a), è possibile far girare il modello in **BF16 nativo** grazie all'offloading.
   - **Quantizzazione (Opzionale)**: `int8_weight_only` è suggerita solo per multitasking o generazioni molto lunghe.

> [!TIP]
> **Tier 6a (16GB)**: La RTX 5060 Ti supporta ACE-Step 1.5 XL in **BF16 nativo**. La quantizzazione è utile per risparmiare VRAM ma non è mandatoria per l'inferenza isolata.

### Guida alla Configurazione Flat (Blackwell 16GB)
Per la massima stabilità su RTX 5060 Ti, creare un file `.toml` con struttura piatta:
```toml
# Esempio blackwell_xl_config.toml
config_path = "data/assets/models/acestep-v15-xl-sft"
quantization = "int8_weight_only"
offload_to_cpu = true
duration = 5.0
# ... altre chiavi non nidificate
```

### 🚀 Blackwell sm_120 Stability Workarounds
Oltre all'ambiente specifico, per modelli DiT XL è necessario:

1.  **INT8 Quantization**: Obbligatoria per rientrare nei 16GB senza crash (riduce DiT da 9GB a 4.5GB).
2.  **Unicode Safety**: Evitare emoji nei log o messaggi di sistema se si esegue su console Windows standard (CP1252) per prevenire `UnicodeEncodeError`.
3.  **Variabili d'Ambiente**:
    - `ACESTEP_COMPILE_MODEL="0"`: Disabilita JIT compilation per evitare errori Triton su Blackwell.
    - `CUDA_MODULE_LOADING="LAZY"`: Riduce la pressione sulla memoria all'avvio.

```powershell
# Esempio di setup stabilità
$env:CUDA_MODULE_LOADING="LAZY"
$env:TORCH_CUDA_ARCH_LIST="12.0"
```


---

## 7. Riproducibilità (Template YAML)

Per ricostruire rapidamente gli ambienti in caso di corruzione o migrazione, utilizzare i file template in `envs/templates/`:

*   **Orchestratore**: `envs/templates/orchestrator.yml`
*   **Qwen3-TTS**: `envs/templates/qwen3-tts.yml`
*   **Fish-Speech**: `envs/templates/fish-speech.yml`

Esempio di utilizzo:
```cmd
conda env create --prefix %ARIA_ROOT%\envs\nuovo_env --file envs\templates\qwen3-tts.yml
```

---

*ARIA Environments Setup — Marzo 2026*
*Documenti correlati: `docs/ARIA-blueprint.md` (sezione 4b), `docs/hybrid-tts-architecture.md`*
