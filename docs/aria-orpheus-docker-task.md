# 🎯 TASK: Build ARIA Server — Docker Image con Orpheus TTS Italiano
## Documento per Agent — Trae / Kimi K2

---

## CONTESTO

Stiamo costruendo **ARIA Server**, un servizio di inferenza GPU locale che gira su un PC gaming Windows con RTX 5060 Ti (16GB VRAM, architettura Blackwell sm_120, CUDA 12.8).

ARIA Server è un broker agnostico che riceve task da client sulla LAN via Redis e li esegue sulla GPU. Il primo backend da implementare è **Orpheus TTS italiano**.

Il progetto ARIA usa **Docker Desktop su Windows** con GPU passthrough (nvidia-container-toolkit). Redis gira sul minipc sempre acceso ed è raggiungibile via LAN.

---

## PROBLEMA RISOLTO (non riaprire)

**vLLM non è compatibile con RTX 5060 Ti / Blackwell sm_120** in nessuna versione precompilata. Non esiste PyTorch 2.8.x+cu128, creando un deadlock di dipendenze. Anche con build da sorgente, vLLM causa word-skipping su Orpheus nel 60-70% dei casi.

**Soluzione adottata**: `llama.cpp server` come backend di inferenza + `Orpheus-FastAPI` come layer TTS. Stack collaudato, stabile, zero problemi di dipendenze CUDA su Blackwell.

---

## STACK TECNOLOGICO SCELTO

```
┌─────────────────────────────────────────────────┐
│              ARIA SERVER (Docker)               │
│                                                 │
│  Container 1: llama-cpp-server                  │
│  ├── Base: nvidia/cuda:12.8.0-runtime-ubuntu22  │
│  ├── llama.cpp compilato per sm_120             │
│  ├── Modello: Orpheus-3b-Italian_Spanish GGUF   │
│  └── Porta: 5006 (interna)                      │
│                                                 │
│  Container 2: orpheus-fastapi                   │
│  ├── Base: python:3.11-slim                     │
│  ├── Orpheus-FastAPI (Lex-au/Orpheus-FastAPI)   │
│  ├── Connette a llama-cpp-server:5006           │
│  └── Porta: 5005 (esposta sulla LAN)            │
│                                                 │
│  Container 3: aria-broker (DA SVILUPPARE DOPO)  │
│  ├── Redis client → legge code LAN              │
│  ├── Chiama orpheus-fastapi:5005                │
│  └── Scrive risultati su Redis LAN              │
└─────────────────────────────────────────────────┘

RETE LAN:
  Redis: minipc:6379 (già attivo, non toccare)
  Orpheus API: gaming-pc:5005 (esposta dopo questo task)
  ARIA Broker API: gaming-pc:7860 (fase successiva)
```

---

## OBIETTIVO DI QUESTO TASK

Creare l'immagine Docker per i **primi due container** (llama-cpp-server + orpheus-fastapi) funzionanti su RTX 5060 Ti con modello italiano.

Il Container 3 (aria-broker) è fuori scope per ora — verrà sviluppato nel task successivo.

**Al termine di questo task deve funzionare:**
```bash
# Su gaming PC Windows
docker compose -f docker-compose-gpu.yml up

# Test da qualsiasi device sulla LAN:
curl http://192.168.x.x:5005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"orpheus","input":"pietro: Ciao, sono una voce italiana generata da Orpheus.","voice":"pietro","response_format":"wav"}' \
  --output test_italiano.wav
```

---

## RIFERIMENTO PRINCIPALE

Repo base da cui partire: **https://github.com/Lex-au/Orpheus-FastAPI**

Questo repo ha già:
- `Dockerfile.gpu` per orpheus-fastapi ✅
- `docker-compose-gpu.yml` funzionante ✅
- Supporto modello italiano con voci `pietro`, `giulia`, `carlo` ✅
- API OpenAI-compatible su `/v1/audio/speech` ✅
- Batching automatico per testi lunghi + crossfade ✅

**Il problema**: il `docker-compose-gpu.yml` originale usa llama.cpp con un'immagine base generica non ottimizzata per Blackwell sm_120. Devi adattarlo per RTX 5060 Ti.

---

## OPERAZIONI RICHIESTE

### Step 1 — Clona il repo base

```bash
git clone https://github.com/Lex-au/Orpheus-FastAPI.git aria-server
cd aria-server
```

### Step 2 — Crea `.env` per il modello italiano

Crea `.env` partendo da `.env.example`:

```env
# Modello italiano/spagnolo ottimizzato
ORPHEUS_MODEL_NAME=Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf

# HuggingFace repo del modello italiano
ORPHEUS_MODEL_REPO=lex-au/Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf

# URL interno al llama-cpp-server (nome servizio Docker)
ORPHEUS_API_URL=http://llama-cpp-server:5006/v1/completions

# Parametri generazione
ORPHEUS_API_TIMEOUT=300
ORPHEUS_MAX_TOKENS=8192
ORPHEUS_TEMPERATURE=0.6
ORPHEUS_TOP_P=0.9
ORPHEUS_SAMPLE_RATE=24000

# Server FastAPI
ORPHEUS_PORT=5005
ORPHEUS_HOST=0.0.0.0
```

### Step 3 — Crea Dockerfile ottimizzato per llama.cpp + Blackwell

Crea `Dockerfile.llama-blackwell`:

```dockerfile
# Base CUDA 12.8 — unica versione compatibile con RTX 5060 Ti (sm_120)
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_DOCKER_ARCH=sm_120
ENV GGML_CUDA=1

# Dipendenze build
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libcurl4-openssl-dev \
    wget \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp da sorgente con supporto sm_120
RUN git clone https://github.com/ggerganov/llama.cpp /llama.cpp && \
    cd /llama.cpp && \
    cmake -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="120" \
      -DGGML_CUDA_F16=ON \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j$(nproc) --target llama-server

# Directory modelli (montata come volume)
RUN mkdir -p /models

WORKDIR /llama.cpp

# Script di avvio
COPY start-llama-server.sh /start-llama-server.sh
RUN chmod +x /start-llama-server.sh

EXPOSE 5006

CMD ["/start-llama-server.sh"]
```

Crea `start-llama-server.sh`:

```bash
#!/bin/bash
set -e

MODEL_PATH="/models/${ORPHEUS_MODEL_NAME}"

# Verifica che il modello esista
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ ERRORE: Modello non trovato in $MODEL_PATH"
    echo "Scarica il modello prima di avviare il container."
    echo "Comando: aria-download.bat"
    exit 1
fi

echo "✅ Modello trovato: $MODEL_PATH"
echo "🚀 Avvio llama-server per Orpheus..."

exec /llama.cpp/build/bin/llama-server \
    -m "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 5006 \
    --ctx-size "${ORPHEUS_MAX_TOKENS:-8192}" \
    --n-predict "${ORPHEUS_MAX_TOKENS:-8192}" \
    --rope-scaling linear \
    -ngl 99 \
    --flash-attn \
    -c "${ORPHEUS_MAX_TOKENS:-8192}" \
    --cache-type-k q8_0 \
    --cache-type-v q8_0
```

### Step 4 — Crea docker-compose ottimizzato per Blackwell

Crea `docker-compose-aria-gpu.yml` (non sovrascrivere il file originale):

```yaml
version: "3.8"

services:

  # Container 1: llama.cpp server — inferenza Orpheus
  llama-cpp-server:
    build:
      context: .
      dockerfile: Dockerfile.llama-blackwell
    image: aria-llama-blackwell:latest
    container_name: aria-llama-server
    restart: unless-stopped
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - CUDA_VISIBLE_DEVICES=0
      - ORPHEUS_MODEL_NAME=${ORPHEUS_MODEL_NAME}
      - ORPHEUS_MAX_TOKENS=${ORPHEUS_MAX_TOKENS:-8192}
    volumes:
      # Modelli: cartella locale Windows → container
      - C:\models\orpheus:/models:ro
    networks:
      - aria-internal
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:5006/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s

  # Container 2: Orpheus-FastAPI — API TTS
  orpheus-fastapi:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    image: aria-orpheus-fastapi:latest
    container_name: aria-orpheus-api
    restart: unless-stopped
    env_file: .env
    environment:
      - ORPHEUS_API_URL=http://llama-cpp-server:5006/v1/completions
    ports:
      - "5005:5005"    # Esposta sulla LAN per client ARIA
    volumes:
      - ./outputs:/app/outputs
    networks:
      - aria-internal
    depends_on:
      llama-cpp-server:
        condition: service_healthy

networks:
  aria-internal:
    driver: bridge
```

### Step 5 — Script download modello italiano

Crea `aria-download.bat` per Windows:

```bat
@echo off
echo [ARIA] Download modello Orpheus italiano...

set MODEL_NAME=Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf
set MODEL_REPO=lex-au/Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf
set MODEL_DIR=C:\models\orpheus

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

if exist "%MODEL_DIR%\%MODEL_NAME%" (
    echo [ARIA] Modello gia' presente in %MODEL_DIR%\%MODEL_NAME%
    echo [ARIA] Nulla da fare.
    pause
    exit /b 0
)

echo [ARIA] Scarico da HuggingFace: lex-au/%MODEL_NAME%
echo [ARIA] Destinazione: %MODEL_DIR%

huggingface-cli download ^
    "lex-au/Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf" ^
    "%MODEL_NAME%" ^
    --local-dir "%MODEL_DIR%" ^
    --local-dir-use-symlinks False

if %ERRORLEVEL% NEQ 0 (
    echo [ARIA] ❌ Download fallito. Verifica connessione e huggingface-cli.
    pause
    exit /b 1
)

echo [ARIA] ✅ Modello scaricato correttamente in %MODEL_DIR%
pause
```

### Step 6 — Verifica `Dockerfile.gpu` originale

Controlla che `Dockerfile.gpu` del repo originale sia compatibile con Python 3.11 (non 3.12 — non supportato da Orpheus-FastAPI). Se la versione Python è 3.12, correggila:

```dockerfile
# Deve essere 3.11, NON 3.12
FROM python:3.11-slim
```

### Step 7 — Struttura file finale

Al termine la struttura deve essere:

```
aria-server/
├── .env                          ← creato al Step 2
├── .env.example                  ← originale del repo
├── Dockerfile.gpu                ← originale del repo (verifica Python 3.11)
├── Dockerfile.llama-blackwell    ← NUOVO — Step 3
├── docker-compose-gpu.yml        ← originale del repo (non toccare)
├── docker-compose-aria-gpu.yml   ← NUOVO — Step 4
├── start-llama-server.sh         ← NUOVO — Step 3
├── aria-download.bat             ← NUOVO — Step 5
├── app.py                        ← originale del repo
├── requirements.txt              ← originale del repo
├── tts_engine/                   ← originale del repo
├── templates/                    ← originale del repo
├── static/                       ← originale del repo
└── outputs/                      ← creato vuoto per i WAV di output
```

---

## PROCEDURA DI BUILD E TEST

### Prima del primo avvio (una tantum)

```bash
# 1. Scarica il modello (richiede huggingface-cli installato su Windows)
aria-download.bat

# 2. Build immagini Docker
docker compose -f docker-compose-aria-gpu.yml build

# 3. Verifica GPU visibile
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

### Avvio normale

```bash
docker compose -f docker-compose-aria-gpu.yml up -d
```

### Log in tempo reale

```bash
# Log llama-server (inferenza)
docker logs -f aria-llama-server

# Log FastAPI (richieste)
docker logs -f aria-orpheus-api
```

### Test funzionale

```bash
# Test base in italiano con voce pietro
curl http://localhost:5005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "orpheus",
    "input": "pietro: Aprì la porta lentamente. Non ceera nessuno. Solo il vento freddo.",
    "voice": "pietro",
    "response_format": "wav"
  }' \
  --output test_pietro.wav

# Test con tag emotivi
curl http://localhost:5005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "orpheus",
    "input": "giulia: Aprì la porta. <gasp> Non cera nessuno. <sigh> Come sempre.",
    "voice": "giulia",
    "response_format": "wav"
  }' \
  --output test_giulia_emotional.wav

# Test health llama-server
curl http://localhost:5006/health

# Test health FastAPI
curl http://localhost:5005/health
```

### Test da minipc sulla LAN

```bash
# Sostituisci con IP reale del gaming PC
curl http://192.168.1.XX:5005/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"orpheus","input":"carlo: Test di rete dalla LAN.","voice":"carlo","response_format":"wav"}' \
  --output test_lan.wav
```

---

## TROUBLESHOOTING NOTO

### Build llama.cpp fallisce con errore CUDA architecture

Verifica che nel Dockerfile sia presente:
```dockerfile
-DCMAKE_CUDA_ARCHITECTURES="120"
```
RTX 5060 Ti è sm_120 (Blackwell). Qualsiasi altro valore causerà fallimento o performance degradate.

### Container llama-server si avvia ma non risponde su :5006

Il modello GGUF Q8_0 è ~3.5GB — il caricamento richiede 30-60 secondi. Il healthcheck ha `start_period: 60s` per questo motivo. Aspetta prima di diagnosticare.

### Flash Attention warning

Su Blackwell, llama.cpp usa Flash Attention v2 (non v3). Il warning è normale e non impatta la qualità. Non aggiungere flag per disabilitarlo.

### Python 3.12 non supportato

Orpheus-FastAPI usa `pkgutil.ImpImporter` rimosso in Python 3.12. La base image deve essere `python:3.11-slim` obbligatoriamente.

### Voce italiana non riconosciuta

Le voci italiane (`pietro`, `giulia`, `carlo`) sono disponibili solo con il modello `Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf`. Il modello inglese base non le supporta. Verifica il valore di `ORPHEUS_MODEL_NAME` nel `.env`.

### `C:\models\orpheus` non trovato in Docker

Docker Desktop su Windows richiede che il drive `C:\` sia abilitato alla condivisione. Vai in Docker Desktop → Settings → Resources → File Sharing → aggiungi `C:\models`.

---

## VOCI ITALIANE DISPONIBILI

| Voce | Genere | Carattere |
|------|--------|-----------|
| `pietro` | Maschile | Appassionato, espressivo |
| `giulia` | Femminile | Espressiva, calda |
| `carlo` | Maschile | Raffinato, misurato |

Tag emotivi supportati: `<laugh>` `<sigh>` `<chuckle>` `<cough>` `<sniffle>` `<groan>` `<yawn>` `<gasp>`

---

## COSA NON FARE

- ❌ Non usare vLLM — non compatibile con Blackwell senza build da sorgente + causa word-skipping su Orpheus
- ❌ Non usare `nvidia/cuda:12.8.0-*` con CUDA < 12.8 — RTX 5060 Ti richiede CUDA 12.8 minimo
- ❌ Non usare Python 3.12 — non supportato da Orpheus-FastAPI
- ❌ Non usare PyTorch direttamente per Orpheus — llama.cpp è la via corretta
- ❌ Non modificare `docker-compose-gpu.yml` originale — crea solo `docker-compose-aria-gpu.yml`
- ❌ Non esporre la porta 5006 (llama-server) all'esterno — è solo interna tra container

---

## PROSSIMO TASK (fuori scope ora)

Una volta che i due container funzionano e il test WAV italiano va a buon fine, il task successivo sarà sviluppare **Container 3: aria-broker** che:
- Si connette a Redis sul minipc (LAN)
- Legge task dalla coda `gpu:queue:tts:orpheus-3b`
- Chiama `orpheus-fastapi:5005/v1/audio/speech`
- Scrive risultati su `gpu:result:{client_id}:{job_id}`
- Implementa semaforo, heartbeat, e crash recovery

Il broker sarà un container Python separato aggiunto allo stesso `docker-compose-aria-gpu.yml`.

---

## CRITERI DI SUCCESSO

- [ ] `docker compose -f docker-compose-aria-gpu.yml build` completa senza errori
- [ ] `docker compose -f docker-compose-aria-gpu.yml up -d` avvia entrambi i container
- [ ] `nvidia-smi` dal container llama-server mostra RTX 5060 Ti
- [ ] `curl localhost:5006/health` risponde `{"status":"ok"}`
- [ ] `curl localhost:5005/health` risponde positivamente
- [ ] Test WAV italiano con voce `pietro` produce audio comprensibile
- [ ] Test con tag `<gasp>` produce effetto emotivo nell'audio
- [ ] API raggiungibile da minipc sulla LAN su porta 5005

---

*ARIA Server — Task AS-1/AS-6 parziale — Orpheus ITA Docker Setup*
*Febbraio 2026*
