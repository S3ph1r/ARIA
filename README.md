# ARIA - Distributed GPU Inference Broker

ARIA è un broker di inferenza GPU distribuito per homelab, progettato per gestire modelli AI su GPU NVIDIA con supporto per TTS, STT e altri workload.

## Architettura

- **Backend**: Python FastAPI per API REST
- **GPU Support**: NVIDIA Container Toolkit con CUDA
- **Storage**: Condivisione di rete per modelli e output
- **Deployment**: Docker Compose con GPU passthrough

## Quick Start

```bash
# Build e run
docker-compose up --build

# Test GPU
curl http://localhost:7860/health
```

## Struttura

```
aria_server/
├── api/          # Endpoints REST
├── backends/     # Implementazioni backend (TTS, STT, etc.)
└── core/         # Core logic
tests/            # Unit e integration tests
scripts/          # Script di utilità
```

## Requisiti

- NVIDIA GPU con supporto CUDA
- Docker Desktop con GPU support
- Windows 10/11 con WSL2

## Licenza

Progetto indipendente che segue la filosofia NH-Mini (minimalismo, crescita organica).