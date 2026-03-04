# ARIA - Adaptive Resource for Inference and AI
## Distributed GPU Inference Broker per Homelab

> Piattaforma di inferenza AI privata per reti domestiche.
> ARIA trasforma il PC Gaming con GPU in un servizio AI condiviso sulla LAN —
> senza costi ricorrenti, senza privacy compromessa.

---

## Architettura

```
MiniPC (LXC 190)          Redis (CT120)          PC Gaming (Windows 11)
DIAS + ARIA Client   ◄──► Message Bus       ◄──► ARIA Node Controller
                          192.168.1.10:6379        RTX 5060 Ti 16GB
```

**Principio fondamentale**: un modello alla volta in VRAM, code Redis per tipo modello,
backend avviati on-demand dall'Orchestratore. Nessun dato lascia la rete locale.

### Ambienti Python (project-local)

```
C:\Users\%USERNAME%\
├── miniconda3\                    ← Python base (Orchestratore + Tray Icon)
└── aria\envs\
    ├── qwen3tts\                  ← Python 3.12 + PyTorch + qwen-tts
    └── fish-speech-env\           ← Python 3.10 + PyTorch + fish-speech
```

Ogni backend ha il suo `python.exe` isolato con le proprie dipendenze.
L'Orchestratore avvia/spegne i backend automaticamente via `subprocess.Popen()`.
Documentazione completa: `docs/environments-setup.md`.

---

## Backend Attivi

| Backend | Modello | VRAM | Porta | Stato |
|---|---|---|---|---|
| TTS (Audiolibro) | Qwen3-TTS-1.7B | ~4-5 GB | 8083 | ✅ Funzionante |
| TTS (Espressivo) | Fish Audio S1-mini | ~3-4 GB | 8080 | 🔄 Env da ricreare |
| Voice Cloning    | VQGAN (Fish) | CPU | 8081 | 🔄 Env da ricreare |
| LLM | Llama 3.1 8B (Q4) | ~5 GB | 8085 | 🔲 In sviluppo |
| Music | MusicGen-small | ~4 GB | — | 🔲 Futuro |
| STT | Whisper Large v3 | ~3 GB | — | 🔲 Futuro |

---

## Avvio Rapido (Nodo GPU)

```bat
:: Avvia l'Orchestratore ARIA (i backend partono on-demand)
Avvia_Tutti_Server_ARIA.bat
```

L'Orchestratore si connette a Redis, mostra l'icona verde nella systray, e avvia
i backend TTS automaticamente quando arrivano task dalle code Redis.

---

## Documentazione

| File | Contenuto |
|---|---|
| `docs/ARIA-blueprint.md` | **Documento principale** — architettura, filosofia, topologia, Redis |
| `docs/environments-setup.md` | **Guida ambienti** — setup Python, 3 livelli, come aggiungere backend |
| `docs/hybrid-tts-architecture.md` | Voice routing, Voice Library, ICL, Auto-Padding/Chunking |
| `docs/master-roadmap.md` | Stato completo del progetto — cosa è fatto e cosa è da fare |
| `docs/fish-tts-backend.md` | Backend Fish S1-mini: setup, emotion markers, integrazione |
| `docs/qwen3-tts-backend.md` | Backend Qwen3-TTS: setup, chunking, ICL |
| `docs/llm-backend.md` | Backend Llama 3.1 8B: setup, server API, integrazione |

> **Archivio**: `docs/archive/` — documentazione storica (Docker/Orpheus, backends-roadmap legacy)

---

## Requisiti Hardware / Software

- NVIDIA GPU con supporto CUDA 12.x (testato su RTX 5060 Ti, sm_120)
- Windows 11 con Miniconda (vedi `docs/environments-setup.md`)
- PyTorch 2.6+ con CUDA 12.4/12.8 (indici specifici per architettura GPU)
- Redis accessibile sulla rete locale

---

## Filosofia NH-Mini

Progetto open source che segue la filosofia NH-Mini:
minimalismo, crescita organica, privacy totale, zero dipendenze cloud.