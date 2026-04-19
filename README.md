# ARIA — Adaptive Resource for Inference and AI

ARIA è un orchestratore di nodi AI distribuito, progettato per erogare servizi AI ad alte prestazioni (TTS, LLM, Sound Generation) tramite un sistema di code Redis centralizzato.

**Visione: La Stampante di Rete per l'Inferenza.** ARIA opera in modo completamente asincrono. Le applicazioni client (come DIAS) accodano task su Redis senza conoscere lo stato dell'hardware. ARIA gestisce la GPU (ottimizzata per NVIDIA Blackwell sm_120), esegue il caricamento JIT degli ambienti, e restituisce il risultato (WAV, testo, JSON) via callback Redis e HTTP (Porta 8082).

---

## Caratteristiche Principali

- **Orchestrazione Multi-Backend**: gestione di più modelli AI (Fish Speech, Qwen3-TTS, Qwen3.5 LLM, ACE-Step, AudioGen, MusicGen) con caricamento/scaricamento JIT per ottimizzare l'uso della VRAM.
- **Architettura Ibrida**: modelli locali su GPU con fallover automatico verso API cloud (Gemini Flash Lite) quando le risorse locali sono occupate.
- **Thinking Mode**: estrazione e gestione nativa dei reasoning token per LLM avanzati (Qwen3.5 MoE).
- **Rate Limiting Avanzato**: pacing globale e gestione quote per provider cloud per prevenire errori 429.
- **API Standardizzata**: protocollo Redis semplice per submit task e ricezione risultati, integrabile da qualsiasi applicazione.
- **Sound Engine Unificato**: `dias-sound-engine` è il singolo ambiente Python per tutti i modelli audio — ACE-Step (PAD/Leitmotif), AudioGen (AMB/SFX), MusicGen (STING), HTDemucs (stem separation).

---

## Architettura

ARIA segue un pattern distribuito "Node & Orchestrator":

1. **Node Controller**: daemon Windows che monitora le code Redis e avvia processi figlio per i backend AI locali.
2. **Model Backends**: ambienti isolati per task specializzati (TTS, LLM, Sound).
3. **Cloud Manager**: worker dedicato per chiamate API di terze parti senza bloccare le risorse GPU locali.

```
RETE LOCALE
  ┌─────────────────┐     Redis (LXC 120)      ┌──────────────────────┐
  │  BRAIN NODE     │     192.168.1.120:6379    │  WORKER NODE (GPU)   │
  │  LXC 190        │◄─────────────────────────►│  PC 139              │
  │  DIAS / altri   │                           │  ARIA SERVER         │
  └─────────────────┘                           │  192.168.1.139       │
                                                │  RTX 5060 Ti 16GB    │
                                                └──────────────────────┘
```

---

## Struttura del Progetto

```
aria/
├── aria_node_controller/            # Core Orchestratore
│   ├── core/                        # Main loop, QueueManager, BatchOptimizer, RateLimiter
│   ├── backends/                    # Connector per ogni backend
│   │   ├── acestep.py               # ACE-Step connector (porta 8084)
│   │   ├── audiocraft.py            # AudioGen/MusicGen connector (porta 8086)
│   │   ├── qwen3_tts.py             # Qwen3-TTS connector (porta 8083)
│   │   └── qwen35_llm.py            # Qwen3.5 LLM connector (porta 8085)
│   └── config/
│       └── backends_manifest.json   # Registry JIT: porte, script, env per ogni backend
├── backends/
│   ├── acestep/                     # ACE-Step CLI wrapper (porta 8084)
│   │   └── aria_wrapper_server.py   # FastAPI, relay chunking, tonal lock
│   ├── audiocraft/                  # AudioGen + MusicGen wrapper (porta 8086)
│   │   └── aria_audiocraft_server.py
│   ├── llm/                         # Qwen3.5 35B server (porta 8085)
│   └── qwen3tts/                    # Qwen3-TTS server (porta 8083)
├── envs/                            # Ambienti Python isolati (git-ignored)
│   ├── dias-sound-engine/           # ★ Env unificato: ACE-Step + AudioGen + MusicGen + Demucs
│   ├── fish-speech-env/             # Fish S1-mini TTS + Voice Cloning
│   ├── qwen3tts/                    # Qwen3-TTS 1.7B
│   ├── nh-qwen35-llm/               # Qwen3.5 35B MoE (llama.cpp)
│   ├── aria-cloud/                  # Google Gemini SDK
│   └── sox/                         # SoX audio tool
├── data/
│   ├── assets/
│   │   ├── models/                  # Pesi modelli (git-ignored)
│   │   ├── sound_library/           # Output audio: pad/ amb/ sfx/ sting/ music/
│   │   └── voices/                  # Voice Library (ref.wav + ref.txt per voce)
│   └── outputs/                     # Output generici TTS/LLM
├── docs/                            # Documentazione tecnica
└── node_settings.json               # Configurazione nodo (git-ignored)
```

---

## Setup

1. **Clona il repository**: `git clone https://github.com/S3ph1r/aria.git && cd aria`
2. **Configura il nodo**: copia `node_settings.json.example` → `node_settings.json`, inserisci IP Redis e API key.
3. **Installa dipendenze core**: `pip install -r requirements/core.txt`
4. **Setup ambienti backend**: vedi [Hardware & Environments Setup](docs/hardware-environments-setup.md).

---

## Documentazione

### Architettura
- **[ARIA Blueprint](docs/ARIA-blueprint.md)** — Manuale universale: architettura, componenti, schema task, ciclo di vita, semaforo GPU.
- **[ARIA Service Registry](docs/ARIA-Service-Registry.md)** — Registro operativo di tutti i servizi: porte, code Redis, ambienti, stato corrente.
- **[Hardware & Environments Setup](docs/hardware-environments-setup.md)** — Guida ambienti Python in `envs/`, criticità CUDA sm_120 Blackwell.

### Integrazione DIAS
- **[DIAS ↔ ARIA Sound Integration](docs/DIAS-ARIA-ACEStep-Integration.md)** — Protocollo completo: Stage D2, routing ACE-Step/Audiocraft, leitmotif, HTDemucs, Stage E.
- **[DIAS Sound Production Roadmap](docs/dias-aria-sound-production-roadmap.md)** — Filosofia, pipeline Stage A-G, roadmap fasi, stato implementativo.
- **[DIAS to ARIA Integration Spec](docs/dias-to-aria-integration-spec.md)** — Specifiche payload per tipo asset (PAD, AMB, SFX, STING, Leitmotif).

### Backend
- **[Fish Audio S1-mini](docs/backends/fish-s1-mini.md)** — TTS con emotion tagging e voice cloning (Porte 8080/8081).
- **[Qwen3-TTS 1.7B](docs/backends/qwen3-tts.md)** — TTS narrativo LLM-based (Porta 8083).
- **[ACE-Step 1.5 XL SFT](docs/backends/acestep-payload-strategy.md)** — Sound engine musicale: PAD, leitmotif, relay chunking, HTDemucs (Porta 8084).
- **[Audiocraft — AudioGen & MusicGen](docs/backends/audiocraft-backend.md)** — Sound engine AMB/SFX/STING (Porta 8086).
- **[Qwen3.5 35B MoE](docs/backends/qwen35-llm-moe.md)** — LLM locale ibrido con fallover Gemini cloud (Porta 8085).
- **[Hybrid TTS Routing](docs/backends/hybrid-tts-routing.md)** — Routing TTS (Fish vs Qwen3), Voice Library, onboarding voci.
- **[ACE-Step Auto-Chaining](docs/backends/acestep-auto-chaining-development.md)** — Relay per tracce musicali >120s.

---

*Parte del framework NH-Mini — sviluppato da S3ph1r.*
