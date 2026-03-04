# ARIA — Master Roadmap
## Stato Completo del Progetto — Realizzato vs. Da Fare

> **Aggiornato**: 2026-03-04
> **Hardware**: GPU NVIDIA (vedi `docs/environments-setup.md` per il deploy corrente)
> **Filosofia**: Un modello alla volta in VRAM, queue Redis, External HTTP Backends

---

## LEGENDA

| Simbolo | Stato |
|---|---|
| ✅ | Completato — testato e funzionante |
| 🔄 | In corso — parzialmente implementato |
| 🔲 | Da fare — solo sulla carta |
| ❌ | Archiviato / Superato — non da implementare |

---

## 🏗️ INFRASTRUTTURA BASE

| ID | Task | Stato | Note |
|---|---|---|---|
| INF-1 | Redis su MiniPC come message bus | ✅ | `%REDIS_HOST%` |
| INF-2 | Nodo GPU raggiungibile via SSH da dev server | ✅ | Workflow CI/CD domestico |
| INF-3 | Miniconda su Nodo GPU Windows | 🔄 | `%MINICONDA_ROOT%` — da reinstallare |
| INF-4 | HTTP Asset Server (porta 8082) per WAV output | ✅ | Integrato in Node Controller |
| INF-5 | ARIA Node Controller (Tray App) | 🔄 | Core funzionante, path da allineare |

---

## 🐟 BACKEND TTS — FISH S1-MINI

> Documento dettagliato: `docs/fish-tts-backend.md`
> Setup ambiente: `docs/environments-setup.md`

| ID | Task | Stato | Note |
|---|---|---|---|
| FS-1 | Env conda `fish-speech-env` in `aria/envs/` | 🔄 | Da ricreare (Python 3.10 + PyTorch 2.7+cu128) |
| FS-2 | Fish S1-mini server TTS (porta 8080) | ✅ | `tools/api_server.py` |
| FS-3 | Voice Cloning VQGAN (porta 8081) | ✅ | Unificato in `fish-speech-env` |
| FS-4 | Backend integrato nell'Orchestratore | ✅ | External HTTP Backend on-demand |
| FS-5 | Voice Library (`data/voices/`) | ✅ | angelo, luca — ref.wav + ref.txt |
| FS-6 | `voice_prepper.py` per creare nuovi sample | ✅ | Scraping YouTube + Gemini trascrizione |
| FS-7 | Test E2E: DIAS → Redis → Fish → WAV | ✅ | Primo capitolo audiolibro generato |
| FS-8 | Aggiornamento TextDirector DIAS (tag Fish) | 🔲 | Prompt con marker Fish invece di Orpheus |

---

## 🔧 BACKEND LLM — LLAMA 3.1 8B

> Documento dettagliato: `docs/llm-backend.md`
> Roadmap operativa: `docs/backends-roadmap.md` (FASE BK-2)

| ID | Task | Stato | Note |
|---|---|---|---|
| LLM-1 | Analisi ambienti Fish — unificazione? | 🔲 | Prerequisito BK-1 in backends-roadmap.md |
| LLM-2 | Env conda `llm-backend` (Python 3.11 + PyTorch 2.7+cu128) | 🔲 | Stesso index cu128 di Fish |
| LLM-3 | Download modello Llama 3.1 8B Instruct | 🔲 | ~16GB, richiede HF token + licenza |
| LLM-4 | `llm_server.py` (FastAPI + bitsandbytes Q4) | 🔲 | Porta 8085 |
| LLM-5 | Test qualità output italiano per DIAS | 🔲 | Scene Director + Text Director |
| LLM-6 | Backend `aria_server/backends/llm_backend.py` | 🔲 | External HTTP Backend |
| LLM-7 | Aggiornamento `config.yaml` + orchestratore | 🔲 | `llama-3.1-8b: enabled: true` |
| LLM-8 | Avvio automatico nel `.bat` | 🔲 | +90s wait per model load |
| LLM-9 | Test E2E: DIAS → Redis → LLM → JSON scena | 🔲 | |

---

## 🎵 BACKEND MUSIC — MUSICGEN (FUTURO)

| ID | Task | Stato | Note |
|---|---|---|---|
| MU-1 | Backend `musicgen_backend.py` | 🔲 | Da progettare |
| MU-2 | Setup env conda `music-backend` | 🔲 | audiocraft richiede env dedicato |
| MU-3 | Integrazione queue `gpu:queue:music:musicgen-small` | 🔲 | |

---

## 🖼️ BACKEND IMAGE — DIFFUSERS (FUTURO)

| ID | Task | Stato | Note |
|---|---|---|---|
| IMG-1 | Backend `image_backend.py` (SDXL / Flux) | 🔲 | Da progettare |
| IMG-2 | Flux-dev richiede ~12GB — solo con VRAM libera | 🔲 | Vincolo hardware critico |

---

## 🎙️ BACKEND STT — WHISPER (FUTURO)

| ID | Task | Stato | Note |
|---|---|---|---|
| STT-1 | Backend `stt_backend.py` (faster-whisper) | 🔲 | 3GB VRAM, potrebbe coesistere |

---

## 🧠 ORCHESTRATORE — ARIA NODE CONTROLLER

| ID | Task | Stato | Note |
|---|---|---|---|
| ORC-1 | Queue Manager (BRPOP Redis) | ✅ | Funzionante |
| ORC-2 | Result Writer (TTL + crash recovery) | ✅ | Funzionante |
| ORC-3 | Semaforo (Green/Red/Busy) | ✅ | Tray icon integrata |
| ORC-4 | Heartbeat Redis ogni 10s | ✅ | `gpu:server:heartbeat` |
| ORC-5 | VRAM Manager (load/unload) | 🔄 | Base funzionante, OOM handling da raffinare |
| ORC-6 | Batch Optimizer — strategia "finisci coda corrente" | 🔄 | Base funzionante, lazy-switch da implementare |
| ORC-7 | Crash Recovery (task in `gpu:processing:*` al riavvio) | 🔲 | Da implementare |
| ORC-8 | Dead Letter Handler (task scaduti) | 🔲 | Da implementare |
| ORC-9 | API HTTP `GET /status`, `POST /semaphore` | 🔲 | Da implementare |
| ORC-10 | Dashboard Web monitoring (opzionale) | 🔲 | Bassa priorità |

---

## 📦 INFRASTRUTTURA AMBIENTI

> Documento dettagliato: `docs/environments-setup.md`

| ID | Task | Stato | Note |
|---|---|---|---|
| ENV-1 | Miniconda globale su nodo GPU | 🔄 | `%MINICONDA_ROOT%` — da installare |
| ENV-2 | Ambiente Qwen3-TTS project-local | ✅ | `%ARIA_ROOT%/envs/qwen3tts` — Python 3.12 |
| ENV-3 | Ambiente Fish-Speech project-local | 🔄 | `%ARIA_ROOT%/envs/fish-speech-env` — da creare |
| ENV-4 | Fix tutti i path nel codice sorgente | 🔄 | Eliminare riferimenti hardcoded |

---

## 🗃️ DOCUMENTAZIONE

| ID | Task | Stato | Note |
|---|---|---|---|
| DOC-1 | **ARIA Blueprint** (doc principale) | ✅ | `docs/ARIA-blueprint.md` — architettura, filosofia, topologia |
| DOC-2 | **Guida Ambienti** | ✅ | `docs/environments-setup.md` — **NUOVO** setup Python |
| DOC-3 | **Hybrid TTS Architecture** | ✅ | `docs/hybrid-tts-architecture.md` — voice routing, ICL |
| DOC-4 | Master Roadmap (questo file) | ✅ | `docs/master-roadmap.md` |
| DOC-5 | Fish TTS Backend (tecnico) | ✅ | `docs/fish-tts-backend.md` |
| DOC-6 | Qwen3 TTS Backend (tecnico) | ✅ | `docs/qwen3-tts-backend.md` |
| DOC-7 | LLM Backend (tecnico) | ✅ | `docs/llm-backend.md` |
| DOC-8 | README.md | ✅ | Root del repo |
| DOC-9 | Archivio legacy | ✅ | `docs/archive/` — Docker, Orpheus, backends-roadmap |

---

## 🔢 PRIORITÀ PROSSIMI PASSI

```
1. ENV-1  → Installare Miniconda globale su nodo GPU (30 min)
2. ENV-4  → Fix path nel codice sorgente (bat, main_tray, orchestrator) (1 ora)
3. INFRA  → Test E2E Qwen3 sotto profilo corretto (1 ora)
4. ENV-3  → Ricreare env Fish-Speech (1 giorno)
5. QW-3   → Test comparativo A/B Fish vs Qwen3 (2-3 giorni)
6. LLM-2  → Setup env LLM + server (1 giorno)
7. ORC-7  → Crash Recovery (2-3 giorni)
```

---

*ARIA Master Roadmap — Marzo 2026*
*Documenti collegati:*
- *`docs/ARIA-blueprint.md` — architettura, filosofia, topologia*
- *`docs/environments-setup.md` — guida ambienti Python*
- *`docs/hybrid-tts-architecture.md` — voice routing e ICL*
- *`docs/fish-tts-backend.md` — backend Fish S1-mini*
- *`docs/qwen3-tts-backend.md` — backend Qwen3-TTS*
- *`docs/llm-backend.md` — backend Llama 3.1 8B*
