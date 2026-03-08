# ARIA — Master Roadmap
## Stato Completo del Progetto — Realizzato vs. Da Fare

> **Aggiornato**: 2026-03-07
> **Versione**: v2.1 (Resilient & Agnostic)

---

## 🏗️ INFRASTRUTTURA ROLE-BASED

| ID | Task | Stato | Note |
|---|---|---|---|
| INF-1 | Redis Message Mesh (Decoupled) | ✅ | Indipendente dal nodo fisico |
| INF-2 | Heartbeat Redis (v2.1) | ✅ | `aria:global:node:{ip}:status` |
| INF-3 | Idempotenza Lato Worker (v2.1) | ✅ | Skip GPU se file esistente |
| INF-4 | ARIA Network Interface Spec | ✅ | `docs/ARIA-network-interface.md` |
| INF-5 | HTTP Asset Server (porta 8082) | ✅ | Servizio asset nativo |

---

## 🧠 ORCHESTRATORE — ARIA NODE CONTROLLER

| ID | Task | Stato | Note |
|---|---|---|---|
| ORC-1 | Queue Manager (v2.1) | ✅ | Supporto per registri esterni |
| ORC-2 | Result Writer (TTL 24h) | ✅ | |
| ORC-3 | Semaforo (Tray App) | ✅ | |
| ORC-4 | Crash Recovery (Visibility Lock) | ✅ | |
| ORC-5 | VRAM Manager (load/unload) | 🔄 | |
| ORC-6 | Batch Optimizer | 🔄 | |

---

## 🎧 BACKENDS - DETTAGLI TECNICI

| Componente | Documento di Riferimento | Stato |
|------------|--------------------------|-------|
| **Qwen3-TTS** | `docs/qwen3-tts-backend.md` | ✅ Operativo |
| **Fish-Speech** | `docs/fish-tts-backend.md` | 🔄 In Ripristino |
| **Music & SFX** | `docs/music-backend.md` | 🔲 Progettato |
| **LLM Backend** | `docs/llm-backend.md` | 🔲 Progettato |

---

| ID | Task | Stato | Note |
|---|---|---|---|
| OBS-1 | ARIA Health Dashboard (Streamlit) | 🔲 | Monitoraggio GPU e Heartbeats |
| OBS-2 | DIAS Project Dashboard (Streamlit) | 🔲 | Stato avanzamento audiolibro |

---
*ARIA Master Roadmap — Aggiornamento 7 Marzo 2026*
