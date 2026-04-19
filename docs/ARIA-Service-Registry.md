# ARIA Service Registry
## Stato Operativo — Aprile 2026

Registro completo di tutti i servizi attivi sul nodo ARIA (PC 139, `192.168.1.139`).
Aggiornato ad ogni modifica architetturale significativa.

---

## Infrastruttura

| Componente | Host | Indirizzo | Note |
|---|---|---|---|
| **Redis** | LXC 120 | `192.168.1.120:6379` | Infrastruttura condivisa, sempre attiva |
| **ARIA Node** | PC 139 | `192.168.1.139` | Windows 11 Pro, RTX 5060 Ti 16GB VRAM |
| **DIAS Brain** | LXC 190 | `192.168.1.190` | Ubuntu LXC, client principale |
| **Asset HTTP Server** | PC 139 | `192.168.1.139:8082` | Sempre attivo con l'orchestratore, serve `ARIA_ROOT/data/` |

---

## Backend Attivi

| Backend | Porta | Ambiente | Script | Stato | VRAM |
|---|---|---|---|---|---|
| Fish S1-mini TTS | 8080 | `envs/fish-speech-env` | `tools/api_server.py` | ✅ Operativo | ~3-4 GB |
| Fish Voice Cloning | 8081 | `envs/fish-speech-env` | `voice_cloning_server.py` | ✅ Operativo | CPU |
| Asset HTTP | 8082 | (orchestratore) | `AriaAssetHandler` | ✅ Sempre attivo | — |
| Qwen3-TTS 1.7B | 8083 | `envs/qwen3tts` | `backends/qwen3tts/server.py` | ✅ Operativo | ~4-5 GB |
| ACE-Step 1.5 XL SFT | 8084 | `envs/dias-sound-engine` | `backends/acestep/aria_wrapper_server.py` | ✅ Operativo | ~8 GB |
| Qwen3.5 35B MoE | 8085 | `envs/nh-qwen35-llm` | `backends/llm/server.py` | ✅ Operativo | ~13-14 GB |
| Audiocraft (AudioGen+MusicGen) | 8086 | `envs/dias-sound-engine` | `backends/audiocraft/aria_audiocraft_server.py` | ✅ Operativo | ~4-6 GB |

> I backend su porta 8084 e 8086 condividono lo stesso ambiente `dias-sound-engine` ma sono processi distinti avviati in momenti diversi — mai in contemporanea per gestione VRAM.

---

## Code Redis

Tutte le code seguono il pattern: `aria:q:{type}:local:{model_id}:{client_id}`

| Coda | Backend | Tipo Task | Client |
|---|---|---|---|
| `aria:q:tts:local:qwen3-tts-1.7b:dias` | Qwen3-TTS (8083) | Sintesi vocale narrativa | DIAS |
| `aria:q:tts:local:fish-s1-mini:dias` | Fish S1-mini (8080) | TTS con emotion tagging | DIAS |
| `aria:q:llm:local:qwen3.5-35b-moe-q3ks:dias` | Qwen3.5 35B (8085) | LLM ragionamento | DIAS |
| `aria:q:mus:local:acestep-1.5-xl-sft:dias` | Orchestratore | Musica/Suono (PAD, AMB, SFX, STING, Leitmotif) | DIAS |
| `aria:q:cloud:*` | CloudManager | Gemini API (fallback) | vari |

> La coda `aria:q:mus:local:acestep-1.5-xl-sft:dias` gestisce **tutti** i task audio. Il routing interno (ACE-Step vs Audiocraft) avviene tramite il campo `model_id` nel payload:
> - `"model_id": "acestep-1.5-xl-sft"` → porta 8084 (PAD, Leitmotif)
> - `"model_id": "audiocraft-medium"` → porta 8086 (AMB, SFX, STING)

---

## Ambienti Python (`envs/`)

| Ambiente | Python | PyTorch | Contenuto principale | Stato |
|---|---|---|---|---|
| `fish-speech-env` | 3.10 | 2.7+cu128 | Fish Audio S1-mini, VQGAN voice cloning | ✅ Operativo |
| `qwen3tts` | 3.12 | 2.6+cu124 | Qwen3-TTS 1.7B, DAC codec | ✅ Operativo |
| `nh-qwen35-llm` | 3.11 | — (llama.cpp) | Qwen3.5 35B MoE Q3KS GGUF | ✅ Operativo |
| `dias-sound-engine` | 3.11 | 2.11.0+cu128 | ACE-Step CLI, Demucs HTDemucs 6s, audiocraft 1.3.0 (AudioGen + MusicGen) | ✅ Operativo (★ env unificato) |
| `aria-cloud` | 3.12 | — | Google GenAI SDK (Gemini) | ✅ Operativo |
| `sox` | — | — | SoX audio processing tool | ✅ Operativo |
| `audiocraft-env` | 3.11 | 2.11.0+cu128 | — | ⛔ Deprecato (sostituito da `dias-sound-engine`) |

---

## Routing Sound Engine (Stage D2 → ARIA)

Il Stage D2 di DIAS determina il backend in base al tipo di asset:

| Asset Type | model_id inviato | Backend | Modello | Note |
|---|---|---|---|---|
| `pad` | `acestep-1.5-xl-sft` | ACE-Step (8084) | ACE-Step 1.5 XL SFT | Relay multi-chunk, HTDemucs |
| `leitmotif` | `acestep-1.5-xl-sft` | ACE-Step (8084) | ACE-Step 1.5 XL SFT | No relay, no demucs, 24s |
| `amb` | `audiocraft-medium` | Audiocraft (8086) | AudioGen medium | Routing interno wrapper |
| `sfx` | `audiocraft-medium` | Audiocraft (8086) | AudioGen medium | Routing interno wrapper |
| `sting` | `audiocraft-medium` | Audiocraft (8086) | MusicGen large | Routing interno wrapper |

---

## Health Check URLs

```
http://localhost:8080/v1/health    → Fish S1-mini TTS
http://localhost:8081/health       → Fish Voice Cloning
http://localhost:8082/             → Asset HTTP Server
http://localhost:8083/health       → Qwen3-TTS
http://localhost:8084/health       → ACE-Step wrapper
http://localhost:8085/v1/health    → Qwen3.5 35B LLM
http://localhost:8086/health       → Audiocraft (AudioGen + MusicGen)
```

---

## Modelli su Disco

Tutti i pesi risiedono in `ARIA_ROOT/data/assets/models/` (git-ignored).

| Modello | Path relativo | Dimensione |
|---|---|---|
| Fish Audio S1-mini | `models/fish-s1-mini/` | ~3 GB |
| Qwen3-TTS 1.7B | `models/qwen3-tts-1.7b/` | ~3.5 GB |
| Qwen3.5 35B MoE Q3KS | `models/qwen3.5-35b-moe-q3ks/` | ~14 GB |
| ACE-Step LM 1.7B | `backends/acestep/checkpoints/acestep-5Hz-lm-1.7B/` | ~3.5 GB |
| ACE-Step DiT XL SFT | `backends/acestep/checkpoints/acestep-v15-xl-sft/` | ~6 GB |
| MusicGen Large | `models/audiocraft/models--facebook--musicgen-large/` | ~3.3 GB |
| MusicGen Small | `models/audiocraft/models--facebook--musicgen-small/` | ~0.5 GB |
| AudioGen Medium | (scaricato da HuggingFace al primo avvio) | ~1.5 GB |
| HTDemucs 6s | (scaricato da HuggingFace al primo avvio) | ~0.5 GB |

---

## Documenti Correlati

- [ARIA Blueprint](ARIA-blueprint.md) — Architettura e principi di sistema
- [Hardware & Environments Setup](hardware-environments-setup.md) — Setup ambienti Python
- [DIAS ↔ ARIA Sound Integration](DIAS-ARIA-ACEStep-Integration.md) — Protocollo produzione audio
- [Audiocraft Backend](backends/audiocraft-backend.md) — Dettagli AudioGen/MusicGen
- [ACE-Step Payload Strategy](backends/acestep-payload-strategy.md) — Payload per PAD/Leitmotif
