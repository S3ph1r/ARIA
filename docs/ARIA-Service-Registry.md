# ARIA Service Registry
## Stato Operativo — Maggio 2026

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
| Lifelog ASR (Qwen3-ASR-1.7B) | 8087 | `envs/lifelog-asr` | `backends/lifelog_asr/server.py` | ⏸️ Standby (sostituito da WhisperX) | ~9 GB |
| Lifelog LLM (qwen3-14b-q4km) | 8090 | `envs/lifelog-llm` | `llama-server.exe` b9119 (CUDA 13.1 sm_120) | ✅ Operativo (`startup_wait: 600s` — caricamento ~10 min) | ~9 GB |
| Lifelog WhisperX large-v3 | 8091 | `envs/lifelog-whisperx` | `backends/lifelog_whisperx/server.py` | ✅ Operativo (2026-05-14) | ~10 GB |
| FLUX.2-klein-4B | 8092 | `envs/flux-aria` | `backends/flux_imagegen/server.py` | ✅ Operativo (2026-05-16) | ~12.8 GB |

> I backend su porta 8084 e 8086 condividono lo stesso ambiente `dias-sound-engine` ma sono processi distinti avviati in momenti diversi — mai in contemporanea per gestione VRAM.

> Il backend 8087 è JIT: avviato dall'orchestratore al primo task Lifelog2, termina dopo 30 min di inattività. Include Qwen3-ASR-1.7B + ForcedAligner-0.6B + pyannote community-1.

---

## Code Redis

Tutte le code seguono il pattern: `aria:q:{type}:local:{model_id}:{client_id}`

| Coda | Backend | Tipo Task | Client |
|---|---|---|---|
| `aria:q:tts:local:qwen3-tts-1.7b:dias` | Qwen3-TTS (8083) | Sintesi vocale narrativa | DIAS |
| `aria:q:tts:local:fish-s1-mini:dias` | Fish S1-mini (8080) | TTS con emotion tagging | DIAS |
| `aria:q:llm:local:qwen3.5-35b-moe-q3ks:dias` | Qwen3.5 35B (8085) | LLM ragionamento | DIAS |
| `aria:q:mus:local:acestep-1.5-xl-sft:dias` | Orchestratore | Musica/Suono (PAD, AMB, SFX, STING, Leitmotif) | DIAS |
| `aria:q:stt:local:whisperx-large-v3:lifelog` | Lifelog WhisperX (8091) | Trascrizione + diarizzazione + voiceprint 256d | Lifelog2 |
| `aria:q:stt:local:qwen3-asr-1.7b:lifelog` | Lifelog ASR (8087) | ⏸️ Standby — sostituito da whisperx-large-v3 | Lifelog2 |
| `aria:q:llm:local:qwen3-14b-q4km:lifelog` | Lifelog LLM (8090) | LLM enrichment — MemoryAtom extraction | Lifelog2 |
| `aria:q:imagegen:local:flux2-klein-4b:lifelog` | FLUX.2-klein-4B (8092) | Image generation — cover episodi Lifelog2 | Lifelog2 |
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
| `lifelog-asr` | 3.12 | 2.11.0+cu128 | Qwen3-ASR-1.7B, ForcedAligner-0.6B, pyannote.audio 4.0.1 | ⏸️ Standby |
| `lifelog-whisperx` | 3.12 | 2.8.0+cu128 | WhisperX large-v3 (float16), pyannote community-1, wespeaker-resnet34-LM | ✅ Operativo (2026-05-14) |
| `flux-aria` | 3.12 | 2.x+cu128 | FLUX.2-klein-4B diffusion model, diffusers | ✅ Operativo (2026-05-16) |

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
http://localhost:8087/health       → Lifelog ASR (Qwen3-ASR-1.7B + ForcedAligner + pyannote) [standby]
http://localhost:8090/health       → Lifelog LLM (qwen3-14b-q4km via llama-server)
http://localhost:8091/health       → Lifelog WhisperX (large-v3 + align + pyannote + wespeaker)
http://localhost:8092/health       → FLUX.2-klein-4B image generation
```

> **Nota implementativa**: L'orchestratore ARIA risolve automaticamente `localhost` → `self.local_ip` (es. `192.168.1.139`) nelle URL di health check (`_health_check` in `orchestrator.py`). Questo bypassa il port-forwarding automatico di Firebase Studio (Antigravity IDE Google) che intercetta i port 8090/8091/8093 su `127.0.0.1` quando connesso via Remote SSH a LXC 190. Vedere sezione "Note Operative" in fondo a questo documento.

---

## Modelli su Disco (aggiornato 2026-05-17)

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
| Qwen3-ASR-1.7B | `models/qwen3-asr-1.7b/` | ~3.5 GB |
| Qwen3-ForcedAligner-0.6B | `models/qwen3-forced-aligner-0.6b/` | ~1.3 GB |
| pyannote speaker-diarization-community-1 | (HF cache `~/.cache/huggingface/`) | ~2 GB |
| FLUX.2-klein-4B | `data/assets/models/flux2-klein-4b/` | ~15.8 GB |

---

## Note Operative

### Firebase Studio (Antigravity IDE Google) — Port Conflict su loopback

**Problema**: Firebase Studio (Antigravity IDE Google), quando connesso via Remote SSH a LXC 190, fa port-forwarding automatico di servizi su LXC 190 verso `127.0.0.1:{port}` sul PC Windows 139. I port 8090, 8091, 8093 risultano occupati su loopback dal tunnel SSH di Firebase Studio.

**Sintomo**: L'orchestratore ARIA health-checkava `http://localhost:8090/health` → rispondeva Firebase Studio invece di `llama-server` → la risposta non era un JSON di health valido → timeout → loop infinito di riavvii di Qwen3-14B.

**Fix implementato** (`orchestrator.py`, metodo `_health_check`, 2026-05-17):
```python
url = self.MODEL_CONFIGS[model_id]["health_url"].replace("localhost", self.local_ip)
```
`self.local_ip` (es. `192.168.1.139`) viene risolto all'avvio tramite `get_node_ip()`. Usando l'IP esterno invece di `localhost`, il traffico va sulla NIC fisica e bypassa il port-forwarding su `127.0.0.1`. Il file `backends_manifest.json` rimane con `localhost` — nessun hardcoding.

**Regola operativa**: Se ARIA sembra avviare/riavviare ripetutamente un backend senza che esso risponda, verificare se Firebase Studio (Antigravity) è aperto e connesso via SSH a LXC 190 — potrebbe intercettare le porte su loopback.

### FLUX.2-klein-4B — Endpoint e Pattern Output

`backends/flux_imagegen/server.py` (porta 8092, binding `0.0.0.0`):

- `GET /health` → `{"status": "ok", "model": "flux2-klein-4b", "device": "cuda", "vram_gb": 12.8, "ready": true}`
- `POST /generate` → genera JPEG, salva in `ARIA_ROOT/data/outputs/{job_id}.jpeg`, ritorna `{"output_path": "...", "processing_time": 10.4}`
- `DELETE /output/{filename}` → elimina il JPEG locale dopo che il client ha scaricato e persistito l'immagine

**Pattern callback Redis** (campo `output` nel risultato):
```json
{
  "job_id": "uuid",
  "image_url": "http://192.168.1.139:8082/{job_id}.jpeg",
  "local_path": "C:\\Users\\Roberto\\aria\\data\\outputs\\{job_id}.jpeg",
  "processing_time_seconds": 10.4
}
```

Il client (Lifelog2 Stage G) scarica da `image_url`, persiste su MinIO, poi chiama `DELETE http://192.168.1.139:8092/output/{job_id}.jpeg` per liberare disco su PC139.

**Tutti i backend 0.0.0.0 (fix 2026-05-17):** tutti i backend FastAPI/uvicorn su PC139 sono stati migrati da `127.0.0.1` a `0.0.0.0` per essere raggiungibili via IP esterno (`192.168.1.139:PORT`). Necessario dopo la fix health-check Firebase Studio.

### Qwen3-14B startup_wait

`startup_wait` per `qwen3-14b-q4km` (porta 8090) impostato a **600s** (10 minuti). Il modello GGUF Q4KM richiede ~8-10 minuti per caricarsi completamente in VRAM con `llama-server`. Valori inferiori causano falsi timeout durante il caricamento.

---

## Documenti Correlati

- [ARIA Blueprint](ARIA-blueprint.md) — Architettura e principi di sistema
- [Hardware & Environments Setup](hardware-environments-setup.md) — Setup ambienti Python
- [DIAS ↔ ARIA Sound Integration](DIAS-ARIA-ACEStep-Integration.md) — Protocollo produzione audio
- [Audiocraft Backend](backends/audiocraft-backend.md) — Dettagli AudioGen/MusicGen
- [ACE-Step Payload Strategy](backends/acestep-payload-strategy.md) — Payload per PAD/Leitmotif
- [Lifelog ASR Backend](backends/lifelog-asr.md) — Qwen3-ASR-1.7B + ForcedAligner + pyannote [standby]
- [Lifelog WhisperX Backend](backends/lifelog-whisperx.md) — WhisperX large-v3 + voiceprint 256d, backend primario Lifelog2
