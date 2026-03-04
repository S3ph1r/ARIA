# 🔵 Qwen3-TTS Backend — Roadmap Operativa

> Checklist da spuntare step by step. Basata su `qwen3-tts-backend.md`.
> Ogni box `[ ]` diventa `[x]` quando completato.
> Backend parallelo a Fish S1-mini — porta 8083, ambiente `aria/envs/qwen3tts`, Python 3.12.
> Setup ambiente: `docs/environments-setup.md`.

---

## QW-0 — Test Manuale Standalone (nessuna modifica ad ARIA)

> **Obiettivo**: validare Qwen3 su Windows prima di toccare qualsiasi codice ARIA.
> **Criterio di successo**: italiano riconoscibile come voce del narratore, accenti
> corretti su "pàtina" e "futòn", nessun artefatto evidente. RTF > 1.0x.

- [ ] Creare ambiente: `conda create --prefix %ARIA_ROOT%\envs\qwen3tts python=3.12 -y`
- [ ] Installare PyTorch cu128: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`
- [ ] Installare dipendenze: `pip install transformers>=4.52.0 accelerate>=1.7.0 soundfile numpy fastapi uvicorn huggingface_hub`
- [ ] (Opzionale) Installare flash-attention 2 per cu128 + Python 3.12
- [ ] Download modello: `huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir C:\models\qwen3-tts-1.7b`
- [ ] Creare `ref_padded.wav` con `create_padded_ref.py` (sample narratore + 0.5s silenzio)
- [ ] Eseguire `test_qwen3_direct.py` — test sintesi senza cloning
- [ ] Eseguire `test_qwen3_direct.py` — test voice cloning dal sample narratore
- [ ] Ascoltare output e verificare: timbro, accenti italiani, artefatti
- [ ] Misurare RTF su RTX 5060 Ti (obiettivo: RTF > 1.0x)
- [ ] **GO/NO-GO**: qualità timbrica soddisfacente? Se no → rivalutare

**Stima**: 1 giorno

---

## QW-1 — Server FastAPI Standalone (porta 8083)

> **Obiettivo**: `qwen3_server.py` funzionante su porta 8083 con chunking automatico.

- [ ] Implementare `%ARIA_ROOT%\aria_node_controller\qwen3_server.py` (codice in §9 di qwen3-tts-backend.md)
- [ ] Test health check: `curl http://localhost:8083/health`
- [ ] Test sintesi base via HTTP: testo breve < 100 parole
- [ ] Test chunking: testo da 300 parole → verifica concatenazione audio corretta
- [ ] Test con `voice_ref_audio_path` — verifica voice cloning via HTTP
- [ ] Creare script `start-qwen3-tts.bat` (`conda activate qwen3-tts && python qwen3_server.py`)
- [ ] Configurare Task Scheduler: avvio automatico con ritardo 90s (dopo Fish su 8080)
- [ ] Test: reboot Windows → Qwen3 server disponibile su `:8083` entro 5 min
- [ ] Verificare raggiungibilità da MiniPC (LXC 192.168.1.120): `curl http://192.168.1.139:8083/health`

**Stima**: 1-2 giorni

---

## QW-2 — Backend ARIA `qwen3_tts.py`

> **Obiettivo**: ARIA Node Controller riconosce e usa `qwen3-tts-1.7b` come modello TTS.

- [ ] Creare `aria_node_controller/backends/qwen3_tts.py` (classe `Qwen3TTSBackend`)
- [ ] Aggiornare `config.yaml` — aggiungere sezione `models.tts.qwen3-tts-1.7b`
- [ ] Aggiornare `orchestrator.py` — registrare la coda `gpu:queue:tts:qwen3-tts-1.7b`
- [ ] Test: push manuale task su Redis (`redis-cli LPUSH gpu:queue:tts:qwen3-tts-1.7b ...`)
- [ ] Verifica risultato in `gpu:result:dias-minipc:{job_id}` con URL HTTP
- [ ] Verifica WAV scaricabile dall'URL HTTP (porta 8082 asset server)
- [ ] Test con task Fish in parallelo — verifica no interferenza tra code

**File da modificare**:
```
aria_node_controller/backends/qwen3_tts.py   ← NUOVO
aria_node_controller/core/orchestrator.py    ← aggiungere coda qwen3
aria_node_controller/core/config.yaml        ← sezione models.tts
```

**Stima**: 2-3 giorni

---

## QW-3 — Test Comparativo A/B Fish vs Qwen3

> **Obiettivo**: confronto oggettivo sugli stessi testi DIAS.

- [ ] Implementare `scripts/test_ab_comparison.py` (codice in §13 di qwen3-tts-backend.md)
- [ ] Eseguire test sui 3 estratti da "Cronache del Silicio": breve (50 parole), medio (150 parole), lungo (280 parole)
- [ ] Raccogliere WAV: `fish_breve.wav`, `qwen3_breve.wav`, `fish_medio.wav`, ecc.
- [ ] Ascolto A/B cieco (almeno 2 persone, senza sapere quale è quale)
- [ ] Valutazione su: timbro, accenti italiani, prosodia, pause, artefatti, naturalezza
- [ ] Misurare RTF comparativo: Fish vs Qwen3 sugli stessi testi
- [ ] Documentare risultati in `docs/ab_test_results.md`
- [ ] **Decisione formale**: Qwen3 sostituisce Fish, affianca Fish, o viene scartato?

**Stima**: 2-3 giorni (incluso ascolto e valutazione)

---

## QW-4 — Integrazione DIAS Completa (solo se QW-3 positivo)

> **Obiettivo**: DIAS usa Qwen3 come backend TTS principale o come alternativa selezionabile.

- [ ] Aggiornare `dias/src/stages/stage_d_voice_gen.py` — aggiungere `build_qwen3_payload()`
- [ ] Implementare mappa `primary_emotion → instruct Qwen3` (§6 di qwen3-tts-backend.md)
- [ ] Aggiornare `config.yaml` DIAS: `voice_backend.primary: "qwen3-tts-1.7b"` o mantenere Fish
- [ ] Test E2E: Libro → Stage A → B → C → Stage D (Qwen3) → WAV
- [ ] Confronto audio E2E: Fish end-to-end vs Qwen3 end-to-end su capitolo completo
- [ ] Aggiornare `docs/blueprint.md` DIAS sezione Stage D con info Qwen3
- [ ] Aggiornare `development-history.mdc` con entry FEATURE/ARCHITECTURE

**Stima**: 1 settimana

---

*Creato: 2026-03-04 — Da `qwen3-tts-backend.md`*
*Pattern speculare a `fish-tts-roadmap-checklist.md`*
