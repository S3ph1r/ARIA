# ARIA — State of Gaps

Registro gap architetturali e funzionali noti. Ogni entry ha un ID univoco, stato e data ultima modifica.

**Stato valori:** `open` | `in-progress` | `resolved` | `wont-fix`

---

## Gap Aperti

### A1-0 — Asset server: nessuna autenticazione su porta 8082
**Stato:** open  
**Priorità:** media  
**Scoperto:** 2026-04-24  
**Descrizione:** L'HTTP Asset Server integrato in `orchestrator.py` espone la cartella `outputs/` e `assets/` sulla LAN senza alcuna autenticazione. Chiunque sulla rete può leggere o listare i file.  
**Contesto:** Accettabile su rete locale fidata (dichiarato in OVERRIDES del .project-context). Diventa critico se ARIA viene esposta su internet via gateway CT202.  
**Fix proposto:** Auth basica HTTP o token Bearer sull'endpoint `/outputs/` prima di qualsiasi esposizione pubblica.

---

### A1-1 — Credenziali Redis in `node_settings.json` locale
**Stato:** open  
**Priorità:** bassa  
**Scoperto:** 2026-04-24  
**Descrizione:** Le credenziali Redis sono salvate in `node_settings.json` su Windows, fuori dal sistema SOPS+Age di NH-Mini.  
**Fix proposto:** Migrare a NH-Mini credential_manager con storage SOPS+Age.

---

### A1-2 — `dialogue_notes` enrichment path morto
**Stato:** open  
**Priorità:** bassa  
**Scoperto:** 2026-05-03  
**Descrizione:** `qwen3_tts.py` arricchisce `instruct` con `dialogue_notes` solo se `has_dialogue=True` e `dialogue_notes` è presente nel payload. DIAS Stage C produce sempre `dialogue_notes: null` e Stage D non forwarda `has_dialogue`. Il path non scatta mai.  
**Impatto:** Basso — il contesto personaggio è già sintetizzato da Gemini dentro `qwen3_instruct` a Stage C.  
**Fix proposto:** Stage C popola `dialogue_notes` con note carattere-specifiche; Stage D forwarda `has_dialogue` + `dialogue_notes` nel payload ARIA. Rilevante solo con casting multi-voce futuro.

---

### A1-3 — `subtalker_top_k` e `subtalker_top_p` non inviati da DIAS Stage D
**Stato:** open  
**Priorità:** bassa  
**Scoperto:** 2026-05-03  
**Descrizione:** Stage D invia `subtalker_temperature` da theatrical_standard ma non `subtalker_top_k` e `subtalker_top_p`. ARIA usa i default del server (50, 0.9). Non c'è modo di ottimizzarli per progetto senza modificare Stage D.  
**Fix proposto:** Estendere `theatrical_standard` in preproduction.json con `subtalker_top_k` e `subtalker_top_p`, e forwardarli in Stage D.

---

### A1-4 — Ambienti Conda in Miniconda globale (non isolati)
**Stato:** open  
**Priorità:** bassa  
**Scoperto:** 2026-04-24  
**Descrizione:** Gli ambienti Conda per Fish, Qwen3 ecc. sono installati nel Miniconda globale di Windows, non come `--prefix` isolati per ARIA. Rischio di conflitti di dipendenze con altri tool.  
**Fix proposto:** Migrare a `conda create --prefix C:\Users\Roberto\aria\envs\{nome}`.  
**Nota (2026-05-07):** L'env `lifelog-asr` è già stato creato con `--prefix` isolato — pattern corretto adottato per i nuovi backend.

---

## Gap Risolti

### A0-3 — Nessun backend LLM dedicato per Lifelog2 Stage D
**Stato:** resolved  
**Risolto:** 2026-05-13  
**Descrizione:** Lifelog2 Stage D richiedeva un modello LLM per estrarre MemoryAtom (summary, topics, entities, speaker_turns_annotated) da trascrizioni. Il backend LLM esistente (qwen3.5-35b-moe @ 8085) era dimensionato per DIAS, non per Lifelog. Necessario backend dedicato, più leggero, per operazioni di enrichment.  
**Soluzione adottata:**
- Env `lifelog-llm` su PC139 (Blackwell RTX 5060 Ti 16GB, sm_120)
- Modello: `qwen3-14b-q4km` via `llama-server.exe` build b9119 (CUDA 13.1, sm_120 native)
- Porta 8090, `/health` endpoint, prompt cache 8192 MiB
- Coda Redis: `aria:q:llm:local:qwen3-14b-q4km:lifelog`
- Timeout 600s (BatchOptimizer carica ASR prima se ci sono task in coda — competizione cold start)
- E2E testato: 21 segmenti → 21 MemoryAtom, timing ~17-21s per inferenza (warm)

---

### A0-2 — Nessun backend STT/ASR disponibile per Lifelog2
**Stato:** resolved  
**Risolto:** 2026-05-07  
**Descrizione:** ARIA non aveva alcun backend Speech-to-Text. Lifelog2 Stage C richiedeva trascrizione, diarizzazione speaker e word timestamps per costruire la memoria strutturata Z1.  
**Soluzione adottata:**
- Env `lifelog-asr` (Python 3.12, PyTorch 2.11.0+cu128, sm_120 native)
- Modelli: Qwen3-ASR-1.7B (WER IT 5.40%) + ForcedAligner-0.6B + pyannote community-1
- Backend `LifelogASRBackend` su porta 8087, JIT via `ModelProcessManager`
- Coda Redis: `aria:q:stt:local:qwen3-asr-1.7b:lifelog`
- Doc: [lifelog-asr.md](backends/lifelog-asr.md)

---

### A0-0 — Output WAV non eliminati da PC 139 dopo download Stage D
**Stato:** resolved  
**Risolto:** 2026-05-03  
**Descrizione:** I file WAV generati da Qwen3/Fish rimanevano in `data/outputs/` su PC 139 a tempo indeterminato dopo che Stage D li aveva già copiati su CT201. Spreco di spazio crescente su run lunghi.  
**Fix applicato:** `orchestrator.py` — aggiunto `do_DELETE` in `AriaAssetHandler`. Stage D — aggiunto `_delete_remote_asset()` chiamato dopo ogni download confermato. Commit `a891485` (ARIA) + `642b659` (DIAS).

### A0-4 — WhisperX come backend ASR primario + fix orchestratore (model_logic_ids + RLock)
**Stato:** resolved
**Risolto:** 2026-05-14
**Descrizione:** Tre problemi connessi emersi durante l'integrazione WhisperX large-v3 come ASR primario di Lifelog2:
1. `whisperx-large-v3` mancante da `model_logic_ids` in `orchestrator.py` (_run_loop): ARIA non scansionava la coda Redis del modello → task in coda indefiniti.
2. `backends/lifelog_whisperx.py` (handler class `LifelogWhisperXBackend`) mancante su PC139 → `_BACKENDS_AVAILABLE = False` → TUTTI i backend Python diventavano None, incluso `_asr_backend`.
3. Shutdown deadlock: `_ensure_single()` teneva `threading.Lock` e chiamava `_kill_proc()` che acquisiva lo stesso Lock → deadlock. Fix: `self._lock = threading.RLock()`.
**Soluzione adottata:**
- `orchestrator.py` riga 668: `"whisperx-large-v3"` aggiunto a `model_logic_ids`.
- `backends/lifelog_whisperx.py` deployato su PC139 (`LifelogWhisperXBackend`, `estimated_vram_gb()=12.0`).
- `threading.RLock()` sostituisce `Lock()` a riga 171.
- `backends_manifest.json`: entry `whisperx-large-v3` (porta 8091, env `lifelog-whisperx`, `startup_wait=150`).
- Stage C `stage_c_asr.py` su CT203: coda aggiornata a `aria:q:stt:local:whisperx-large-v3:lifelog`.
**E2E validato:** pipeline A→E in ~91s su segmento 299s audio (3.3× realtime warm). Qwen3-ASR-1.7B (8087) in standby.
**Nota architecturale:** ogni nuovo modello/backend DEVE essere aggiunto a `model_logic_ids` in `_run_loop()` — altrimenti ARIA è cieca alla coda. È un registro hardcoded, non auto-discovery.

---

### A0-1 — `dashboard/server.py` non tracciato su git
**Stato:** resolved  
**Risolto:** 2026-05-03  
**Descrizione:** Il file `aria_node_controller/dashboard/server.py` esisteva su PC 139 ma non era mai stato aggiunto al repo git. La copia su LXC 190 non lo aveva.  
**Fix applicato:** `git add` + push. Commit `f44a78b`.
