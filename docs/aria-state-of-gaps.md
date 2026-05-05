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

---

## Gap Risolti

### A0-0 — Output WAV non eliminati da PC 139 dopo download Stage D
**Stato:** resolved  
**Risolto:** 2026-05-03  
**Descrizione:** I file WAV generati da Qwen3/Fish rimanevano in `data/outputs/` su PC 139 a tempo indeterminato dopo che Stage D li aveva già copiati su CT201. Spreco di spazio crescente su run lunghi.  
**Fix applicato:** `orchestrator.py` — aggiunto `do_DELETE` in `AriaAssetHandler`. Stage D — aggiunto `_delete_remote_asset()` chiamato dopo ogni download confermato. Commit `a891485` (ARIA) + `642b659` (DIAS).

### A0-1 — `dashboard/server.py` non tracciato su git
**Stato:** resolved  
**Risolto:** 2026-05-03  
**Descrizione:** Il file `aria_node_controller/dashboard/server.py` esisteva su PC 139 ma non era mai stato aggiunto al repo git. La copia su LXC 190 non lo aveva.  
**Fix applicato:** `git add` + push. Commit `f44a78b`.
