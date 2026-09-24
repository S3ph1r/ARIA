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

### A1-5 — Self-reporting PID mancante su quasi tutti i backend (kill silenziosamente inefficace) + finestra Console mai chiusa dopo un kill per PID
**Stato:** in-progress (fix strutturale scritto il 2026-09-24 su LXC 190, ancora da deployare su PC 139 e verificare dal vivo su uno swap reale — vedi ultimo aggiornamento sotto)
**Priorità:** alta
**Scoperto:** 2026-09-05 (Roberto, durante il reprocess storico di Lifelog2 — qwen3-14b-q4km e Flux2/covers in alternanza sulla stessa GPU)
**Descrizione:** Il self-reporting del PID reale (introdotto 2026-08-15, vedi Gap Risolti sotto) esisteva SOLO per `flux2-klein-4b`. Per qualunque altro backend (qwen3-14b-q4km incluso — il più usato, il modello LLM condiviso da Lifelog2/DIAS), `_kill_proc()` ricade sul vecchio fallback (`proc.poll() is None` sul Popen del lanciatore 'start'), che è quasi sempre falso perché su Windows quel lanciatore muore da solo pochi secondi dopo aver aperto la finestra reale — il comando di kill non parte mai, l'entry viene comunque rimossa dal tracking interno (`_procs.pop`), e ARIA "dimentica" il backend senza mai ucciderlo. Confermato dal vivo sui log di produzione: uno swap "GPU Exclusivity: Termino qwen3-14b-q4km per far posto a flux2-klein-4b" non è mai seguito da nessuna riga di conferma di terminazione, a differenza della direzione opposta (Flux2, che il self-reporting ce l'ha, termina pulito col proprio PID). Rischio concreto: due modelli caricati insieme sulla stessa GPU (qwen3-14b-q4km, ~9GB+7GB KV cache, e Flux2, ~12.8GB) — a seconda della VRAM disponibile, contesa/rallentamento o superamento della capacità.

**Secondo problema, distinto ma collegato:** anche quando il kill per PID riesce (caso Flux2), uccide solo l'albero radicato in quel PID — mai il `cmd.exe` antenato che ha aperto la finestra reale (lanciata con `start "titolo" cmd.exe /k ...`). La finestra Console resta quindi sempre aperta con un prompt vuoto dopo un kill per PID, anche quando il kill stesso funziona.

**Fix scritto (2026-09-05):**
- `backends/lifelog_llm/launcher.py`: stesso self-reporting PID già in produzione su `flux_imagegen/server.py`, adattato al fatto che questo launcher esegue llama-server.exe come sottoprocesso (si auto-riporta il PROPRIO pid, un taskkill `/T` su quello termina anche il figlio). Import di `psutil` avvolto in try/except esplicito — se assente nell'env `lifelog-llm` (mai verificato prima d'ora), il self-reporting viene saltato con un warning invece di far fallire l'avvio del backend.
- `aria_node_controller/core/orchestrator.py::_kill_proc()`: dopo un kill per PID riuscito, tenta SEMPRE anche la chiusura per titolo finestra (stessa chiamata già usata nel ramo fallback) — le due cose sono complementari, non alternative. Aggiunto anche un controllo esplicito del codice di uscita di ogni `taskkill` (prima veniva dichiarato "terminato" incondizionatamente).
**Aggiornamento, stesso giorno (Roberto: "aria non può salvare il titolo finestra e il PID in un file quando starta i backend?"):** aggiunta una SECONDA via, universale, che copre ogni backend senza toccarne il codice — `_discover_pid_by_window_title()` in `orchestrator.py`, chiamata subito dopo un health-check riuscito (sia al primo avvio che quando un backend risulta "già attivo", il percorso più frequente in steady-state). Interroga `tasklist /FI "WINDOWTITLE eq ..."` per lo stesso titolo già usato ovunque nel file, trova il PID di `cmd.exe` (il processo che possiede la finestra — non quello del backend, che ne è figlio) e lo salva nello stesso pid-file. Vantaggio rispetto al self-reporting: un solo `taskkill /T` su questo PID chiude insieme backend E finestra, e funziona per QUALUNQUE backend incluso whisperx/fish-speech/qwen3-tts/acestep/qwen3-asr/qwen3.5-moe senza toccarne il codice. Il self-reporting resta preferito quando esiste (`_read_pid_file` guarda prima quello — cattura il PID del backend un istante dopo la sua creazione, più preciso), questa via è il fallback universale per tutto il resto.
**Non ancora fatto:** self-reporting esplicito per gli altri backend resta comunque possibile in futuro (stesso pattern di `launcher.py`/`flux_imagegen/server.py`) se mai servisse la precisione in più — ma con la scoperta via titolo finestra sopra, non è più un gap bloccante.
**Verificato dal vivo (2026-09-05, dopo il riavvio con il fix):** `logs/pids/qwen3-14b-q4km.pid` popolato correttamente dalla scoperta via titolo finestra (self-reporting del launcher non è mai partito, verosimilmente psutil assente nell'env `lifelog-llm` — il fallback ha coperto correttamente il buco, esattamente come disegnato). PID salvato = quello di `cmd.exe`, confermato via `Get-CimInstance Win32_Process` (albero cmd.exe → python.exe/launcher.py → llama-server.exe). Confermato anche lato Lifelog2 (Redis): nessun messaggio perso durante l'intera finestra di stop/fix/restart — il job in coda al momento dello stop è stato ripescato dal meccanismo di re-push su timeout già esistente (`AriaLLMClient: job consumato senza risposta — re-push`) e completato correttamente dopo il riavvio.
**Ancora da osservare dal vivo:** un kill/swap reale (qwen3→Flux2 o viceversa) con il fix attivo, per confermare che la finestra si chiuda davvero e non solo il processo.

**Aggiornamento 2026-09-24 (Roberto — scoperta dal vivo, causa radice trovata e corretta):** il fix del 2026-09-05 riduceva il problema ma non lo chiudeva. Durante una sessione normale, uno swap reale flux2-klein-4b → qwen3-14b-q4km ha chiuso una finestra PowerShell **completamente estranea ad ARIA** (il "Sniper watchdog", un progetto indipendente di Roberto sul PC 139, con la propria daemon+viewer). Diagnosi (verificata dal vivo via SSH read-only su PC 139, nessuna modifica: `Get-CimInstance Win32_Process`, lettura pid-file, lettura log orchestratore):

- `envs\lifelog-llm` (l'env di `qwen3-14b-q4km`) non ha `psutil` installato → il self-reporting di `launcher.py` fallisce silenziosamente (solo un warning nel log del *backend*, invisibile nel log dell'orchestratore) → ARIA ricade sempre sulla scoperta via titolo finestra per questo backend.
- La scoperta via titolo finestra ha salvato PID **29728** in `qwen3-14b-q4km.pid`. Verificato via `Get-CimInstance Win32_Process`: quel PID è **`WindowsTerminal.exe`**, non il backend — il vero albero era `cmd.exe (37028) → python.exe/launcher.py (33844) → llama-server.exe (42072)`, completamente slegato da 29728.
- Causa: Windows 11 delega l'allocazione di **ogni** nuova console a Windows Terminal (l'app terminale predefinita) — `tasklist /FI "WINDOWTITLE eq ..."` può quindi restituire il PID del contenitore grafico (`WindowsTerminal.exe`/`OpenConsole.exe`) invece del processo applicativo. Un `taskkill /PID {quel_pid} /T /F` al prossimo swap avrebbe abbattuto l'intera applicazione terminale — con qualunque altra finestra/scheda al suo interno, incluso Sniper se ospitato nella stessa istanza.
- **Non è un fallback "a maglie larghe" nel codice** — verificato riga per riga sul log dell'orchestratore e con un test dal vivo (`tasklist` su un titolo inesistente → correttamente zero risultati, non un dump di tutti i processi): ogni comando eseguito allo swap è scoping-ato per PID esatto o titolo esatto. Il problema è che il *bersaglio* scoperto era quello sbagliato, non il meccanismo di selezione.

**Fix strutturale (2026-09-24, non solo una toppa sull'env `lifelog-llm`):** eliminata la dipendenza da self-reporting/scoperta-per-titolo come UNICA fonte di verità per il PID di avvio.
- `orchestrator.py::_ensure_single` (spawn Windows): rimosso `start "title" cmd.exe /k ...` con `shell=True`. Ora `subprocess.Popen(["cmd.exe", "/c", f'title {title} && {cmd_str}'], creationflags=subprocess.CREATE_NEW_CONSOLE, ...)` — spawn diretto, `new_proc.pid` è sempre il PID reale del `cmd.exe` che avviamo noi, dato dal sistema operativo al momento dello spawn stesso, indipendente da quale frontend Windows scelga per renderizzare la finestra. `/c` invece di `/k`: la finestra si chiude da sola a fine processo.
- Nuovo helper `orchestrator.py::_get_tracked_pid(model_id)`: prova in ordine (1) pid-file self-reported, (2) `self._procs[model_id].pid` (ora sempre corretto grazie allo spawn diretto — **funziona per QUALUNQUE backend, zero dipendenza da psutil nel suo env**), (3) scoperta via titolo finestra solo come ultimissima risorsa (backend rimasto vivo da un riavvio precedente dell'orchestratore, mai tracciato in questa sessione). Usato ora sia in `_ensure_single` che in `_kill_proc`.
- `orchestrator.py::_discover_pid_by_window_title`: aggiunto un filtro esplicito che scarta il PID scoperto se il suo Image Name è `WindowsTerminal.exe`/`OpenConsole.exe`/`conhost.exe` (un contenitore, mai un backend applicativo) — rete di sicurezza per il caso residuo (3) sopra, anche per backend futuri con la stessa lacuna di dipendenze.

**Effetto pratico:** il gap non richiede più di installare `psutil` env per env (non fa comunque male farlo, ma non è più necessario) — la correzione è centralizzata nell'orchestratore e si applica automaticamente a ogni backend che ARIA orchestra, presente o futuro, senza toccarne il codice.

**Ancora da verificare dal vivo:** deploy su PC 139 (`git pull` + riavvio manuale di `aria.bat`) e conferma su uno swap reale che `self._procs[model_id].pid` sia effettivamente quello letto da `_kill_proc` (non più scoperta per titolo) e che nessuna finestra estranea ad ARIA venga toccata.

**Aggiunta correlata (2026-09-05, Roberto): posizione fissa della finestra Console.** Windows/il console host non hanno un modo nativo di fissare la posizione di avvio via riga di comando — le finestre si aprivano sparse per il desktop ad ogni avvio. Aggiunta `_position_window()` in `orchestrator.py`: dopo il primo avvio fresco (mai sul percorso "già attivo", che gira ad ogni health check), usa `EnumWindows`/`GetWindowText`/`SetWindowPos` (ctypes, stdlib, nessuna nuova dipendenza) per trovare la finestra per titolo e spostarla a (0,0). Complessità bassa, pattern Win32 standard. **Non verificabile dalla sessione SSH usata per il deploy** (testato: `EnumWindows` da una sessione Windows diversa da quella interattiva di Roberto non vede le finestre — isolamento per sessione, atteso) — funziona perché il codice reale gira nell'orchestratore, nella STESSA sessione desktop che apre le finestre (stesso motivo per cui il taskkill per titolo funziona altrove in questo file). Da confermare visivamente al prossimo avvio fresco di un backend.

---

### A1-6 — `aria.bat` uccideva ogni `python.exe` del sistema all'avvio (non solo i propri)
**Stato:** in-progress (fix scritto il 2026-09-24 su LXC 190, ancora da deployare su PC 139)
**Priorità:** alta
**Scoperto:** 2026-09-24 (Roberto, durante l'indagine su [[A1-5]] — sintomo iniziale: il watchdog Python di un progetto indipendente sul PC 139, "Sniper", moriva ad ogni avvio di ARIA senza mai essere resuscitato, perché nulla protegge il watchdog stesso una volta ucciso)
**Descrizione:** `aria.bat`, prima operazione all'avvio ("zombie prevention"), eseguiva:
```
taskkill /F /IM python.exe
taskkill /F /IM python3.exe
```
Nessun filtro per command line, titolo finestra o percorso — uccide **ogni** `python.exe`/`python3.exe` in esecuzione sul sistema, di qualunque progetto, non solo quelli di ARIA. Confermato dal vivo correlando i timestamp: `watchdog.log` di Sniper si interrompe di netto esattamente nel secondo in cui `main_tray.py`/`dashboard/server.py` di ARIA vengono ricreati da un riavvio di `aria.bat` — e da quel momento nessun `python.exe` del watchdog risulta più in esecuzione, perché nulla lo resuscita (il watchdog protegge il daemon Sniper, ma nessuno protegge il watchdog).
**Nota:** distinto da [[A1-5]] — quello riguarda i kill *runtime* (swap/idle-timeout tra backend, già scoping-ati per PID/titolo, solo il *bersaglio* scoperto poteva essere sbagliato). Questo è l'unico punto in tutta la codebase ARIA con un kill genuinamente non filtrato.
**Fix scritto (2026-09-24):** sostituito con una pulizia scoping-ata via PowerShell, che colpisce solo i `python.exe`/`python3.exe` la cui command line contiene `%ARIA_ROOT%` (qualunque script sotto `aria_node_controller/`, `backends/`, `dashboard/`):
```powershell
Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'python.exe' -or $_.Name -eq 'python3.exe') -and $_.CommandLine -like ('*' + $Env:ARIA_ROOT + '*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
```
Stesso effetto di pulizia zombie per ARIA, zero impatto su processi Python di altri progetti (Sniper incluso).
**Ancora da verificare dal vivo:** deploy su PC 139 e conferma che un riavvio di `aria.bat` non tocchi più processi Python estranei.

---

## Gap Risolti

### A0-6 — Backend legati a 127.0.0.1 — irraggiungibili da IP esterno
**Stato:** resolved
**Risolto:** 2026-05-17
**Descrizione:** Tutti i backend FastAPI/uvicorn su PC139 erano in listen su `127.0.0.1:{PORT}`. Dopo la fix health-check Firebase Studio (orchestratore usa `self.local_ip` = `192.168.1.139` invece di `localhost`), le richieste arrivavano all'IP esterno e venivano rifiutate con `Connection refused`. Il server bound a loopback non accetta connessioni dall'IP di rete.
**Backend interessati:** FLUX 8092, WhisperX 8091, Lifelog ASR 8087, Audiocraft 8086, AceStep wrapper.
**Soluzione:** migrati tutti a `host="0.0.0.0"` (uvicorn) o `--host 0.0.0.0` (argparse). Fix applicato via script `fix_hosts.py` distribuito su PC139. Commit `ca25b05` (ARIA master).

---

### A0-5 — Nessun backend image generation su ARIA
**Stato:** resolved
**Risolto:** 2026-05-16
**Descrizione:** ARIA non aveva alcun backend per la generazione di immagini. Lifelog2 Stage G richiedeva un modello text-to-image per generare cover degli episodi.
**Soluzione adottata:**
- Env `flux-aria` (Python 3.12, PyTorch 2.x+cu128)
- Modello: `FLUX.2-klein-4B` (~15.8 GB VRAM)
- Backend `backends/flux_imagegen/server.py` su porta 8092 (binding `0.0.0.0`)
- Coda Redis: `aria:q:imagegen:local:flux2-klein-4b:lifelog`
- Callback: `image_url = http://192.168.1.139:8082/{job_id}.jpeg` (asset server), `job_id` nel payload output
- Endpoint `DELETE /output/{filename}` per cleanup locale post-download
- Timing: ~10-11s/immagine (512×512, 20 steps), VRAM 12.8 GB
- E2E testato: 12/12 cover episodi generate e caricate su MinIO (2026-05-17)

---

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
