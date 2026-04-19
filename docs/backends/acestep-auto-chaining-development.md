# Sviluppo: Auto-Chaining per Task Lunghi in ACE-Step

Questo documento descrive in dettaglio le specifiche da seguire per l'implementazione della funzionalità di "Auto-Chaining", che permetterà ad ACE-Step di superare i limiti nativi legati all'uso della VRAM, consentendo la generazione di tracce musicali ininterrotte lunghe oltre i 5 minuti (es. per interi capitoli).

## Obiettivo
Gestire tracce audio di lunga durata (> 5 minuti) superando i limiti fisici della VRAM su GPU da 16GB (come la RTX 5060 Ti). 
La soluzione tecnica prevede che il Wrapper API ripartisca internamente il task in "chunk" più piccoli (es. 4 minuti) eseguendoli in sequenza, e iniettando gli ultimi 30 secondi dell'audio del *Chunk N-1* come `reference_audio` per il *Chunk N*. Questo assicura continuità timbrica, armonica e stilistica.

## Stato Attuale (Pre-Chaining)
1. L'Orchestratore (da Stage B2) invia un payload JSON con `duration: X` al Wrapper (`aria_wrapper_server.py`).
2. Il Wrapper crea un file `.toml` corrispondente e chiama `cli.py`.
3. Se `duration` è eccessiva (es. > 300 secondi / 5 minuti):
    - Il processo del DiT XL satura la VRAM (OOM Error).
    - Oppure, se si abbassa drasticamente la risoluzione, la generazione perde l'attinenza stilistica nella seconda metà del brano (Drifting).
4. Attualmente il Wrapper si aspetta che la `duration` inviata sia nei limiti (es. 120-180s come da policy Hyperion).

## Nuovo Comportamento Desiderato (Post-Chaining)

Il **Wrapper API** si espanderà da semplice "esecutore di task" a vero e proprio **Manager del Chaining**:

1. **Controllo del Limite Superiore**:
    - Nel Wrapper verrà definita una costante `MAX_CHUNK_DURATION = 240.0` (4 minuti).
    - Quando riceve una richiesta con `duration` (es. 600 secondi), il Wrapper determina quanti split sono necessari (es. 240s + 240s + 120s).

2. **Esecuzione Chunk 1**:
    - Il Wrapper crea un `chunk_1.toml` con `duration = 240`.
    - Spawna `cli.py`. Terminato, salva temporaneamente `chunk_1.wav`.

3. **Esecuzione Chunk Successivi (Auto-Injection)**:
    - Il Wrapper crea `chunk_2.toml` con `duration = 240`, ma aggiunge nel TOML la direttiva `reference_audio = "chunk_1.wav"`.
    - `cli.py` eseguirà la generazione forzando il condizionamento Cross-Attention sul DNA del riferimento (in genere gli ultimi 30s) per generare il seguito.
    - Se presente `lyrics` (roadmap di pacing), questa viene correttamente scalata da Qwen3 LM.
    - Così per gli N chunk necessari.

4. **Merge (Ricongiungimento) e Consegna**:
    - Una volta generati tutti i frammenti (`chunk_1.wav`, `chunk_2.wav`, ecc.), il Wrapper utilizza librerie standard come il modulo nativo `wave` di Python o `pydub` installabile in `dias-sound-engine`, per cucire gli spezzoni.
    - Un singolo file unificato `job_id.wav` viene depositato nel Warehouse in corrispondenza del task originario.
    - I file `.wav` temporanei vengono cancellati.
    - L'Orchestratore viene informato che il `job_id` è completo e non rileva alcun cambiamento rispetto al proprio flusso. Gestione totalmente "Zero-Touch" per lo Stage B2.

## Modifiche Architetturali da Effettuare

### 1. `backends/acestep/aria_wrapper_server.py`
Questo sarà il **solo** file che accoglierà logica core. Modifiche previste in `_run_task`:
*   Introdurre un blocco loop `while remaining_duration > 0:`.
*   Creare naming customizzati per i `.toml` in itinere (es. `<id>_chunk_<i>.toml`).
*   Monitorare `previous_audio_path` tra un ciclo e l'altro.
*   Introdurre un blocco finale di Merge Audio:
    ```python
    # Pseudo-codice potenziale usando modulo d'appoggio:
    # merged_audio = merge_wavs(chunk_wav_paths)
    # save_audio(merged_audio, final_audio_path)
    # delete_temp(chunk_wav_paths)
    ```

### 2. Nessuna Modifica a `cli.py` (Backend ACE-Step)
La funzionalità di inputamento di `reference_audio` in `text2music` task è già supportata da ACE-Step v1.5 nativamente (utilizzato cross-attention da Qwen3 per generare Audio Codes in sequenza logica dal reference e gestito poi dal DiT XL). Si riutilizza quello strumento as-is.

### 3. Nessuna Modifica al Connettore (`aria_node_controller/backends/acestep.py`)
Dal punto di vista dell'Orchestratore ARIA, lo splitting, execution multipla e merge avvengono nell'underworld. Il connettore Python esegue una HTTP POST a `/generate` con timeout di 7200s, ed è ben disposto ad attendere tutto il tempo dell'auto-chaining pur di avere `job_id.wav` in ritorno.

## Avvertenze Tecniche per Implementazione Futura
*   Il tempo da computare complessivo (A -> TOML -> cli -> WAV) scalerà linearmente col numero di chunk. Assicurarsi di impostare i vari timeout a livello adeguato in Production.
*   Lo shift dei `lyrics` timestamps per le tracce multiple richiederà che, se lo Stage B2 include i tag temporali (es `[06:00 - ... ]`), il TOML per il chunk N deve estrarre/rimodulare i timestamp correntemente associati al proprio intervallo (es `[02:00 - ... ]` rispetto allo start _locale_ del chunk). Questo task delegato potrebbe spingere l'editing per estrarre porzioni di parsing. Per la release v1 dell'Auto-chaining si consiglia generazione senza strict timestamps in loop > 5 minuti.

**(Documento Pronto per prossima esecuzione di implementazione)**
