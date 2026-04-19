# Analisi Critica e Strategia di Ripristino: ARIA ACE-Step 1.5 XL

Questo documento sintetizza le ragioni del fallimento tecnico della pipeline di generazione musicale a staffetta (Relay) e delinea la strategia per implementare una generazione long-form (>180s) stabile e di alta qualità.

## 1. Analisi dei Fallimenti (Post-Mortem)

### A. Crash di Sistema (FileExistsError)
Il Wrapper crashava sistematicamente nella fase di spostamento dei file su Windows alla fine del primo chunk (pt0).
- **Causa**: L'uso di `pathlib.Path.rename()` su Windows solleva un errore se il file di destinazione esiste già. Il Wrapper cercava di spostare lo `score.json` ma trovava un residuo precedente.

### B. Blocco dell'Interattività (Il Buco di 15 Minuti)
È stato rilevato un ritardo massiccio (~15 min) tra la fase di "Thinking" e la generazione effettiva.
- **Causa**: Il file `cli.py`, pur invocato da uno script, si ferma in attesa di un input manuale (`Press Enter`) se la modalità `thinking` è attiva e non trova un file di istruzioni pre-caricato. La pipeline rimane "appesa" finché non scade il timeout.

### C. Deriva Qualitativa (Prompt Drift)
L'audio finale prodotto era di scarsa qualità ("orrendo") nonostante un prompt tecnico molto preciso.
- **Causa**: Il parametro `use_cot_caption` è attivo di default. L'LM "ripensa" il prompt dell'utente (es. "ARP 2600 synth, dissonant brass") e lo semplifica drasticamente (es. "A somber cinematic piece"), perdendo tutta la tessitura timbrica richiesta. Il DiT genera quindi sulla base di questa versione semplificata.

---

## 2. Scoperte dalla Documentazione Locale (`CLI.md`, `INFERENCE.md`)

Dall'audit della documentazione ufficiale del backend sono emersi i seguenti punti chiave per l'automazione:

1.  **Meccanismo di Bypass**: Esiste una funzione di override: se è presente un file chiamato `instruction.txt` nel root del processo, il CLI lo carica e **prosegue automaticamente senza stop interattivi**.
2.  **Essenzialità del Thinking**: Il `thinking` **NON deve essere disattivato** per tracce lunghe. Senza di esso, il modello DiT non riceve i "sound codes" guida e deraglia dopo circa 30 secondi, producendo rumore statico ciclico.
3.  **Parametri CoT**: L'API supporta i campi `cot_bpm`, `cot_keyscale`, `cot_duration`. Possiamo usarli per garantire che ogni chunk erediti perfettamente il DNA del chunk precedente.

---

## 3. Strategia di Risolvement definitiva (Indicazioni per ARIA)

### Azione 1: Pre-generazione dell'Instruction (Shielding)
Il Wrapper deve essere aggiornato per **costruire fisicamente** il file `instruction.txt` prima di ogni chiamata al CLI.
- Il file deve contenere il prompt tecnico originale avvolto nei tag corretti (`<|im_start|>user...`).
- Questo istruirà il CLI ad andare avanti senza attese, usando i dati esatti dello script.

### Azione 2: Modalità "Non-Creative" (Strict Prompt)
Impostare `use_cot_caption = False` nel TOML generato.
- Questo obbliga l'LM a generare i sound codes basandosi sul prompt tecnico dell'utente, senza parafrasarlo o semplificarlo.

### Azione 3: Robustezza del File System
- Sostituire ogni comando di spostamento/rinomina con `os.replace()`, che garantisce la sovrascrittura fluida anche su Windows.

### Azione 4: Sincronizzazione Temporale Adattiva
- Continuare a usare la durata reale del WAV prodotto per calcolare lo start del chunk successivo, ma iniettare la `cot_duration` corretta per ogni segmento nel nuovo file `instruction.txt`.

---

## Conclusione
ACE-Step 1.5 XL ha una logica interna molto potente (Thinking) che però è pensata per uso interattivo. Per renderlo una "macchina da staffetta" nel contesto di ARIA, dobbiamo passare da un'invocazione passiva del CLI a una gestione attiva dell'environment (file `instruction.txt`) per ogni singolo chunk.
