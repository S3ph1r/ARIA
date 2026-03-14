# ARIA Hybrid TTS Architecture & Voice Routing

Questo documento descrive l'architettura ibrida sviluppata per il sistema di sintesi vocale (TTS) in ARIA, concepita per orchestrare in modo totalmente trasparente diversi backend (come Fish-Speech e Qwen3-TTS) esponendo a DIAS un'interfaccia unificata e agnostica basata su code Redis e "Voice ID" astratti.

---

## 1. La Filosofia Architetturale: L'Orchestratore

ARIA **non è** un server TTS monolitico. ARIA è un "Nodo di Computazione AI Distribuita".
Il componente chiave, in esecuzione su LXC 190 (`orchestrator.py`), agisce da *Vigile Urbano*.

L'Orchestratore ascolta costantemente una serie di code (Code Redis) predefinite. A seconda della coda in cui l'applicativo client (es. DIAS) immette il task, ARIA risveglia il backend Python corrispondente (su PC Gaming GPU locale o in cloud), inoltra la computazione in modo specifico per quel modello LLM, attende il WAV risultante, e lo rende disponibile al client.

### Nomi e Funzione delle Code Redis
Il routing avviene **esclusivamente decidendo il nome della coda**:
- Per usare **Fish Audio** (ideale per voci espressive, emotion-markers tra parentesi):
  `gpu:queue:tts:fish-s1-mini`
- Per usare **Qwen3 TTS** (ideale per voci calde, stabili, stile audiolibro):
  `gpu:queue:tts:qwen3-tts-1.7b`

*Esempio futuro: se introdurremo un modello Llama3 per la generazione testi, i task verranno semplicemente inseriti nella coda `gpu:queue:llm:llama3-8b`.*

---

## 2. Astrazione Geografica: Il "Vocabolario" (Voice Library)

Il principio cardine è: **DIAS (o qualsiasi client) non deve mai conoscere i percorsi di rete (Path Windows o Linux) né le peculiarità tecniche (padding, chunking) richieste dai singoli modelli LLM vocali.**

Il client invia nel payload solo l'ID astratto della voce desiderata:
```json
{
  "voice_id": "luca",
  "text": "Il sole sorgeva lentamente all'orizzonte."
}
```

### Struttura della Voice Library su ARIA (PC Gaming)
ARIA possiede una cartella fisica detta "Vocabolario" (`%ARIA_ROOT%\data\voices`).
Al suo interno, ogni voce corrisponde a una sottocartella che contiene gli "Asset ICL" (In-Context Learning) necessari ai modelli per clonare la voce:

```text
voices/
├── luca/
│   ├── ref.wav            (L'audio base estratto, usato da Fish)
│   ├── ref_padded.wav     (Versione con +0.5s di silenzio, usato da Qwen3)
│   └── ref.txt            (La trascrizione esatta di cosa dice il ref.wav)
├── angelo/
│   ├── ref.wav
│   ├── ref_padded.wav
│   └── ref.txt
```

### Come i Backend gestiscono i Voice ID
Quando arriva il task `"voice_id": "luca"`:

1. **Se la coda è Fish (`fish_tts.py`)**:
   Il backend ARIA carica automaticamente `voices/luca/ref.wav` e `voices/luca/ref.txt` convertendoli nei formati e payload HTTP specifici per il demone Fish su porta 8080.
2. **Se la coda è Qwen3 (`qwen3_tts.py`)**:
   Il backend ARIA interroga il server FastApi Qwen3 (porta 8083) passandogli la directory radice. Sarà **esclusivamente** il backend Server di Qwen a caricare `ref_padded.wav` e `ref.txt`.

---

## 3. Gestione Intelligente dei Difetti LLM (Auto-Padding & Auto-Chunking)

I moderni modelli TTS autoregressivi hanno difetti congeniti che ARIA risolve automaticamente (Auto-Healing) sollevando le applicazioni client (DIAS) da ogni onere:

### IL PROBLEMA QWEN3: Bleeding Fonetico
**Difetto**: Qwen3, essendo autoregressivo rigido, se riceve un sample audio di base (`ref_audio`) che si interrompe di colpo, tende a propagare l'impronta finale (un respiro mozzo o uno schiocco) a tutti i successivi token generati, rovinando l'audio.
**Soluzione (Auto-Padding)**: Qwen3 Server, all'arrivo di una richiesta per `luca`, cerca `luca/ref_padded.wav`. Se *non* esiste (perché l'utente ha appena aggiunto la cartella di una nuova voce fornendo solo `ref.wav`), il Server invoca automaticamente in background `ffmpeg` aggiungendo 0.5 secondi di silenzio totale (Padding), lo salva e da quel momento utilizza solo quello.

### IL PROBLEMA GENERALE: Overload della Finestra di Contesto
**Difetto**: Se passiamo in un unico colpo un paragrafo di testo di 1000 parole a Qwen3 o Fish, la VRAM della RTX 5060 esplode (Out Of Memory) o il modello collassa allucinando e ripetendo sillabe.
**Soluzione (Auto-Chunking)**: Entrambi i backend server (sia Fish che Qwen3) implementano internamente uno spatial-chunking. Se la richiesta HTTP contiene un romanzo:
1. Lo dividono automaticamente sui punti (`. ? !`) in sottomultipli di sicurezza (es. max 250 parole).
2. Generano gli `.wav` separati per ogni sottomultiplo.
3. Li concatenano in un singolo file WAV finale immettendo `80 millisecondi` fisiologici di silenzio tra le frasi ("respiro").
Il client ottiene 1 solo file WAV perfetto a prescindere dalla larghezza in input.

### Gestione Naming (Determinismo)
Per workflow di produzione narrativa (DIAS), ARIA adotta una politica di **Naming Coerente**: il client può imporre il nome del file finale inviando un `job_id` descrittivo. Questo trasforma l'output di ARIA da un semplice buffer temporaneo a un asset persistente rintracciabile deterministicamente via HTTP, eliminando la necessità di database di mappatura complessi tra i due sistemi.

---

## 4. In-Context Learning (ICL) Dinamico

Un tempo (es. Vall-E antico) registravi una voce e generavi 1 "vettore pesi" da usare a vita.
Oggi, sia Qwen3 che Fish operano in puro **Zero-Shot / In-Context Learning (ICL)**: ad *ogni singola richiesta API*, il modello riceve nello stesso payload sia l'audio della persona che parla (`ref`), sia il testo nuovo da creare.

L'**ICL perfetto** si raggiunge quando oltre all'audio di base forniamo al modello la trascrizione esatta (`ref.txt`) di quell'audio. Questo permette l'ancoraggio (grounding) fonetico.
Se ARIA trova un file `ref.txt` accanto al `.wav`, **obbliga** il server in modalità ICL ad altissima fedeltà. Se il file `.txt` manca (utente disattento), ARIA instrada in automatico il motore in degradazione verso lo "Zero-Shot Voice Only" mode.

---

## 5. Come aggiungere una nuova voce al sistema (Onboarding)

Aggiungere un nuovo attore per DIAS richiede *solo* 1 azione manuale:

1. Modifica o crea un mini-script bash invocando il tool ufficiale (che gira sul Mini-PC/LXC):
   ```bash
   python /home/Projects/NH-Mini/sviluppi/ARIA/scripts/voice_prepper.py "https://youtube.com/watch?v=LINK_MIO_ATTORE" "isabella" --start 00:15 --end 00:27
   ```
2. Lo script `voice_prepper` eseguirà automaticamente:
   - Download con `yt-dlp`
   - Clipping con `ffmpeg`
   - Trascrizione con `Gemini Flash 2.5 AI`
   - Auto-creazione di `ref.wav`, auto-padding in `ref_padded.wav`, auto-salvataggio di `ref.txt`
   - Spostamento di tutto nella `/voices/isabella/`

Da questo preciso istante, la stringa `"voice_id": "isabella"` diviene universalmente riconosciuta e operabile in tutto l'ecosistema DIAS/ARIA sia verso Fish che verso Qwen3.
