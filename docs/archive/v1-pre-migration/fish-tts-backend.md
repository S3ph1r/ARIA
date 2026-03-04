# Fish Audio S1-mini — Backend TTS per ARIA
## Sostituzione di Orpheus: Filosofia, Dettagli Tecnici e Roadmap

> **Contesto**: Questo documento sostituisce `backends/orpheus.py` con
> `backends/fish_tts.py` nell'architettura ARIA. Tutto il resto — BatchOptimizer,
> Redis, Samba, semaforo, DIAS — resta invariato. Solo il backend TTS cambia.

---

## 1. PERCHÉ FISH AUDIO S1-MINI

### Il problema con Orpheus italiano

Orpheus 3B è un modello eccellente in inglese. Il fine-tune italiano
(`canopylabs/3b-es_it-ft`) è un modello di ricerca acerbo: addestrato su
pochi dati italiani, con un tokenizer che ha problemi di compatibilità con i
token audio SNAC, e una pipeline di decodifica (finestre mobili sovrapposte)
che introduce artefatti sistematici. Il risultato è una voce piatta, con
consonanti allungate e mancanza di dinamica emotiva.

Oltre al modello, c'è il problema infrastrutturale: Orpheus richiede
llama.cpp nativo Windows perché il mapping GGUF sulla VRAM attraverso WSL2
fallisce su Blackwell sm_120. Questo significa un processo Windows separato,
uno script di avvio Task Scheduler, e una catena di chiamate HTTP fragile.

### Perché Fish Audio S1-mini risolve entrambi i problemi

**Qualità**: addestrato su 2 milioni di ore di audio con RLHF (Reinforcement
Learning from Human Feedback) — correzione iterativa da valutatori umani sulla
prosodia. Architettura Dual-AR che modella semantica e acustica in un singolo
modello, eliminando gli artefatti dei pipeline "semantic-only". Ranked #1 su
TTS-Arena-V2 (valutazione umana blind test). WER italiano <1%.

**Infrastruttura**: usa PyTorch nativo, non GGUF. Gira come processo Python
diretto con CUDA 12.8 su Windows — niente llama.cpp, niente problemi di
linker, niente WSL2. Si integra in ARIA come backend Python standard, dentro
o fuori Docker, con la stessa interfaccia di qualsiasi altro backend.
> ⚠️ **Limiti Architetturali Noti (RTX 5000 / Blackwell sm_120) & Soluzione**: 
> Le build wheel ufficiali di PyTorch incluse in conda tramite i comandi standard spesso non supportano l'architettura `sm_120` prodotta nel 2025. Il modulo VQGAN (Voice Cloning) con `DEVICE="cuda"` crascia istantaneamente con errore 500 `CUDA error: no kernel image is available for execution on the device`.
> **SOLUZIONE DEFINITIVA**: Installare PyTorch stabile 2.7+ con supporto a CUDA 12.8 direttamente dal repository specifico di PyTorch.
> Per l'ambiente `fish-voice-cloning`:
> ```cmd
> call conda activate fish-voice-cloning
> pip uninstall torch torchvision torchaudio -y
> pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
> ```
> In questo modo, l'encoding VQGAN può operare su `"cuda"` senza crash. (Se non si ha l'ambiente aggiornato, si è obbligati a forzare `voice_cloning_server.py` in `DEVICE="cpu"`).

**Italiano supportato nativamente**: Fish Audio S1 è nell'elenco ufficiale
delle lingue supportate con emotion markers. I tag emotivi funzionano in
italiano, non solo in inglese.

---

## 2. ARCHITETTURA TECNICA DI FISH AUDIO S1

### Come funziona internamente (Dual-AR)

Fish Audio S1 usa un'architettura a due livelli chiamata Dual-AR:

```
TESTO INPUT
    │
    ▼
┌─────────────────────────────────┐
│  SLOW TRANSFORMER (semantico)   │  ← capisce il significato, il contesto,
│  "cosa dire e come dirlo"       │    le emozioni, il ritmo narrativo
└─────────────────┬───────────────┘
                  │ token semantici
                  ▼
┌─────────────────────────────────┐
│  FAST TRANSFORMER (acustico)    │  ← traduce in caratteristiche vocali:
│  "come suona concretamente"     │    timbro, frequenza, velocità, pausa
└─────────────────┬───────────────┘
                  │ token audio
                  ▼
┌─────────────────────────────────┐
│  DECODER (DAC codec)            │  ← decodifica token → PCM audio
│  "waveform finale"              │
└─────────────────────────────────┘
                  │
                  ▼
              WAV OUTPUT
```

La differenza rispetto a Orpheus: in Orpheus il decoder SNAC riceve token da
una singola testa semantica, con il rischio di disallineamento tra semantica
e acustica. In Fish S1 i due transformer lavorano in parallelo e si
condizionano a vicenda, producendo una voce che "capisce" il testo invece di
limitarsi a sintetizzarlo.

### Perché la qualità emotiva è superiore

Il training con RLHF significa che valutatori umani hanno ascoltato migliaia
di generazioni e dato feedback su naturalezza, prosodia e emozione. Il modello
ha imparato a correggere i pattern meccanici che rendono il TTS "artificiale".
Non è una questione di parametri — è una questione di dati di training e
processo di ottimizzazione.

ElevenLabs v3 fa la stessa cosa ma con ordini di grandezza più dati e
valutatori. S1-mini è la cosa open source più vicina a quel livello.

### Confronto diretto con Orpheus italiano

| Aspetto | Orpheus ITA | Fish S1-mini |
|---------|-------------|--------------|
| Parametri | 3B (GGUF Q8) | 0.5B (PyTorch) |
| VRAM richiesta | ~4GB (Q8) | ~3GB |
| Training italiano | Fine-tune limitato | Nativo, 2M ore |
| Emotion markers | 8 tag (`<gasp>` etc.) | 50+ tag (`(scared)` etc.) |
| RLHF | No | Sì (GRPO online) |
| Architettura | Single-AR + SNAC | Dual-AR + DAC |
| Artefatti noti | Word-skipping, SNAC overlap | Nessuno documentato |
| Infrastruttura | llama.cpp nativo Windows | PyTorch nativo Windows |
| RTF su RTX 5060 Ti | ~1:2 stimato | ~1:5 stimato |
| Integrazione ARIA | External backend (HTTP proxy) | Native backend (Python diretto) |

---

## 3. EMOTION MARKERS — LISTA COMPLETA PER ITALIANO

Fish Audio S1 supporta emotion markers in italiano. La sintassi è
`(marker)testo` oppure `testo (marker) continuazione`.

### Emozioni di base

```
(angry)       (sad)         (excited)     (surprised)   (satisfied)
(delighted)   (scared)      (worried)     (upset)       (nervous)
(frustrated)  (depressed)   (empathetic)  (embarrassed) (disgusted)
(moved)       (proud)       (relaxed)     (grateful)    (confident)
(interested)  (curious)     (confused)    (joyful)
```

### Emozioni complesse (utili per narrativa)

```
(disdainful)  (unhappy)     (anxious)     (hysterical)  (indifferent)
(impatient)   (guilty)      (scornful)    (panicked)    (furious)
(reluctant)   (keen)        (disapproving)(sarcastic)   (conciliative)
(comforting)  (sincere)     (sneering)    (hesitating)  (yielding)
(painful)     (awkward)     (amused)      (serious)
```

### Limite di Contesto e Chunking Automatico
Fish S1-mini ha una lunghezza macchina fissata a `max_seq_len = 8192` (circa 1024 token VQGAN audio in output). 
Per limitare i troncamenti improvvisi dell'audio per scene lunghe, il **Node Orchestrator** di ARIA incorpora un sistema di **Chunking Trasparente**:
1. Il testo ricevuto (anche fino a 300 parole) viene splittato usando i segni di interpunzione finale (`.` `!` `?` `...`) per non superare le **~150 parole per elaborazione**.
2. Il server Fish processa in modo iterativo ogni chunk. (Grazie al `memory_cache` attivo in Fish, il timbro vocale rimane costante senza bisogno di ricaricare il sample di riferimento per ogni frase).
3. Il `Node Orchestrator` riconcatena fisicamente i WAV prodotti, inviando a DIAS (o al client HTTP) un **singolo file WAV** completo.

### Effetti paralinguistici

```
(laughing)        (chuckling)       (sobbing)
(crying loudly)   (sighing)         (panting)
(groaning)        (crowd laughing)  (background laughter)
(audience laughing)
```

### Confronto tag Orpheus → Fish per DIAS TextDirector

| Tag Orpheus (vecchio) | Tag Fish equivalente | Note |
|----------------------|---------------------|------|
| `<gasp>` | `(scared)` o `(surprised)` | Dipende dal contesto |
| `<sigh>` | `(sighing)` | Effetto diretto |
| `<laugh>` | `(laughing)` | Effetto diretto |
| `<chuckle>` | `(chuckling)` | Effetto diretto |
| `<groan>` | `(groaning)` | Effetto diretto |
| `<cough>` | nessun equivalente esatto | Usa `(awkward)` |
| `<yawn>` | nessun equivalente esatto | Usa `(relaxed)` |
| nessuno | `(panicked)` | Nuovo — molto utile |
| nessuno | `(sarcastic)` | Nuovo — molto utile |
| nessuno | `(hesitating)` | Nuovo — perfetto per dialoghi |
| nessuno | `(whispering)` | Non in lista ufficiale ma funziona |

### Esempi di utilizzo per audiolibri italiani

```python
# Scena di tensione narrativa
"(nervous)La porta era socchiusa. (hesitating)Dovevo entrare... (scared)ma qualcosa mi fermava."

# Dialogo con sarcasmo
"(sarcastic)Certo, tutto perfetto. (disdainful)Come sempre."

# Momento emotivo
"(moved)Non me lo aspettavo. (sincere)Grazie, davvero."

# Azione drammatica
"(panicked)Corri! (furious)Non c'è tempo!"

# Narrazione distaccata con riflessione
"(serious)Era il 2087. (relaxed)Il mondo era cambiato, ma le persone no."
```

### Come usarli nel copione DIAS

Il TextDirector di DIAS dovrà essere aggiornato con un prompt che istruisce
Gemini a usare la sintassi Fish invece dei tag Orpheus. La modifica è minimale:

```
# Vecchio prompt TextDirector (Orpheus)
"Inserisci tag emotivi nel formato <gasp>, <sigh>, <laugh>..."

# Nuovo prompt TextDirector (Fish - Ottimizzato)
"Inserisci emotion markers ufficiali nel formato (scared), (sighing), (serious)...
 Posizionali PRIMA della parola o frase che devono colorare emotivamente.

 REGOLE DI QUALITÀ:
 1. Usa la PUNTEGGIATURA DRAMMATICA: inserisci '...' per micro-pause nel testo.
 2. PAUSE TITOLI: per titoli di libri/capitoli, usa '... .' subito dopo per forzare il silenzio.
 3. MAI usare staffe quadre [Instruction] o tag non ufficiali.
 4. Scegli tra: (serious), (sincere), (whispering), (scared), (hesitating), (indifferent), (angry), (sad)."
```

---

## 4. PARAMETRI DI INFERENZA

### Avvio del server API Fish

```bash
# Windows nativo (conda environment fish-speech, Python 3.10)
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "C:/models/fish-s1-mini" \
    --decoder-checkpoint-path "C:/models/fish-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq \
    --compile

# Nota sull'Architettura Conda Unificata (Dal 03/03/2026)
Entrambi i server ora vengono eseguiti sotto un UNICO ambiente Conda (`fish-speech`)
per semplificare la gestione delle dipendenze PyTorch sm_120 e ridurre il footprint.
- **Porta 8080**: TTS Synthesis (Fish API Server standard)
- **Porta 8081**: VQGAN Encoder (Voice Cloning Server custom)
- **Porta 8082**: Asset HTTP Server (per il download dei WAV generati, integrato nel Node Controller)
```

Il flag `--compile` abilita `torch.compile` — la prima esecuzione è lenta
(~2 minuti di compilazione JIT), ma le successive sono 2-3x più veloci.

### Parametri chiamata API (per `fish_tts.py`)

```python
payload = {
    "text": "(nervous)La porta era socchiusa...",
    "references": [
        {
            "audio": base64_encoded_reference_wav,  # DEVE essere il WAV originale (non token NPY) per il voice cloning
            "text": "testo corrispondente al reference audio"
        }
    ],
    "reference_id": None,       # alternativo a references
    "normalize": False,         # CRITICO: DEVE essere False per far funzionare i tag emotivi e i (break)
    "format": "wav",            # wav | mp3 | opus
    "mp3_bitrate": 64,          # solo se format=mp3
    "opus_bitrate": -1000,      # solo se format=opus
    "latency": "normal",        # normal | balanced
    "streaming": False,         # False per batch processing ARIA
    "use_memory_cache": "off",  # on | off (on = riusa reference encodings)
}
```

### ⚠️ Bug Noto: First-word cutoff (Le prime parole mangiate)
Fish Speech S1-mini soffre di un bug noto ([Issue #881](https://github.com/fishaudio/fish-speech/issues/881)) dove l'audio generato "taglia" la primissima parola di un blocco di testo.
**Workaround in ARIA**: L'Orchestrator inietta automaticamente un tag `(break)` (es. `"(break) testo originale"`) all'inizio di ogni chunk di testo inviato all'API. Il glitch si "mangia" il silenzio del break, salvando le parole reali.

### Parametri di qualità consigliati per audiolibri

```python
# Per narrazione lunga (capitoli interi)
{
    "normalize": False,         # IMPORTANTISSIMO: False per non strippare i tag emotivi 
    "format": "wav",            # qualità massima, ARIA non ricomprime
    "latency": "normal",        # qualità > velocità per batch offline
    "streaming": False,         # batch completo, non streaming
    "use_memory_cache": "on",   # riusa reference encoding tra scene stesso personaggio
    "temperature": 0.7,         # Valore ottimale per stabilità (evita distorsioni metalliche)
    "top_p": 0.7                # Bilanciamento tra varietà e coerenza
}
```

### ⚠️ Requisito Critico: Accento Italiano (Voice Cloning)

Il modello Fish S1-mini è multilingua, ma tende a defaultare su un accento inglese se non guidato correttamente. Per garantire un accento italiano nativo durante il voice cloning:

1.  **voice_ref_text Preciso**: Il campo `voice_ref_text` DEVE contenere la trascrizione esatta, parola per parola, del file audio di riferimento (`narratore.wav`).
2.  **Allineamento**: Se il testo di riferimento non corrisponde all'audio, il modello non riesce ad "agganciare" la fonetica italiana e produrrà una voce con accento inglese o robotica.
3.  **Esempio Narratore**: Per il file `narratore.wav` standard (Ezechiele 25:17), il testo di riferimento deve essere:
    > "Leggi la Bibbia, Brett? E allora ascolta questo passo che conosco a memoria, è perfetto per l'occasione: Ezechiele 25:17. Il cammino dell'uomo timorato è minacciato da ogni parte dalle iniquità degli esseri egoisti e dalla tirannia degli uomini malvagi. Benedetto sia colui che nel nome della carità e della buona volontà conduce i deboli attraverso la valle"

### Gestione Voice Cloning — Voice Library (ARIA-side)

Fish S1-mini fa voice cloning zero-shot con 10-30 secondi di audio di riferimento. 
Invece di far passare i path al client, ARIA gestisce una **Voice Library** locale 
sul nodo GPU Windows (`%ARIA_ROOT%\data\voices`).

#### Creazione Automatica di Nuovi Sample (Voice Prepper)
Per creare una nuova voce a partire da un video YouTube, utilizzare lo script dedicato `scripts/voice_prepper.py`. Questo strumento scarica, taglia l'audio, e utilizza Gemini per generare la trascrizione al volo.
Genera **simultaneamente** sia il `ref.wav` normale per Fish Audio, sia il `ref_padded.wav` (con 0.5s di silenzio aggiuntivo) necessario per Qwen3-TTS.

**Uso:**
```bash
python scripts/voice_prepper.py "https://youtube.com/watch?v=XYZ" "nome_voce" --start 00:15 --end 00:30
```

#### Struttura della Libreria Generata
Ogni voce (intent) corrisponde a una sottocartella:
```
%ARIA_ROOT%\data\voices\
├── luca/
│   ├── ref.wav             ← Usato da Fish TTS
│   ├── ref_padded.wav      ← Usato da Qwen3 TTS
│   └── ref.txt             ← Usato da entrambi (ICL)
```

#### Risoluzione Automatica
Quando il backend riceve un `voice_id`:
1. Verifica l'esistenza della cartella in `data/voices/{voice_id}/`.
2. Carica automaticamente `ref.wav` come audio di riferimento.
3. Carica automaticamente `ref.txt` come `voice_ref_text` (fondamentale per l'accento italiano).
4. Procede all'inferenza senza parametri aggiuntivi dal client.

Il backend Fish codifica il reference audio al primo utilizzo e lo cachea in memoria durante la sessione (`use_memory_cache: on`), evitando di ricodificarlo per ogni scena. In alternativa, il frontend DIAS può inviare il reference audio codificato in base64.

> **Fail-safe Accento**: L'orchestratore include un fallback automatico che inietta il testo di Ezechiele 25:17 se `voice_ref_text` è vuoto e viene richiesto il modello `fish-s1-mini`. Questo garantisce che il narratore mantenga sempre l'accento italiano anche in assenza di parametri espliciti dal client.

### Chunking per testi lunghi

Fish S1-mini non ha un limite fisso di caratteri come Orpheus, ma per
scene narrative molto lunghe (>500 parole) è consigliabile dividere
per paragrafo con crossfade da 50ms. Il backend gestisce questo
automaticamente prima di chiamare l'API.

---

## 5. INTEGRAZIONE CON ARIA — ARCHITETTURA

### Posizione nell'architettura ibrida

Fish S1-mini è un **Native Python Backend** — a differenza di Orpheus
che era un External Backend (chiamata HTTP a llama-server.exe nativo),
Fish gira come processo Python direttamente.

```
╔════════════════════════════════════════════════════════════════╗
║  PC GAMING (Windows 11)                                        ║
║                                                                ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │  DOCKER (Broker ARIA)                                    │  ║
║  │  ├── queue_manager.py  ← legge Redis minipc             │  ║
║  │  ├── batch_optimizer.py                                  │  ║
║  │  └── backends/fish_tts.py                               │  ║
║  │       └── chiama http://host.docker.internal:8080       │  ║
║  └──────────────────────────────────────────────────────────┘  ║
║                         │ HTTP                                 ║
║                         ▼                                      ║
║  ┌──────────────────────────────────────────────────────────┐  ║
║  │  WINDOWS NATIVO (Fish API Server)                        │  ║
║  │  python -m tools.api_server --listen 0.0.0.0:8080       │  ║
║  │  Processo Python con PyTorch CUDA 12.8                   │  ║
║  │  Accesso diretto GPU RTX 5060 Ti                         │  ║
║  └──────────────────────────────────────────────────────────┘  ║
╚════════════════════════════════════════════════════════════════╝
```

**Perché ancora nativo Windows invece di dentro Docker?**

Stesso motivo di llama.cpp: PyTorch con CUDA 12.8 su Blackwell sm_120 non
ha immagini Docker ufficiali stabili. La via più affidabile su RTX 5060 Ti
è Python nativo Windows con conda + PyTorch 2.7+cu128. Quando Docker
supporterà ufficialmente Blackwell, il backend potrà essere spostato
nel container senza modificare niente altro.

### Schema Redis — coda per Fish

```
INPUT:   gpu:queue:tts:fish-s1-mini
OUTPUT:  gpu:result:{client_id}:{job_id}
```

### Payload task da DIAS per Fish

```json
{
  "job_id": "uuid-v4",
  "client_id": "dias-minipc",
  "model_type": "tts",
  "model_id": "fish-s1-mini",
  "queued_at": "2026-02-26T10:00:00Z",
  "priority": 1,
  "timeout_seconds": 1800,
  "callback_key": "gpu:result:dias-minipc:uuid-v4",
  "file_refs": {
    "input": [
      {
        "ref_id": "voice_reference",
        "url": "http://192.168.1.120:8000/voices/narratore_it.wav" 
      }
    ]
  },
  "payload": {
    "text": "(nervous)La porta era socchiusa. (hesitating)Dovevo entrare...",
    "voice_ref": "voice_reference",
    "voice_ref_inline": "base64_encoded_audio...", // Alternativa all'url
    "voice_ref_text": "frase del reference audio per voice cloning",
    "output_format": "wav",
    "normalize": true,
    "use_memory_cache": "on"
  }
}
```

---

## 6. IMPLEMENTAZIONE — `backends/fish_tts.py`

### Struttura del modulo

```python
# aria_server/backends/fish_tts.py

import base64, requests, json, os, time
from pathlib import Path
from typing import Optional
from .base_backend import BaseBackend
from ..models import AriaTaskPayload, AriaTaskResult

FISH_API_URL = os.getenv("FISH_API_URL", "http://host.docker.internal:8080")

class FishTTSBackend(BaseBackend):
    """
    Backend TTS per Fish Audio S1-mini.
    Chiama il processo fish-api-server su Windows nativo via HTTP.
    load() = health check sull'API server
    unload() = no-op (il processo è gestito da Windows, non da ARIA)
    run() = chiamata HTTP → salva WAV su disco locale (es. C:\..\outputs) e ritorna URL HTTP
    """

    model_id = "fish-s1-mini"
    model_type = "tts"

    def __init__(self, config: dict):
        self.api_url = config.get("api_url", FISH_API_URL)
        self.timeout = config.get("request_timeout_seconds", 300)
        self._loaded = False
        self._voice_cache = {}  # ref_id → base64 encoded audio

    def load(self) -> None:
        """Health check sul fish-api-server. Fallisce se non risponde."""
        try:
            r = requests.get(f"{self.api_url}/v1/health", timeout=5)
            r.raise_for_status()
            self._loaded = True
            print(f"[fish_tts] API server raggiungibile su {self.api_url}")
        except Exception as e:
            raise RuntimeError(f"Fish API server non raggiungibile: {e}")

    def unload(self) -> None:
        """No-op: il processo Fish gira su Windows, non lo gestiamo noi."""
        self._loaded = False
        self._voice_cache.clear()

    def estimated_vram_gb(self) -> float:
        return 3.0  # S1-mini con torch.compile su sm_120

    def is_loaded(self) -> bool:
        return self._loaded

    def run(self, task: AriaTaskPayload) -> AriaTaskResult:
        import time
        start = time.time()

        payload = task.payload
        file_refs = task.file_refs or {}

        # 1. Risolvi path Samba
        output_ref_id = payload.get("output_ref", "audio_output")
        output_path = self._resolve_output_path(file_refs, output_ref_id)

        # 2. Carica voice reference (con cache)
        voice_ref_id = payload.get("voice_ref")
        reference = self._load_voice_reference(file_refs, voice_ref_id)

        # 3. Chunking se testo lungo
        text = payload.get("text", "")
        chunks = self._chunk_text(text, max_words=400)

        # 4. Genera audio per ogni chunk
        audio_chunks = []
        for i, chunk in enumerate(chunks):
            audio_bytes = self._generate_chunk(
                text=chunk,
                reference=reference,
                reference_text=payload.get("voice_ref_text", ""),
                normalize=payload.get("normalize", True),
                use_memory_cache=payload.get("use_memory_cache", "on"),
            )
            audio_chunks.append(audio_bytes)

        # 5. Merge chunk con crossfade 50ms se necessario
        if len(audio_chunks) > 1:
            final_audio = self._merge_with_crossfade(audio_chunks, sr=44100)
        else:
            final_audio = audio_chunks[0]

        # 6. Salva su Samba
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(final_audio)

        duration = len(final_audio) / (44100 * 2)  # approx WAV 16bit mono
        processing_time = time.time() - start

        return AriaTaskResult(
            job_id=task.job_id,
            client_id=task.client_id,
            model_type=self.model_type,
            model_id=self.model_id,
            status="done",
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            processing_time_seconds=round(processing_time, 2),
            output={
                "audio_ref": output_ref_id,
                "output_path": output_path,
                "duration_seconds": round(duration, 2),
                "chunks_generated": len(audio_chunks),
            }
        )

    def _generate_chunk(self, text, reference, reference_text,
                        normalize, use_memory_cache) -> bytes:
        """Chiama Fish API server per un singolo chunk di testo."""
        body = {
            "text": text,
            "normalize": normalize,
            "format": "wav",
            "latency": "normal",
            "streaming": False,
            "use_memory_cache": use_memory_cache,
        }
        if reference:
            body["references"] = [
                {"audio": reference, "text": reference_text}
            ]

        r = requests.post(
            f"{self.api_url}/v1/tts",
            json=body,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.content  # WAV bytes

    def _load_voice_reference(self, file_refs: dict,
                               ref_id: Optional[str]) -> Optional[str]:
        """Carica il file WAV di reference e lo codifica in base64."""
        if not ref_id:
            return None
        if ref_id in self._voice_cache:
            return self._voice_cache[ref_id]

        input_refs = file_refs.get("input", [])
        for ref in input_refs:
            if ref.get("ref_id") == ref_id:
                path = self._resolve_samba_path(ref["shared_path"])
                with open(path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                self._voice_cache[ref_id] = encoded
                return encoded
        return None

    def _resolve_output_path(self, file_refs: dict, ref_id: str) -> str:
        output_refs = file_refs.get("output", [])
        for ref in output_refs:
            if ref.get("ref_id") == ref_id:
                return self._resolve_samba_path(ref["shared_path"])
        raise ValueError(f"Output ref '{ref_id}' non trovato in file_refs")

    @staticmethod
    def _resolve_samba_path(shared_path: str) -> str:
        """Converte path minipc → path container Docker."""
        return shared_path.replace("/mnt/aria-shared", "/aria-shared")

    @staticmethod
    def _chunk_text(text: str, max_words: int = 400) -> list[str]:
        """Divide il testo in chunk da max_words parole, rispettando i paragrafi."""
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
        current = []
        current_words = 0

        for para in paragraphs:
            words = len(para.split())
            if current_words + words > max_words and current:
                chunks.append(" ".join(current))
                current = [para]
                current_words = words
            else:
                current.append(para)
                current_words += words

        if current:
            chunks.append(" ".join(current))

        return chunks if chunks else [text]

    @staticmethod
    def _merge_with_crossfade(audio_chunks: list[bytes],
                               sr: int = 44100,
                               crossfade_ms: int = 50) -> bytes:
        """Unisce chunk WAV con crossfade. Dipendenza: numpy + scipy."""
        import numpy as np
        import io, wave

        def wav_to_array(wav_bytes):
            with wave.open(io.BytesIO(wav_bytes)) as w:
                frames = w.readframes(w.getnframes())
                return np.frombuffer(frames, dtype=np.int16).astype(np.float32)

        def array_to_wav(arr, sample_rate=44100):
            buf = io.BytesIO()
            with wave.open(buf, "w") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(arr.astype(np.int16).tobytes())
            return buf.getvalue()

        crossfade_samples = int(sr * crossfade_ms / 1000)
        arrays = [wav_to_array(c) for c in audio_chunks]

        merged = arrays[0]
        for nxt in arrays[1:]:
            fade_out = np.linspace(1, 0, crossfade_samples)
            fade_in  = np.linspace(0, 1, crossfade_samples)
            merged[-crossfade_samples:] *= fade_out
            nxt[:crossfade_samples]     *= fade_in
            merged = np.concatenate([merged[:-crossfade_samples],
                                     merged[-crossfade_samples:] + nxt[:crossfade_samples],
                                     nxt[crossfade_samples:]])

        return array_to_wav(merged, sr)
```

---

## 7. CONFIGURAZIONE ARIA — `config.yaml`

```yaml
models:
  tts:
    fish-s1-mini:
      enabled: true
      api_url: "http://host.docker.internal:8080"
      request_timeout_seconds: 300
      estimated_vram_gb: 3.0
      max_retries: 2
      use_memory_cache: "on"
      # Orpheus disabilitato
    orpheus-3b:
      enabled: false
```

---

## 8. AVVIO FISH API SERVER SU WINDOWS (Task Scheduler)

Fish API server deve avviarsi automaticamente con Windows, prima che
Docker avvii il container ARIA broker. La sequenza corretta è:

```
Windows Boot
    │
    ├─► Task Scheduler: start-fish-api.bat (ritardo 60s dopo login)
    │       └─► conda activate fish-speech
    │           └─► python -m tools.api_server --listen 0.0.0.0:8080 ...
    │
    └─► Docker Desktop: avvia container ARIA broker
            └─► ARIA broker: load FishTTSBackend → health check :8080
```

`start-fish-api.bat`:
```bat
@echo off
cd C:\fish-speech
call conda activate fish-speech
python -m tools.api_server ^
    --listen 0.0.0.0:8080 ^
    --llama-checkpoint-path "C:\models\fish-s1-mini" ^
    --decoder-checkpoint-path "C:\models\fish-s1-mini\codec.pth" ^
    --decoder-config-name modded_dac_vq ^
    --compile
```

---

## 9. AGGIORNAMENTO DIAS — TextDirector

Il prompt di Gemini in `TextDirector` va aggiornato per usare i tag Fish
invece dei tag Orpheus. È l'unica modifica necessaria in DIAS.

### Tag mapping per il prompt

```python
# In dias/stages/text_director.py — aggiorna EMOTION_TAG_INSTRUCTIONS

EMOTION_TAG_INSTRUCTIONS = """
Stai annotando un testo narrativo italiano per la sintesi vocale con Fish Audio S1.

SINTASSI: posiziona il marker PRIMA della parola/frase che deve colorare.
Esempio: "(scared)Non c'era nessuno." oppure "Aprì la porta. (hesitating)Forse..."

MARKER DISPONIBILI PER NARRATIVA:
- Tensione/Paura:    (scared) (nervous) (panicked) (anxious) (worried)
- Tristezza:         (sad) (depressed) (sobbing) (sighing) (moved)
- Rabbia:            (angry) (furious) (frustrated) (impatient)
- Sorpresa:          (surprised) (astonished) (confused)
- Ironia/Carattere: (sarcastic) (disdainful) (sneering) (indifferent)
- Dialogo naturale: (hesitating) (sincere) (comforting) (serious)
- Gioia/Energia:    (excited) (delighted) (joyful) (proud) (laughing)
- Effetti:          (laughing) (chuckling) (sobbing) (crying loudly) (groaning)

REGOLE:
1. Massimo 2 marker ogni 3 frasi — non sovraccaricare
2. Usa solo se il contesto narrativo lo giustifica davvero
3. Preferisci marker di stato emotivo a effetti sonori diretti
4. Per il narratore: privilegia (serious), (moved), (sincere)
5. Per i personaggi: usa l'emozione dominante della scena
"""
```

---

## 10. SETUP AMBIENTE — Windows Nativo

```bash
# 1. Installa conda se non presente
# https://docs.conda.io/en/latest/miniconda.html

# 2. Crea ambiente dedicato (Python 3.10 — Fish non supporta 3.12)
conda create -n fish-speech python=3.10
conda activate fish-speech

# 3. PyTorch 2.7+ con CUDA 12.8 (FONDAMENTALE per RTX 5060 Ti sm_120)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Clona fish-speech
git clone https://github.com/fishaudio/fish-speech.git C:\fish-speech
cd C:\fish-speech
pip install -e .

# 5. Scarica modello S1-mini
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir C:\models\fish-s1-mini

# 6. Test inferenza diretta (prima di avviare il server API)
python fish_speech/models/text2semantic/inference.py \
    --text "(excited)Ciao! Sono una voce italiana generata da Fish Audio." \
    --compile

# 7. Avvia server API
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "C:\models\fish-s1-mini" \
    --decoder-checkpoint-path "C:\models\fish-s1-mini\codec.pth" \
    --decoder-config-name modded_dac_vq \
    --compile
```

**Nota sulla prima esecuzione con `--compile`**: `torch.compile` richiede
2-5 minuti la prima volta (compilazione JIT dei kernel CUDA per sm_120).
Le esecuzioni successive sono immediate. Non interrompere la prima esecuzione.

---

## 11. ROADMAP SVILUPPO

### FS-0 — Test manuale (prerequisito, nessuna modifica a ARIA)

**Obiettivo**: validare Fish S1-mini su Windows prima di toccare il codice.

- [ ] Setup ambiente conda `fish-speech` con PyTorch cu128
- [ ] Download modello `fishaudio/openaudio-s1-mini`
- [ ] Test inferenza CLI con testo italiano e emotion markers
- [ ] Confronto qualità audio: testo grezzo vs testo annotato
- [ ] Conferma RTF su RTX 5060 Ti (obiettivo: >1:3)
- [ ] Avvio server API su porta 8080
- [ ] Test chiamata HTTP diretta con `curl` o Python

**Criterio di successo**: WAV italiano con `(scared)` e `(hesitating)` che
produce variazione emotiva udibile. Se non soddisfacente → rivalutare.

**Stima**: 1-2 giorni

---

### FS-1 — Backend `fish_tts.py` (sostituisce `orpheus.py`)

**Obiettivo**: implementare il backend Fish seguendo l'interfaccia BaseBackend.

- [ ] `backends/fish_tts.py` — classe `FishTTSBackend` completa
- [ ] `backends/mock_fish_tts.py` — mock per sviluppo offline
- [ ] Aggiornamento `config.yaml` — `fish-s1-mini` enabled, `orpheus-3b` disabled
- [ ] Aggiornamento `main.py` — registra `FishTTSBackend` per model_id `fish-s1-mini`
- [ ] Test unit con MockFishTTSBackend
- [ ] Test integrazione: task Redis → WAV su `/aria-shared/`

**File da modificare**:
```
aria_server/backends/fish_tts.py      ← NUOVO
aria_server/backends/mock_fish_tts.py ← NUOVO
aria_server/main.py                   ← modifica registrazione backend
config.yaml                           ← modifica sezione models.tts
requirements.txt                      ← aggiunge requests (già presente?)
```

**Stima**: 3-5 giorni

---

### FS-2 — Avvio automatico Windows

**Obiettivo**: Fish API server si avvia con Windows senza intervento manuale.

- [ ] Crea `start-fish-api.bat` con parametri corretti
- [ ] Task Scheduler: trigger "At startup", ritardo 60s, run as user
- [ ] Test: reboot Windows → Fish API server disponibile su :8080 entro 3 minuti
- [ ] Aggiorna documentazione setup

**Stima**: 1-2 giorni

---

### FS-3 — Aggiornamento DIAS TextDirector

**Obiettivo**: DIAS genera copioni con tag Fish invece di tag Orpheus.

- [ ] Aggiorna prompt `EMOTION_TAG_INSTRUCTIONS` in TextDirector
- [ ] Aggiorna mapping tag nel modulo di annotazione
- [ ] Test: capitolo di prova → copione annotato con tag Fish → WAV generato
- [ ] Confronto qualità: testo grezzo vs testo annotato da DIAS

**Stima**: 2-3 giorni

---

### FS-4 — Test E2E completo DIAS → ARIA → Fish → WAV

**Obiettivo**: pipeline completa funzionante per il primo capitolo.

- [ ] DIAS SceneDirector genera task con `model_id: fish-s1-mini`
- [ ] Task finisce in `gpu:queue:tts:fish-s1-mini`
- [ ] ARIA BatchOptimizer sceglie la coda corretta
- [ ] FishTTSBackend genera WAV sul disco locale di Windows (`/aria/outputs`)
- [ ] Risultato (con l'URL HTTP generata) scritto su `gpu:result:dias-minipc:{job_id}`
- [ ] DIAS Watcher trova il risultato e aggiorna stato pipeline
- [ ] Primo capitolo audiolibro generato end-to-end

**Stima**: 1 settimana

---

### FS-5 — Voice cloning personaggio (futuro)

**Obiettivo**: voci diverse per narratore e personaggi usando reference audio.

- [ ] Crea 3-4 campioni reference audio (10-30s ciascuno) per personaggi principali
- [ ] Aggiorna payload DIAS per includere `voice_ref` per personaggio
- [ ] Test coerenza vocale tra scene dello stesso personaggio
- [ ] Cache voice reference su Redis per evitare ricaricamento tra scene

**Stima**: 3-5 giorni

---

## NOTE FINALI

Il copione generato da DIAS (con emozioni, ritmo narrativo, annotazioni
carattere per carattere) resta **completamente valido**. È la parte più
preziosa del sistema — è il layer di intelligenza che trasforma testo piatto
in narrazione espressiva. Fish S1-mini lo esegue meglio di Orpheus perché
il suo modello emotivo è più ricco e il suo italiano più maturo.

MusicGen, gli effetti sonori, il mixer audio (Stage E e F di DIAS) restano
completamente invariati — Fish sostituisce solo il TTS, non tocca niente altro.

*Fish Audio S1-mini Backend — ARIA v1.1 — Febbraio 2026*
