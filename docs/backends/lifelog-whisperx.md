# Lifelog WhisperX — Backend STT per ARIA

> **Aggiornato**: 2026-07-28
> **Ambiente**: `%ARIA_ROOT%\envs\lifelog-whisperx` (Python 3.12)
> **Porta**: 8091
> **Stato**: ✅ Operativo (Blackwell Stable, confermato 2026-05-14)
> **Client principale**: Lifelog2 (CT190 via Redis)

---

## 1. Panoramica

Lifelog WhisperX è il backend di trascrizione audio di ARIA basato su WhisperX large-v3.
Sostituisce Qwen3-ASR-1.7B come backend primario di Lifelog2, aggiungendo voiceprint 256d
integrati (pyannote wespeaker-resnet34-LM) e diarizzazione nativa.

### Funzionalità principali

- **Trascrizione multilingue**: 99 lingue, rilevamento automatico. Primario: italiano
- **Word timestamps**: wav2vec2 forced alignment, ~30ms di precisione
- **Diarizzazione speaker**: pyannote community-1 — chi parla, quando
- **Segnali di affidabilità della diarizzazione** (dal 2026-07-28): speaker per parola,
  embedding per parlante, statistiche sugli intervalli grezzi di pyannote — vedi §2bis
- **Voiceprint embedding**: wespeaker-resnet34-LM, vettore 256d per speaker, pooling su max 30s
- **Output contract identico a Qwen3-ASR**: Stage C è model-agnostic

---

## 2bis. Segnali di affidabilità della diarizzazione (2026-07-28)

**I turni si costruiscono per SEGMENTO** (voto di maggioranza di whisperx),
come da sempre. Il taglio per **parola** è stato implementato, misurato e
**ritirato**: vedi sotto.

### Campi aggiuntivi nel contratto (additivi, nessuno rompe i consumatori)

| campo | contenuto |
|---|---|
| `word_timestamps[].speaker` | speaker della singola parola, che `assign_word_speakers` calcola già e whisperx scartava |
| `speaker_embeddings` | `{SPEAKER_XX: [256 float]}` — embedding per parlante calcolati da pyannote (`return_embeddings=True`) |
| `diarization_stats` | statistiche sugli intervalli **grezzi** di pyannote, prima dell'incrocio con le parole |

`diarization_stats` contiene: `n_intervals`, `n_speakers`, `dur_min`,
`dur_median`, `dur_max`, `n_under_0s5`, `n_under_1s`, `short_flips`
(alternanze A-B-A con B sotto il secondo).

**A cosa servono**: non a ricostruire i turni, ma a permettere al consumatore
(Stage C1) di **etichettare l'affidabilità** e non tentare estrazioni su
materiale non attendibile.

### Parametri opzionali della richiesta

| campo | default | effetto |
|---|---|---|
| `min_speakers` / `max_speakers` | `None` | vincoli sul numero di parlanti |
| `exclusive_diarization` | `False` | usa `exclusive_speaker_diarization` invece della standard |

Entrambi lasciati al default: **misurati come peggiorativi** su audio
ambientale (vedi sotto).

### Cosa è stato provato e scartato

Tutto misurato sullo stesso segmento reale (SNR 21 dB, tra i migliori
disponibili), conversazione con 4-5 parlanti:

| configurazione | intervalli | parlanti | <0.5s | alternanze brevi |
|---|---|---|---|---|
| **standard (in uso)** | 159 | 2 | 42 | 24 |
| `min_speakers=3` | 185 | 3 | 71 | 38 |
| `min_speakers=4` | 210 | 4 | 99 | 58 |
| `min_speakers=5` | 207 | 5 | 92 | 46 |
| `exclusive_diarization` | 180 | 2 | 72 | **78** |

- **Vincolare i parlanti non separa meglio le voci**: taglia più fine lo stesso
  audio e produce più frammenti spuri.
- **La diarizzazione esclusiva peggiora**: elimina le sovrapposizioni tagliando,
  e su audio dove accavallarsi è la norma triplica le alternanze spurie.
- **Gli embedding per parlante sono inutilizzabili come voiceprint**: confrontati
  con identità confermate presenti nella registrazione danno al massimo
  **+0.038**, contro una soglia di riconoscimento a 0.50, con tutti gli altri
  valori negativi. Sono impasti: 2 etichette per 4-5 persone reali.
- **Il taglio per parola** è corretto in teoria ma su questo audio amplifica il
  rumore: 30% dei turni finiva a 1-2 parole, con 20 alternanze A-B-A spurie.
  Il voto di maggioranza per segmento è impreciso ma **filtra** quel rumore.

Il warning `std(): degrees of freedom is <= 0` in
`pyannote/audio/models/blocks/pooling.py` lo conferma dall'interno: lo
statistics pooling gira su finestre da un frame solo.

**Conclusione**: su audio ambientale con voci lontane e sovrapposte la
diarizzazione ha un tetto, ora misurato. I segnali sopra servono a
riconoscerlo, non a superarlo.

---

## 2. Stack modelli

```
WAV input (16kHz mono, ~5 min)
         │
         ▼
┌──────────────────────────────────┐
│  WhisperX large-v3               │  ← trascrizione + language detection
│  (encoder-decoder Transformer)  │    output: testo raw + lingua
│  ~6 GB VRAM float16              │
└─────────────┬────────────────────┘
              │ testo grezzo
              ▼
┌──────────────────────────────────┐
│  wav2vec2 ITA (align model)      │  ← allineamento forzato testo → audio
│  (whisperx.load_align_model)    │    output: word timestamps (ms)
│  ~0.5 GB VRAM                   │
└─────────────┬────────────────────┘
              │ timestamps parola
              ▼
┌──────────────────────────────────┐
│  pyannote community-1            │  ← diarizzazione speaker
│  (pyannote.audio 3.x)           │    output: SPEAKER_00, SPEAKER_01...
│  ~2 GB VRAM                     │
└─────────────┬────────────────────┘
              │ turn boundaries
              ▼
┌──────────────────────────────────┐
│  wespeaker-resnet34-LM           │  ← estrazione voiceprint pooling
│  (pyannote Inference, window=whole)│   output: vettore 256d
│  ~0.5 GB VRAM                   │
└─────────────┬────────────────────┘
              │
              ▼
         merge: testo + speaker + embedding
              │
              ▼
         SpeakerTurns strutturate → JSON output
```

**VRAM totale**: ~9-10 GB su 16 GB disponibili.
**Startup da cache HF**: ~22s (7.6s ASR + 1.5s align + 1.2s diarize + 2s voiceprint).
**Latency su ~5min audio**: ~30s totali (1.5x realtime, GPU RTX 5060 Ti).

---

## 3. Blackwell Fixes (RTX 5060 Ti, sm_120)

Due problemi noti su architettura Blackwell, risolti in `server.py`:

### Fix 1 — compute_type float16
cuBLAS int8 non è supportato su sm_120 (`CUBLAS_STATUS_NOT_SUPPORTED`).
```python
_model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type="float16")
```

### Fix 2 — Arch spoof per pyannote
NVRTC Jiterator fallisce su FFT complessa a sm_120.
Il monkey-patch deve essere applicato **prima** di qualsiasi import pyannote/torchaudio.
```python
_orig_cap = torch.cuda.get_device_capability
def _patched_cap(device=None):
    cap = _orig_cap(device)
    return (9, 0) if cap[0] >= 12 else cap
torch.cuda.get_device_capability = _patched_cap
```

### Fix 3 — soundfile bypass per audio loading
conda-forge ffmpeg DLL crash su Windows 11 Blackwell (exit 0xC0000139 = STATUS_ENTRYPOINT_NOT_FOUND).
Il server bypassa `whisperx.load_audio()` e legge i WAV direttamente con soundfile:
```python
audio_np, sr = sf.read(wav_path, dtype="float32", always_2d=False)
```

---

## 4. Coda Redis e Payload

### Coda input

```
aria:q:stt:local:whisperx-large-v3:lifelog
```

Pattern standard ARIA: `aria:q:{type}:local:{model_id}:{client_id}`

### Payload task (CT190 → Redis → ARIA)

```json
{
  "job_id": "uuid-v4",
  "client_id": "lifelog",
  "model_type": "stt",
  "model_id": "whisperx-large-v3",
  "callback_key": "aria:c:lifelog:{job_id}",
  "timeout_seconds": 1800,
  "priority": 1,
  "payload": {
    "wav_url": "http://192.168.1.104:9000/lifelog/normalized-audio/{user_id}/{yyyy}/{mm}/{dd}/{segment_id}.wav",
    "segment_id": "uuid-del-segmento",
    "language": "it"
  }
}
```

### Risultato (Redis → CT190)

```json
{
  "job_id": "uuid-v4",
  "status": "done",
  "processing_time": 29.5,
  "output": {
    "transcript": "Allora oggi ho parlato con Francesco del progetto...",
    "language": "it",
    "duration_ms": 299800,
    "transcription_quality": {
      "avg_logprob_mean": -0.35,
      "no_speech_prob_mean": 0.02,
      "no_speech_prob_max": 0.08,
      "n_segments": 12
    },
    "speaker_turns": [
      {
        "speaker": "SPEAKER_00",
        "start_ms": 0,
        "end_ms": 12400,
        "text": "Allora oggi ho parlato con Francesco del progetto",
        "avg_logprob": -0.35,
        "no_speech_prob": 0.02
      },
      {
        "speaker": "SPEAKER_01",
        "start_ms": 12800,
        "end_ms": 31200,
        "text": "Sì esatto, e la scadenza è giovedì",
        "avg_logprob": -0.5,
        "no_speech_prob": 0.1
      }
    ],
    "word_timestamps": [
      {"word": "Allora", "start_ms": 0, "end_ms": 420, "score": 0.87, "speaker": "SPEAKER_00"},
      {"word": "oggi", "start_ms": 440, "end_ms": 680, "score": 0.91, "speaker": "SPEAKER_00"}
    ],
    "voiceprints": {
      "SPEAKER_00": [0.123, -0.456, ...],
      "SPEAKER_01": [0.789, 0.012, ...]
    }
  }
}
```

---

## 5. Architettura del server FastAPI (porta 8091)

```
backends/lifelog_whisperx/
└── server.py   ← FastAPI entrypoint (avviato JIT dall'orchestratore)
```

### Endpoints

```
GET  /health
     → {"status": "ok", "model": "whisperx-large-v3", "device": "cuda", "vram_gb": 9.4, "voiceprint": true}

POST /transcribe
     Body: {"wav_url": "...", "segment_id": "...", "language": "it"}
     → SpeakerTurns JSON con voiceprints (vedi Payload risultato sopra)
```

### Ciclo di vita JIT

```
1. Orchestratore riceve task su aria:q:stt:local:whisperx-large-v3:lifelog
2. subprocess.Popen → avvia envs\lifelog-whisperx\python.exe backends\lifelog_whisperx\server.py
3. LifelogWhisperXBackend.load() → polling GET :8091/health (timeout 150s)
4. Server carica WhisperX + align + diarize + wespeaker (una volta sola in VRAM)
5. Per ogni task: POST /transcribe → scarica WAV da URL → pipeline → JSON
6. Inattività > 45 min → orchestratore termina processo → VRAM liberata
```

---

## 6. Ambiente conda `lifelog-whisperx`

**Path**: `C:\Users\Roberto\aria\envs\lifelog-whisperx\`

| Componente | Versione | Note |
|------------|----------|------|
| Python | 3.12 | |
| PyTorch | 2.8.0+cu128 | sm_120 (Blackwell) — versione richiesta da whisperx |
| `whisperx` | 3.8.5 | WhisperX large-v3 + wav2vec2 align |
| `pyannote.audio` | 3.x | speaker-diarization-community-1 + wespeaker-resnet34-LM |
| `soundfile` | latest | Bypass ffmpeg DLL crash |
| `resampy` | latest | Resampling audio se SR ≠ 16kHz |

### Comandi setup (già eseguito 2026-05-14)

```cmd
conda create --prefix C:\Users\Roberto\aria\envs\lifelog-whisperx python=3.12 -y

:: PyTorch cu128 (whisperx richiede 2.8.x)
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 torchvision==0.23.0+cu128 ^
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

:: WhisperX
pip install whisperx==3.8.5

:: pyannote (wespeaker embedding)
pip install pyannote.audio

:: utility
pip install soundfile resampy fastapi uvicorn python-dotenv minio huggingface_hub requests
```

---

## 7. Modelli su disco

Tutti i pesi in HF cache (`~/.cache/huggingface/`) — scaricati automaticamente al primo avvio.

| Modello | Dimensione approssimativa |
|---------|--------------------------|
| `openai/whisper-large-v3` | ~3 GB |
| `jonatasgrosman/wav2vec2-large-xlsr-53-italian` (align ITA) | ~1.3 GB |
| `pyannote/speaker-diarization-community-1` | ~2 GB |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | ~0.5 GB |

---

## 8. Wrapper orchestratore

```python
# aria_node_controller/backends/lifelog_whisperx.py
class LifelogWhisperXBackend:
    model_id   = "whisperx-large-v3"
    model_type = "stt"
    SERVER_URL = "http://127.0.0.1:8091"

    def load(self, model_path, config):
        # polling GET /health fino a server ready
        ...

    def run(self, payload) -> dict:
        # POST /transcribe → restituisce SpeakerTurns JSON con voiceprints
        ...

    def estimated_vram_gb(self) -> float:
        return 12.0
```

---

## 9. Integrazione Lifelog2 (Stage C)

Stage C su CT190 (`lifelog:stream:asr`):

1. Legge evento da `lifelog:stream:asr` (emesso da Stage B)
2. Costruisce payload ARIA con `wav_url` MinIO del segmento
3. `LPUSH aria:q:stt:local:whisperx-large-v3:lifelog` → job inviato
4. `BRPOP aria:c:lifelog:{job_id}` → attende risultato (timeout 1800s)
5. Parsa `speaker_turns` e `voiceprints` → crea `SpeakerTurn` records in CT105
6. Salva transcript grezzo in MinIO `transcripts/raw/{user_id}/{YYYY}/{MM}/{DD}/{segment_id}.json`
7. Aggiorna `Segment.pipeline_status = "enriching"`
8. Emette su `lifelog:stream:enrich`

---

*Lifelog WhisperX Backend — Maggio 2026*
*Documenti correlati: [ARIA Service Registry](../ARIA-Service-Registry.md), [Lifelog ASR (qwen3)](lifelog-asr.md)*

---

## 10. Taglio mirato dei turni fusi (2026-07-29)

### Il problema, misurato

Su 7900 turni reali in Lifelog2, distribuzione di `n_segments_merged`:

| segmenti fusi | turni | durata media | testo affidabile |
|---|---|---|---|
| 1 | 2694 | 4.4s | 51.1% |
| 2-3 | 1965 | 16.9s | 58.4% |
| 4-8 | 1327 | 41.7s | 63.8% |
| **9+** | **965** | **195.9s** | **76.7%** |

I turni con 9 o più segmenti fusi durano in media più di tre minuti: sono
blocchi in cui pyannote ha dichiarato lo stesso parlante per l'intera durata e
la fusione ha inglobato gli interlocutori. Letti a mano contengono
palesemente due o più persone che si rispondono — l'embedding che ne esce è la
media di più voci e in Lifelog2 fabbrica identità inesistenti.

Da notare il paradosso nell'ultima colonna: **più la fusione è grave, più il
testo sembra affidabile**. È aritmetica — un blocco da tre minuti ha centinaia
di parole, quindi `logprob` e `avg_word_score` si mediano bene. Una metrica
unica di qualità premierebbe esattamente i turni più rotti.

### Perché ora funziona e il 2026-07-28 no

Il tentativo precedente ricostruiva **tutti** i turni a livello di parola e
frammentava: 30% dei turni ridotti a 1-2 parole, alternanze A-B-A spurie
(vedi §2bis). Fu annullato con `06c074f`.

Due differenze:

1. **Si tocca solo il 12% già degenerato** (`n_segments_merged >= 9`). I 2694
   turni da un segmento, che durano 4.4s e funzionano, restano intatti.
2. **Si taglia solo dove il cambio di parlante dura.** Una sequenza di parole
   sotto `SPLIT_MIN_RUN_MS` (1500ms) o `SPLIT_MIN_RUN_WORDS` (4) viene
   riassorbita nella precedente invece di aprire un turno.

Segue una **ricucitura**: riassorbire una sequenza spuria lascia due tratti
dello stesso parlante separati dal buco appena chiuso, e senza rifonderli il
taglio produrrebbe frammentazione al posto di ripararla. È il modo preciso in
cui il tentativo precedente si autosabotava.

### Verifica offline (senza GPU)

| caso | atteso | esito |
|---|---|---|
| A / interiezione 0.2s / A / B / A | 3 turni (A, B, A) | 3 turni |
| un solo parlante per 15s | intatto | 1 turno |
| alternanza ogni 0.3s (patologica) | **non** frammentare | 1 turno |

Il terzo è il controllo decisivo: è la firma del fallimento precedente.

### Dettagli di implementazione

Il taglio avviene **prima** del ciclo degli embedding, così ogni sotto-turno
riceve il proprio embedding sui confini nuovi invece di ereditare la media del
padre — è tutto il punto dell'operazione.

`n_segments_merged` viene **ricalcolato** contando i segmenti Whisper che si
sovrappongono al sotto-turno: ereditarlo dal padre (anche 29) marcherebbe il
sotto-turno come fuso proprio dopo averlo separato.

`avg_logprob`, `no_speech_prob` e `avg_compression_ratio` sono grandezze per
segmento e non sono ripartibili: i sotto-turni le ereditano dal padre. È
un'approssimazione dichiarata e accettabile, perché quelle misure valutano il
**testo**, che nel padre e nei figli è lo stesso materiale. Le grandezze che
cambiano davvero — `word_count`, `avg_word_score`, `n_segments_merged` — sono
ricalcolate sulle parole del sotto-turno.

I turni prodotti dal taglio portano `split_from_fused: true`, così il
consumatore può distinguerli e misurarne l'effetto.

## 11. Intervalli acustici per turno e embedding su confini pyannote (2026-07-29)

### Il difetto misurato

I confini dei turni sono la griglia **testuale** di whisper: l'allineatore
(wav2vec2) incolla le parole all'audio ma non può uscire dalla finestra del
segmento che whisper ha scelto, e ai cambi di parlante comprime le ultime
parole dentro il confine. Misurato su un segmento reale (Lifelog2 `0fc60ef2`,
profilo RMS): la trascrizione dichiara una pausa a 15.39–16.01, la forma
d'onda la mostra a ~15.44–16.24; il vero cambio di parlante è a ~17.08, non a
16.75. Due conseguenze:

1. il taglio audio "dal turno X al turno Y" mozza le ultime parole (~0.3s)
2. **l'embedding del turno successivo contiene la coda della voce precedente**
   — su un turno da 2-3s è il 10-15% del campione

Gli intervalli grezzi di pyannote non soffrono dello squeeze (lavorano
sull'acustica, non sul testo) ma finivano buttati dopo
`assign_word_speakers`.

### Cosa cambia nel contratto

Per ogni turno, quando pyannote ha intervalli per quella etichetta (campi
additivi, assenti altrimenti):

| campo | contenuto |
|---|---|
| `diar_intervals` | `[[start_ms, end_ms], …]` — gli intervalli acustici di QUESTO speaker che toccano il turno, con gli estremi di pyannote (tagliati solo oltre ±1s dalla finestra whisper, per non trascinare dentro turni A-B-A) |
| `acoustic_start_ms` / `acoustic_end_ms` | inviluppo degli intervalli — i confini "veri" per il player |
| `embedding_source` | `"diar"` = embedding calcolato sul concatenato degli intervalli; `"window"` = fallback sulla finestra whisper (nessun intervallo utilizzabile o < 0.5s di parlato) |

L'embedding con `source="diar"` esclude dal campione il parlato dedicato degli
altri parlanti; non può nulla contro il parlato sovrapposto, che è mescolato
nell'audio stesso.

### Consumatori

- Lifelog2 Stage C1: salva i confini acustici per il player della vista
  `/segments` e usa gli embedding più puliti per il raggruppamento
- il taglio audio dei turni (dashboard `/segment/{id}/audio-clip`) userà
  `acoustic_*` quando presenti, con fallback sui confini whisper per lo
  storico non riprocessato

### Aggiornamento 2026-07-29 sera — il vincolo sulla fusione è ritirato

`SPLIT_MIN_MERGED = 9` guardava la **posizione** del difetto invece della sua
**sostanza**, e per questo mancava i casi peggiori.

Il caso che l'ha smontato è il turno #5 del segmento golden `0fc60ef2`, che
fonde tre battute di due persone:

| intervallo | speaker per parola | testo |
|---|---|---|
| 121.70–124.66 | S02 (Alex) | *Ma tutti nei contratti nuovi c'ha già quella cosa lì.* |
| 125.16–127.60 | **S01 (utente)** | *Non so se è una questione d'età o...* |
| 127.66–129.78 | S02 (Alex) | *Eh, prima hai detto d'età, adesso hai detto un po'.* |

`n_segments_merged` = **2**, quindi il taglio non lo guardava nemmeno. Il voto
di maggioranza per segmento assegnava tutto a SPEAKER_02 (19 parole contro 8) e
la frase dell'utente spariva dentro un turno di Alex — con l'embedding
corrispondente contaminato.

Lo speaker per parola di `assign_word_speakers` le separava correttamente:
l'informazione era già nel contratto, la buttava via il montaggio.

**Ora il taglio vale per tutti i turni** e a filtrare restano le sole soglie di
sostanza, che sono sempre state il filtro giusto:

- `SPLIT_MIN_RUN_MS = 1500` — un cambio parlante più breve è rumore
- `SPLIT_MIN_RUN_WORDS = 4` — idem sotto le 4 parole

Misurato sul golden togliendo il vincolo, **17 turni → 26**:

- il #5 si separa nei tre pezzi giusti; il frammento «Ma» (0.2s, 1 parola)
  viene riassorbito
- il #4 restituisce «Dici se li avessi investiti?» (1.7s, 5 parole), una
  battuta di Alex sepolta dentro 53s di monologo dell'utente
- nella discussione fitta di fine segmento solo **6 sequenze su 18** vengono
  promosse a turno: le altre sono riassorbite, come devono

Nessuna traccia del 30% di turni a 1-2 parole che aveva affossato il tentativo
del 28/07: le guardie funzionavano già, mancava loro solo il permesso di
intervenire.

### Nota operativa: riprocessare non è idempotente

Confrontando la trascrizione del 16/07 con quella del 29/07 **dello stesso
m4a**: 647 parole contro 666, 3504 caratteri contro 3606. WhisperX segmenta con
VAD e processa in batch; piccole differenze nel chunking spostano il contesto e
il modello genera testo leggermente diverso. Nel caso specifico la seconda
passata ha recuperato 14 secondi di parlato che la prima aveva perso.

Conseguenza pratica: **i risultati di un riprocessamento vanno giudicati
statisticamente, non diffando due run**. Un turno che cambia fra due passate
non è di per sé una regressione.

### Correzione 2026-07-29 — il parlante della sequenza va portato esplicito

Tolto il vincolo sulla fusione, il turno #5 del golden **continuava** a non
separarsi: usciva `121.5→127.6 SPEAKER_01` con dentro sia la frase di Alex sia
quella dell'utente.

Causa: l'identità di una sequenza veniva riletta da `run[0]["speaker"]`, cioè
dalla sua prima parola. Nel turno la sequenza si apre con un «Ma» di 0.16s
dell'utente, troppo corto per fare turno e quindi assorbito in testa alla frase
di Alex — che da quel momento *sembra* dell'utente. La ricucitura successiva
vede due sequenze consecutive attribuite a SPEAKER_01 e le fonde, inghiottendo
nel mezzo tutto ciò che aveva detto Alex.

L'assorbimento in testa (`pending`) era stato introdotto per non far diventare
un turno a sé i frammenti iniziali; il difetto è che cambiava anche l'identità
di ciò a cui venivano attaccati.

Correzione: `merged_runs` porta le coppie `(parlante, parole)`, col parlante
fissato **prima** dell'assorbimento. Verificato sulle parole reali del turno:

| | | |
|---|---|---|
| S02 | 121.52→124.66 | *Ma tutti nei contratti nuovi c'ha già quella cosa lì.* |
| S01 | 125.16→127.60 | *Non so se è una questione d'età o...* |
| S02 | 127.66→129.78 | *Eh, prima hai detto d'età, adesso hai detto un po'.* |

Nota di metodo: le etichette per parola sono risultate **identiche** fra due
riprocessamenti dello stesso m4a, mentre il testo no (647 vs 666 parole). La
diarizzazione è stabile; la variabilità sta nell'ASR.
