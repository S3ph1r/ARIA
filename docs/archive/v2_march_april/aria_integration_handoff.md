# DIAS ↔ ARIA: SoundFactory Integration Guide (v4.5)

Questo documento definisce il protocollo di comunicazione via Redis (LXC 120) tra lo **Stage D2 (DIAS)** e il **SoundFactory (ARIA)**.

## 📡 Code e Topic
DIAS pubblica i task su code specifiche basate sul tipo di asset.

| Asset Type | Placeholder `model_id` | Redis Queue |
| :--- | :--- | :--- |
| **Pads (Music)** | `mus` | `aria:q:sound:local:mus:dias` |
| **Ambience** | `amb` | `aria:q:sound:local:amb:dias` |
| **SFX** | `sfx` | `aria:q:sound:local:sfx:dias` |
| **Stings** | `sting` | `aria:q:sound:local:sting:dias` |

---

## 📦 Struttura del Task (Payload)

Ogni messaggio inviato da DIAS è un JSON con il seguente schema:

```json
{
  "job_id": "{project_id}-{canonical_id}",
  "client_id": "dias",
  "model_type": "sound",
  "model_id": "mus|amb|sfx|sting",
  "payload": {
    "canonical_id": "nome_univoco_asset",
    "prompt": "Descrizione FISICA professionale per l'LLM",
    "duration_s": 120.5,
    "params": {} 
  },
  "callback_key": "dias:callback:stage_d2:{project_id}:{canonical_id}",
  "timeout_seconds": 900
}
```

### Dettagli dei Campi `payload` per Tipo:

#### 1. Music (`mus`)
- **`prompt`**: Descrizione complessa (strumenti, BPM, stile, timbri). ARIA dovrebbe produrre un brano *loopabile* o comunque coerente per la durata richiesta.
- **`params`**: Contiene la **`pad_arc`**. È una lista di segmenti con l'intensità prevista (`low`, `mid`, `high`). 
  - *Utilizzo su ARIA*: Utile per capire se il modello deve essere configurato per un crescendo o se deve mantenere un'energia costante.
- **`duration_s`**: Durata stimata dell'intero capitolo.

#### 2. Ambience (`amb`)
- **`prompt`**: Rumori di fondo (es: "ronzio astronave", "vento foresta").
- **`duration_s`**: Solitamente fissa o lunga (~60-90s) per permettere il looping in Stage E.

#### 3. SFX (`sfx`) e Sting (`sting`)
- **`prompt`**: Eventi puntuali (es: "esplosione sorda", "colpo metallico").
- **`duration_s`**: Breve (~2-10s). ARIA deve produrre l'evento e "tagliare" il silenzio in eccesso.

---

## 📤 Cosa deve restituire ARIA

Quando l'asset è pronto, ARIA deve pubblicare una risposta sulla `callback_key` indicata nel task.

### Schema Risposta Successo (Valido per PAD Musicali `mus`):
```json
{
  "status": "done",
  "job_id": "{project_id}-{canonical_id}",
  "output": {
    "audio_url": "http://192.168.1.139:8082/{canonical_id}_master.wav",
    "stems": {
       "bass": "http://192.168.1.139:8082/{canonical_id}_bass.wav",
       "melody": "http://192.168.1.139:8082/{canonical_id}_melody.wav",
       "drums": "http://192.168.1.139:8082/{canonical_id}_drums.wav"
    },
    "duration_seconds": 124.2,
    "metadata": {
      "bpm": 120,
      "key": "Am",
      "model_used": "ace-step-1.5-xl"
    }
  }
}
```

> [!NOTE]
> Per asset non musicali (`amb`, `sfx`, `sting`), l'oggetto `stems` sarà omesso o nullo e verra fornito unicamente `audio_url`.

### Schema Risposta Errore:
```json
{
  "status": "error",
  "job_id": "{project_id}-{canonical_id}",
  "error": "Descrizione dell'errore tecnico"
}
```

---

## 🔑 Note Cruciali per lo Stage E (Mixdown)

Oltre al file `.wav`, ci sono alcuni dati che ARIA può fornire per rendere il mixdown di Stage E "magico":

1.  **`duration_seconds` (OBBLIGATORIO)**: DIAS ha bisogno della durata esatta al millisecondo per allineare gli SFX sulla Master Timing Grid.
2.  **`bpm` (OPZIONALE)**: Se l'LLM musicale supporta il controllo del BPM, restituirlo permette a Stage E di sincronizzare eventuali effetti ritmici.
3.  **Qualità Master**: ARIA deve servire il file in formato `.wav` (PCM 16/24 bit, 48kHz) per evitare artefatti di compressione prima del passaggio in HTDemucs.

> [!IMPORTANT]
> **HTDemucs**: Ricorda che Stage D2 (lato DIAS) prenderà il tuo Master musicale (`mus`) e lo separerà automaticamente in Stem. Non è necessario che ARIA produca gli stem, a meno che tu non voglia bypassare Demucs mandando già i file separati (in quel caso aggiorneremo il protocollo).

---

### Prossimi Passi
1. **Pausa su DIAS**: Gli stage B2, l'Orchestratore e il Client D2 sono pronti.
2. **Setup ARIA**: Configura i consumer su PC 139 per leggere da queste code.
3. **Test**: Una volta pronto, lancia Stage D2 per verificare il primo "giro" completo di produzione.
