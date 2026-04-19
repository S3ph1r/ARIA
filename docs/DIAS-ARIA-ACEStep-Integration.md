# DIAS ↔ ARIA: Protocollo di Integrazione Sound Production
*Versione 2.0 — Aprile 2026 (aggiornato: Audiocraft routing, Leitmotif, HTDemucs fix)*

---

## Indice

1. [Visione e Architettura](#1-visione-e-architettura)
2. [Mappa del Flusso Dati DIAS](#2-mappa-del-flusso-dati-dias)
3. [Stage B2: Cosa Produce e Perché](#3-stage-b2-cosa-produce-e-perché)
4. [Il Problema del Dialetto: Sound Designer vs Musicista](#4-il-problema-del-dialetto)
5. [ACE-Step Task Descriptor: Il Contratto](#5-ace-step-task-descriptor-il-contratto)
6. [Traduzione B2 → Prompt ACE-Step](#6-traduzione-b2--prompt-ace-step)
7. [Stage D2: Dispatch verso ARIA](#7-stage-d2-dispatch-verso-aria)
8. [HTDemucs: Stem Separation per il PAD](#8-htdemucs-stem-separation-per-il-pad)
9. [Stage E: Come Consuma gli Asset del Wrapper](#9-stage-e-come-consuma-gli-asset-del-wrapper)
10. [Stage F/G: Mastering e Consegna](#10-stage-fg-mastering-e-consegna)
11. [Protocollo Redis Completo](#11-protocollo-redis-completo)
12. [Esempio End-to-End: Scalzi "Uomini in Rosso"](#12-esempio-end-to-end)
13. [Checklist Implementativa](#13-checklist-implementativa)

---

## 1. Visione e Architettura

### Il Radiodramma Automatizzato

DIAS produce radiodramma in stile BBC/Star Wars da testo letterario. Il risultato finale è:

```
Voce (narrazione + dialoghi)
  + PAD musicale (tappeto emotivo continuo, sincronizzato e duckato)
  + Leitmotif (tema musicale per personaggio/entità, posizionato per scena)
  + AMB (letto ambientale per location)
  + SFX (effetti sonori diegetici e non)
  + STING (punteggiatura drammatica)
  = Mix broadcast-ready (-16 LUFS, stereo)
```

### Divisione dei Ruoli

```
DIAS (LXC 190)                           ARIA (PC 139)
─────────────────────────────────────    ─────────────────────────────────
DECIDE: cosa generare                    ESEGUE: come generare
         quando usarlo                            gestione GPU
         come mixarlo                             codec/formato output
         quali parametri musicali                 relay chunking (PAD lungo)
─────────────────────────────────────    ─────────────────────────────────
Produce: ACE-Step Task Descriptor        Produce: WAV + score.json
         IntegratedCueSheet                       manifest di path locali
         Master Timing Grid
```

### Stack Tecnologico

| Componente | Dove | Asset | Funzione |
|-----------|------|-------|---------|
| ACE-Step 1.5 XL SFT | PC 139, porta 8084 | PAD, Leitmotif | Generazione musicale, relay chunking, tonal lock |
| AudioGen medium | PC 139, porta 8086 | AMB, SFX | Generazione audio non-musicale da testo |
| MusicGen large | PC 139, porta 8086 | STING | Generazione musicale breve |
| HTDemucs 6s | PC 139 (subprocess) | PAD | Separazione stem (bass, drums, other) dal PAD master |
| Stage D2 (DIAS) | LXC 190 | tutti | Client dispatch + download asset + manifest |
| Stage E (DIAS) | LXC 190 | — | Mix timeline voce + tutti gli asset |
| Redis 6379 | LXC 120 | — | Bus di comunicazione DIAS ↔ ARIA |

> **Env unificato**: ACE-Step (8084), AudioGen/MusicGen (8086) e HTDemucs condividono lo stesso ambiente Python `dias-sound-engine` (torch 2.11.0+cu128). Sono processi distinti avviati JIT — mai in contemporanea per la VRAM.

---

## 2. Mappa del Flusso Dati DIAS

```
Stage 0 (Intel)
  └── preproduction.json: palette_choice, casting, project_sound_palette

Stage A (Text Ingester)
  └── Macro-chunk (2500w) + Micro-chunk (300w)

Stage B (Semantic Analyzer)
  └── MacroAnalysisResult: {valence, arousal, tension, primary_emotion,
                            setting, audio_cues[], entities[]}

Stage C (Scene Director)
  └── SceneScript per scena: {scene_id, text_content, voice_direction,
                               timing_estimate, audio_layers}

Stage D (Voice Generator) ← usa ARIA TTS (Qwen3/Fish)
  └── {scene_id}.wav + Master Timing Grid:
      {scene_id → {start_offset_s, voice_duration_s, pause_after_s}}

Stage B2-Macro (Musical Director)
  Input:  preproduction.json + MacroAnalysisResult + Master Timing Grid
  └── MacroCue: {
        pad: PadRequest {
          canonical_id, production_prompt, production_tags,
          negative_prompt, guidance_scale, inference_steps,
          estimated_duration_s, is_leitmotif,
          pad_arc: [{start_s, end_s, intensity, roadmap_item}]
        }
      }

Stage B2-Micro (Sound Designer di Dettaglio) — prompt v4.1
  Input:  MacroCue + SceneScript[] + Master Timing Grid + project_sound_palette
  └── IntegratedCueSheet v4.1: {
        pad_canonical_id,
        leitmotif_events: [{                    ← NUOVO in v4.1
          scene_id, leitmotif_id, timing, reasoning
        }],
        scenes_automation: [{
          scene_id, pad_volume_automation, pad_duck_depth, pad_fade_speed,
          amb_id, amb_offset_s, amb_duration_s,
          sfx_id, sfx_timing, sfx_offset_s,
          sting_id, sting_timing
        }],
        sound_shopping_list: [SoundShoppingItem]
      }

Stage D2 (Sound Factory)
  Input:  sound_shopping_list_aggregata.json + project_sound_palette (leitmotif)
  Routing: PAD/Leitmotif → ACE-Step (8084) | AMB/SFX/STING → Audiocraft (8086)
  Output: WAV locali + manifest.json + stems PAD (HTDemucs: bass, drums, other)

Stage E (Mixdown) ✅ Implementato
  Input:  Voice WAVs (D) + PAD stems (D2) + IntegratedCueSheet (B2-Micro)
          + Master Timing Grid (D) + leitmotif WAVs
  Output: mix stereo WAV per capitolo (PAD + Leitmotif + AMB + SFX + STING + Voce)

Stage F/G (Mastering) [DA IMPLEMENTARE]
  Input:  Mix da Stage E
  Output: -16 LUFS, -1 dBTP, MP3 320kbps
```

---

## 3. Stage B2: Cosa Produce e Perché

### 3.1 B2-Macro: Il Direttore Musicale

B2-Macro decide l'identità sonora del macro-chunk (ca. 2500 parole, ~15-20 minuti). La sua decisione principale è il **PAD**: la traccia musicale continua che accompagna tutta la narrazione.

Il PAD è di tipo `Stem A` nel modello DIAS — il "tappeto orchestrale" che rimane sempre presente, duckato sotto la voce e alzato nelle pause significative.

**Output chiave — il `pad_arc`:**

```json
{
  "pad_arc": [
    {"start_s": 0,    "end_s": 120,  "intensity": "low",  "roadmap_item": "[00:00-02:00]. Apertura atmosferica."},
    {"start_s": 120,  "end_s": 360,  "intensity": "mid",  "roadmap_item": "[02:00-06:00]. Tensione crescente, ingresso ottoni."},
    {"start_s": 360,  "end_s": 420,  "intensity": "high", "roadmap_item": "[06:00-07:00]. Climax orchestrale, picco massimo."},
    {"start_s": 420,  "end_s": 470,  "intensity": "low",  "roadmap_item": "[07:00-07:50]. Decadimento, ritorno al drone."}
  ]
}
```

Il `pad_arc` serve a due scopi distinti:
1. **Per ACE-Step**: si traduce nel campo `lyrics` (road map strutturale per il DiT)
2. **Per Stage E**: determina quali stem Demucs attivare in ogni intervallo temporale

### 3.2 B2-Micro: Il Sound Designer di Dettaglio

B2-Micro lavora a livello di scena (micro-chunk ~300 parole, ~2-3 minuti). Decide:
- Come il PAD respira intorno alla voce (`pad_volume_automation`, `pad_duck_depth`)
- Quale letto ambientale usare (`amb_id`)
- Quali SFX inserire (`sfx_id`, `sfx_timing`)
- Se inserire uno sting drammatico (`sting_id`)

**Parametri di ducking definiti da B2-Micro:**

| `pad_duck_depth` | dB effettivi | Quando usarlo |
|-----------------|-------------|---------------|
| `shallow`       | -6 dB       | Voce bassa, scena intima |
| `medium`        | -12 dB      | Narrazione standard |
| `deep`          | -18 dB      | Dialogo intenso, consonanti critiche |

| `pad_volume_automation` | Effetto | Trigger |
|------------------------|---------|---------|
| `ducking`              | Abbassa il PAD | Voce attiva |
| `neutral`              | PAD a volume pieno | Pausa > 0.8s |
| `build`                | PAD cresce progressivamente | Prima del climax |

| `pad_fade_speed` | Durata transizione | Percepito come |
|-----------------|-------------------|----------------|
| `snap`          | 0.3s | Taglio netto, urgenza |
| `smooth`        | 1.0s | Naturale, BBC standard |
| `slow`          | 2.5s | La musica "emerge" nella pausa |

---

## 4. Il Problema del Dialetto

### 4.1 Il Disallineamento Fondamentale

B2 parla il linguaggio del **sound designer cinematografico**. ACE-Step (via Qwen3 LM) è addestrato sul linguaggio del **compositore di canzoni**. Questo genera prompt drift: Qwen3 "traduce" il prompt tecnico in qualcosa che conosce, perdendo la tessitura timbrica richiesta.

**Esempio reale:**

```
B2-Macro produce:
  production_tags: "70s retro-futuristic, ARP 2600 synth, dissonant brass clusters,
                   orchestral low strings, spring reverb, tape saturation,
                   metallic percussive hits, sub-bass drones"

Qwen3 (senza shielding) re-interpreta come:
  "A somber, cinematic piece with low strings and ambient texture"

DiT genera: archi generici, nessun ARP, nessuna texture analogica
```

### 4.2 La Causa Tecnica

Il pipeline ACE-Step 1.5 XL ha 4 stadi:

```
1. LM Qwen3 Phase 1: Genera metadata CoT (bpm, keyscale, timesig, caption)
2. LM Qwen3 Phase 2: Genera audio codes (600 token) guidato dal think block
3. T5 Text Encoder:  Encoda il caption per il DiT
4. DiT XL (60 step): Diffusione → VAE → WAV
```

Il problema è allo stadio 1: Qwen3 è un "cervello da cantautore" addestrato su canzoni pop/rock/jazz. Terminologia come "ARP 2600" o "spring reverb" non è nel suo vocabolario nativo → la riscrive con termini che conosce → il think block che guida la Phase 2 contiene la versione riscritta → gli audio codes vengono generati sulla base del prompt semplificato.

### 4.3 Mitigazione con `use_cot_caption = False`

Con `use_cot_caption=False` il caption rinegoziato da Qwen3 non viene passato al T5 encoder (stadio 3). Il T5 usa il prompt originale dell'utente. Tuttavia il think block degli audio codes (stadio 2) rimane influenzato dalla rinegoziazione.

**Shielding parziale — impatto:**
- Stadio 3 (T5 → DiT): usa il prompt originale ✓
- Stadio 2 (audio codes Qwen3): ancora influenzato dalla rinegoziazione ⚠

### 4.4 La Soluzione Strutturale: Tradurre in Musicista

La soluzione non è semplificare il suono, è **tradurre il concetto sound-designer in vocabolario nativo ACE-Step**. B2 deve produrre prompt che Qwen3 riconosce già come "musica orchestrale cinematografica".

Regola pratica:

| Linguaggio B2 (Sound Designer) | Linguaggio ACE-Step (Musicista) |
|--------------------------------|--------------------------------|
| "ARP 2600 synth" | "vintage synthesizer, analog keyboard, 1970s" |
| "dissonant brass clusters" | "dark brass ensemble, atonal harmony, dissonant" |
| "spring reverb, tape saturation" | "analog warmth, vintage recording, retro production" |
| "sub-bass drones" | "low frequency drone, deep bass, sustained" |
| "metallic percussive hits" | "metallic percussion, industrial hits, textural drums" |
| "claustrophobic atmosphere" | "tense underscore, suspense, dark cinematic" |
| "AIFF textures" | "lo-fi, grainy, textural, ambient noise" |

---

## 5. ACE-Step Task Descriptor: Il Contratto

Il **ACE-Step Task Descriptor** è il JSON prodotto da B2/D2 per ogni asset musicale. È il formato di interfaccia canonico tra DIAS e il wrapper ARIA.

### 5.1 Schema Completo

```json
{
  "schema_version": "1.0",
  "asset": {
    "canonical_id": "pad_retro_sci_fi_tension_01",
    "type": "pad",
    "stem_role": "A"
  },
  "generation": {
    "thinking": true,
    "use_cot_caption": false,
    "use_cot_lyrics": false,
    "use_cot_language": false,
    "inference_steps": 60,
    "guidance_scale": 4.5,
    "duration_s": 470.4,
    "seed": 42
  },
  "prompt": {
    "tags": "dark orchestral underscore, vintage synthesizer, low strings, distant brass, analog warmth, 1970s cinematic, sustained drone, slow tempo, minor key, tense atmosphere, no vocals, instrumental",
    "negative_prompt": "upbeat, happy, pop, generic ai music, polished modern production, vocals, bright, major key",
    "lyrics_arc": [
      {"tag": "[Intro]",       "description": "Sub-bass drone, sparse texture, isolated synth shimmer.",          "duration_s": 90},
      {"tag": "[Development]", "description": "Brass enters gradually, low strings swell, tension builds.",        "duration_s": 240},
      {"tag": "[Peak]",        "description": "Full orchestral climax, dissonant brass, metallic percussion hits.", "duration_s": 100},
      {"tag": "[Outro]",       "description": "Decay to minimal drone, metallic textures fade to silence.",        "duration_s": 40}
    ]
  },
  "tonal_lock": {
    "bpm": 65,
    "keyscale": "F minor",
    "timesignature": "4"
  },
  "relay": {
    "enabled": true,
    "chunk_s": 120.0,
    "tail_s": 15.0,
    "crossfade_s": 3.0
  },
  "post_processing": {
    "htdemucs": true,
    "stems_requested": ["bass", "melody", "drums"]
  },
  "traceability": {
    "project_id": "scalzi-uomini-in-rosso",
    "chunk_label": "chunk-000",
    "b2_macro_version": "4.0",
    "source_pad_arc": [
      {"start_s": 0,   "end_s": 120, "intensity": "low",  "roadmap_item": "[00:00-02:00]. Apertura atmosferica."},
      {"start_s": 120, "end_s": 360, "intensity": "mid",  "roadmap_item": "[02:00-06:00]. Tensione crescente."},
      {"start_s": 360, "end_s": 420, "intensity": "high", "roadmap_item": "[06:00-07:00]. Climax orchestrale."},
      {"start_s": 420, "end_s": 470, "intensity": "low",  "roadmap_item": "[07:00-07:50]. Decadimento."}
    ]
  }
}
```

### 5.2 Regole per Campo

**`generation.thinking`**
- PAD (Stem A, >60s): sempre `true`. Il modello ha bisogno dei sound codes guidati per tracce lunghe. Senza thinking, il DiT deraglia dopo ~30s producendo rumore statico ciclico.
- STING (<10s): `false` accettabile. La breve durata non beneficia del thinking overhead.
- AMB (bed ambientale): `false`, si genera con prompt diretto per texture non musicale.
- SFX: `false`, prompt diretto, nessuna struttura musicale.

**`generation.guidance_scale`**
- 4.5: texture vintage, realismo analogico, bordi morbidi (consigliato per PAD cinematografici)
- 6.0-7.0: definizione timbrica più netta, adatto per leitmotif con strumento lead chiaro

**`prompt.lyrics_arc`**
I tag strutturali che ACE-Step usa per costruire l'arco narrativo interno della traccia. NON è testo cantato. È la **road map che il DiT legge per pianificare la progressione temporale**.

Tag riconosciuti da ACE-Step (da usare nell'ordine indicato):
- `[Intro]` — apertura, stabilisce il tono
- `[Verse]` — sviluppo principale
- `[Pre-Chorus]` — preparazione al picco (opzionale)
- `[Chorus]` — picco emotivo
- `[Bridge]` — contrasto/transizione (opzionale)
- `[Outro]` — chiusura/decadimento

Per uso cinematografico mappare il pad_arc DIAS su questi tag:

| pad_arc DIAS | Tag ACE-Step |
|-------------|-------------|
| `intensity: low` (apertura) | `[Intro]` |
| `intensity: mid` (sviluppo) | `[Verse]` |
| pre-climax (build) | `[Pre-Chorus]` |
| `intensity: high` (climax) | `[Chorus]` |
| `intensity: low` (chiusura) | `[Outro]` |

**`tonal_lock`**
Parametri ereditati dal `score.json` prodotto da ACE-Step nel chunk pt0 (primo chunk della relay). Per il primo chunk di ogni PAD, B2-Macro può specificare valori suggeriti; il wrapper li usa come hint per il TOML. Dal pt1 in poi, il wrapper li legge automaticamente dal `lm_metadata` del score.json precedente.

**`relay.enabled`**
`true` solo per PAD (Stem A). Tutti gli altri asset (AMB, SFX, STING) sono single-shot. La relay è gestita interamente dal wrapper ARIA — DIAS non deve spezzare manualmente la durata.

**`post_processing.htdemucs`**
`true` solo per PAD. Il master WAV viene passato a HTDemucs che produce 4 stem separati. Stage E usa questi stem per il controllo dinamico dell'intensità.

---

## 6. Traduzione B2 → Prompt ACE-Step

### 6.1 Template Standard Appendice A (adattato)

```
[Role] cinematic underscore, no vocals, instrumental
[Mood] {mood_principale}, {mood_secondario}
[Instrument] {strumenti_in_vocabolario_musicista}
[Structure] {struttura_narrativa}, {loop_o_arco}
[Production] {stile_produzione_in_vocabolario_musicista}
[Technical] key: {tonalità}, tempo: {bpm}bpm
```

Esempio per il macro-chunk "Scalzi - capitolo 1":

```
[Role] cinematic underscore, no vocals, instrumental, no lyrics
[Mood] dark suspense, tension rising, ominous
[Instrument] low strings, distant brass ensemble, vintage analog synthesizer, deep bass
[Structure] slow development, sparse intro, orchestral climax, gradual decay
[Production] 1970s analog warmth, retro cinematic, grainy texture, wide stereo
[Technical] key: F minor, tempo: 65bpm, sustained drone foundation
```

### 6.2 Regola di Costruzione per B2

B2-Macro deve costruire il campo `prompt.tags` concatenando elementi dalle seguenti categorie, in vocabolario ACE-Step:

```python
def build_ace_step_tags(b2_macro_output: PadRequest) -> str:
    # 1. Ruolo funzionale (sempre presente)
    role = "cinematic underscore, no vocals, instrumental"

    # 2. Mood (da MacroAnalysisResult)
    mood_map = {
        "tension":     "dark suspense, tension rising, ominous",
        "action":      "energetic, driving rhythm, intense",
        "melancholy":  "melancholic, introspective, sad",
        "wonder":      "mysterious, ethereal, expansive",
        "resolve":     "triumphant, hopeful, warm",
        "fear":        "horror underscore, unsettling, dissonant",
        "quiet":       "minimal, sparse, meditative",
    }

    # 3. Strumenti (traduzione da terminologia sound-designer)
    instrument_translation = {
        "ARP 2600": "vintage analog synthesizer",
        "brass clusters": "dark brass ensemble",
        "spring reverb": "analog reverb",
        "tape saturation": "vintage tape recording",
        "sub-bass": "deep bass drone",
        "metallic percussion": "metallic percussion, industrial hits",
        "low strings": "low strings, cello section",
    }

    # 4. Stile produzione (da palette_choice in preproduction.json)
    palette_map = {
        "Retro-Futurismo Orchestrale": "1970s analog warmth, retro cinematic, vintage recording",
        "Fantasy Orchestrale": "epic orchestral, full orchestra, dramatic",
        "Thriller Moderno": "modern cinematic, electronic hybrid, tense",
    }

    # 5. Sempre includere: no tempo fisso se unknown, key se disponibile da tonal_lock
    return ", ".join([role, mood_tags, instruments, palette_style, technical])
```

---

## 7. Stage D2: Dispatch verso ARIA

Stage D2 (`src/stages/stage_d2_sound_factory.py`) è il client DIAS che:
1. Riceve la `sound_shopping_list_aggregata.json`
2. Per ogni item costruisce un `AriaTaskPayload` (formato queue Redis ARIA)
3. Invia alla coda Redis e aspetta il callback
4. Scarica il WAV risultante da ARIA HTTP (8082)
5. Per PAD: avvia HTDemucs localmente

### 7.0 Routing per Tipo Asset

Stage D2 determina il `model_id` in base al tipo:

| Asset type | model_id | Backend ARIA | Porta |
|---|---|---|---|
| `pad` | `acestep-1.5-xl-sft` | ACE-Step wrapper | 8084 |
| `leitmotif` | `acestep-1.5-xl-sft` | ACE-Step wrapper | 8084 |
| `amb` | `audiocraft-medium` | Audiocraft wrapper | 8086 |
| `sfx` | `audiocraft-medium` | Audiocraft wrapper | 8086 |
| `sting` | `audiocraft-medium` | Audiocraft wrapper | 8086 |

Tutti i task usano **la stessa coda Redis** — il routing avviene tramite `model_id` nel payload.

### 7.1 Formato Queue Redis — PAD / Leitmotif (ACE-Step)

```
Queue: aria:q:mus:local:acestep-1.5-xl-sft:dias
```

```json
{
  "job_id": "pad_retro_sci_fi_tension_01_chunk000",
  "client_id": "dias",
  "model_type": "mus",
  "model_id": "acestep-1.5-xl-sft",
  "callback_key": "aria:c:dias:pad_retro_sci_fi_tension_01_chunk000",
  "timeout_seconds": 7200,
  "payload": {
    "job_id": "pad_retro_sci_fi_tension_01_chunk000",
    "prompt": "cinematic underscore, no vocals, dark suspense, low strings, vintage analog synthesizer, distant brass ensemble, 1970s analog warmth, key: F minor, tempo: 65bpm",
    "lyrics": "[Intro]\nSub-bass drone, sparse texture, isolated synth shimmer.\n\n[Verse]\nBrass enters gradually, low strings swell, tension builds.\n\n[Chorus]\nFull orchestral climax, dissonant brass, metallic percussion hits.\n\n[Outro]\nDecay to minimal drone, metallic textures fade to silence.",
    "duration": 470.4,
    "seed": 42,
    "guidance_scale": 4.5,
    "inference_steps": 60,
    "output_style": "pad",
    "thinking": true
  }
}
```

**Nota sul campo `lyrics`:** Stage D2 deve costruire la stringa `lyrics` dal `lyrics_arc` del Task Descriptor:

```python
def build_lyrics_string(lyrics_arc: list[dict]) -> str:
    parts = []
    for segment in lyrics_arc:
        parts.append(f"{segment['tag']}\n{segment['description']}")
    return "\n\n".join(parts)
```

### 7.2 Formato Callback Redis (ARIA → DIAS)

```
Key: aria:c:dias:{job_id}
```

```json
{
  "status": "success",
  "job_id": "pad_retro_sci_fi_tension_01_chunk000",
  "audio_url": "http://192.168.1.139:8082/assets/sound_library/pad/pad_retro_sci_fi_tension_01_chunk000/output.wav",
  "local_path": "C:/Users/Roberto/aria/data/assets/sound_library/pad/pad_retro_sci_fi_tension_01_chunk000/output.wav",
  "score_path": "C:/Users/Roberto/aria/data/assets/sound_library/pad/pad_retro_sci_fi_tension_01_chunk000/pad_retro_sci_fi_tension_01_chunk000.score.json",
  "output_style": "pad",
  "duration_seconds": 470.4
}
```

### 7.3 Il Wrapper Gestisce la Relay Internamente

**DIAS non vede i chunk intermedi.** Il wrapper `aria_wrapper_server.py` riceve una singola richiesta con `duration=470.4` e:
1. Spezza in chunk da 120s (4 chunk + 1 parziale)
2. Usa il tail di ciascuno come `reference_audio` del successivo
3. Legge il `score.json` del pt0 per il tonal lock (bpm, keyscale)
4. Assembla via SoX con crossfade 3s
5. Restituisce un singolo WAV di 470.4s

DIAS riceve solo il WAV finale. Il dettaglio della relay è un'implementazione interna di ARIA.

### 7.4 Manifest D2 Prodotto

```json
{
  "project_id": "scalzi-uomini-in-rosso",
  "generated_at": "2026-04-17T14:30:00Z",
  "assets": {
    "pad_retro_sci_fi_tension_01": {
      "type": "pad",
      "master_wav": "/dias/data/projects/scalzi/stages/stage_d2/assets/pad/pad_retro_sci_fi_tension_01/master.wav",
      "duration_s": 470.4,
      "stems": {
        "bass":   "/dias/data/projects/scalzi/stages/stage_d2/assets/pad/pad_retro_sci_fi_tension_01/stems/bass.wav",
        "melody": "/dias/data/projects/scalzi/stages/stage_d2/assets/pad/pad_retro_sci_fi_tension_01/stems/melody.wav",
        "drums":  "/dias/data/projects/scalzi/stages/stage_d2/assets/pad/pad_retro_sci_fi_tension_01/stems/drums.wav"
      },
      "tonal_lock": {
        "bpm": 65,
        "keyscale": "F minor",
        "timesignature": "4"
      },
      "score_json_path": "..."
    },
    "amb_enclosed_spaceship_01": {
      "type": "amb",
      "master_wav": "...",
      "duration_s": 30.0,
      "stems": null
    },
    "sting_reveal_shock_01": {
      "type": "sting",
      "master_wav": "...",
      "duration_s": 4.5,
      "stems": null
    }
  }
}
```

---

## 8. HTDemucs: Stem Separation per il PAD

### 8.1 Perché HTDemucs

ACE-Step genera un **mix completo** (non stem isolati). Per il controllo dinamico dell'intensità in Stage E (bass only → bass+melody → full), è necessario separare il master WAV in stem tramite source separation.

HTDemucs (Facebook Research, modello `htdemucs_6s`) separa:
- `bass.wav` — frequenze basse, fondamentale armonica
- `drums.wav` — percussioni, elementi ritmici
- `other.wav` — armonie, tastiere, trame orchestrali (rinominato `melody`)
- `vocals.wav` — (per PAD strumentale: silenzio o residuo trascurabile)
- `guitar.wav` — eventuali texture chitarristiche
- `piano.wav` — eventuale pianoforte

### 8.2 Attivazione Stem per Intensità (Stage E)

Il `pad_arc` da B2-Macro definisce l'intensità per ogni intervallo temporale. Stage E usa questa mappa:

| Intensità (pad_arc) | Stem attivi | Effetto percepito |
|--------------------|-------------|-------------------|
| `low` | bass | Fondamentale, quasi invisibile, senso di profondità |
| `mid` | bass + melody | Presenza musicale chiara ma non invasiva |
| `high` | bass + melody + drums | Piena potenza orchestrale, picco emotivo |

```python
# Pseudocodice Stage E — attivazione stem per intervallo
def get_active_stems(pad_arc_segment: PadArcSegment) -> list[str]:
    if pad_arc_segment.intensity == "low":
        return ["bass"]
    elif pad_arc_segment.intensity == "mid":
        return ["bass", "melody"]
    else:  # "high"
        return ["bass", "melody", "drums"]
```

### 8.3 Dove Eseguire HTDemucs

**Implementazione attuale (Aprile 2026): Opzione A — Su ARIA (PC 139) post-generazione.**

HTDemucs viene eseguito dall'orchestratore ARIA come subprocess subito dopo la generazione del PAD master, all'interno del backend `ACEStepBackend._run_htdemucs()`:

```python
# aria_node_controller/backends/acestep.py
cmd = [
    str(python_exe), "-m", "demucs",
    "-n", "htdemucs_6s",        # flag corretto: -n, NON --model (deprecato in questa versione)
    "-o", str(stems_dir),
    str(master_path),
]
```

> ⚠️ **Fix Aprile 2026**: la versione di demucs in `dias-sound-engine` usa `-n NAME` (non `--model NAME`). L'uso errato di `--model` causava `rc=2` e `stems: {}` nel callback. Corretto in `acestep.py`.

L'output viene scritto in `stems_dir/htdemucs_6s/{master_name}/` e poi spostato in posizione flat `stems_dir/{stem}.wav`. Gli URL HTTP degli stem sono inclusi nel callback Redis sotto la chiave `stems`.

**Stem prodotti dal PAD** (strumentale — vocals trascurabili):
- `bass.wav` — frequenze basse
- `drums.wav` — elementi ritmici
- `other.wav` — armonie, trame orchestrali

---

## 9. Stage E: Come Consuma gli Asset

Stage E (`src/stages/stage_e_mixdown.py`) è il **mixatore timeline**. ✅ Implementato (Aprile 2026).

Si lancia manualmente al termine di Stage D2:
```bash
python -m src.stages.stage_e_mixdown --project-id {project_id} --chunk 000
```

### 9.1 Input di Stage E

```
1. Master Timing Grid (da Stage D):
   {scene_id → {start_offset_s, voice_duration_s, pause_after_s}}

2. Voice WAVs (da Stage D):
   {scene_id → path_locale.wav}

3. IntegratedCueSheet (da B2-Micro):
   {scenes_automation[] → {pad_duck_depth, pad_fade_speed, amb_id, sfx_id, sting_id}}

4. manifest.json (da Stage D2):
   {canonical_id → {master_wav, stems{bass, melody, drums}}}

5. pad_arc (da MacroCue B2-Macro):
   [{start_s, end_s, intensity}]
```

### 9.2 Algoritmo di Mix Timeline

```
INIZIALIZZAZIONE:
  timeline_duration = master_timing_grid.total_duration_seconds
  Carica tutti i voice WAV sul timeline secondo start_offset_s

TRACCIA PAD (Stem A):
  Per ogni intervallo del pad_arc:
    active_stems = get_active_stems(interval.intensity)
    
    Per ogni stem in active_stems:
      Carica stem WAV da manifest
      Piazza su timeline da interval.start_s a interval.end_s
      Se bordo con intervallo precedente: crossfade 2s tra livelli
    
    Per ogni stem NON in active_stems:
      Fade out progressivo a 0 nel tratto precedente

DUCKING PAD (Per ogni scena):
  Per ogni scena in scenes_automation:
    voice_start = timing_grid[scene_id].start_offset_s
    voice_end   = voice_start + timing_grid[scene_id].voice_duration_s
    
    duck_db = {"shallow": -6, "medium": -12, "deep": -18}[cue.pad_duck_depth]
    attack_ms = {"snap": 300, "smooth": 1000, "slow": 2500}[cue.pad_fade_speed]
    release_ms = 200  # release fisso, naturale
    
    Applica sidechain ducking: PAD scende di duck_db in attack_ms
    Ripristino: quando voce finisce, PAD risale in release_ms
    
    # Pausa speciale: se pause_after > 1.5s → neutral (musica emerge)
    if timing_grid[scene_id].pause_after_s > 1.5:
        pad_fade_to_neutral(speed="slow")

TRACCIA AMB:
  Per ogni scena con amb_id:
    amb_wav = manifest[amb_id].master_wav
    Piazza da scene.start_offset_s + cue.amb_offset_s
    Durata: cue.amb_duration_s o fine della scena
    Crossfade in/out 1.5s
    Volume: -18 dBFS (bed sotto voce e PAD)

TRACCIA SFX:
  Per ogni scena con sfx_id:
    sfx_wav = manifest[sfx_id].master_wav
    offset = timing_grid[scene_id].start_offset_s + cue.sfx_offset_s
    
    if cue.sfx_timing == "start":   piazza a offset
    if cue.sfx_timing == "middle":  piazza a offset + voice_duration/2
    if cue.sfx_timing == "end":     piazza a offset + voice_duration - sfx_duration
    
    Volume: 0 dBFS (nativo), panning per spazializzazione ambiente

TRACCIA STING:
  Per ogni scena con sting_id (max 1 per block):
    sting_wav = manifest[sting_id].master_wav
    
    if cue.sting_timing == "end":    piazza a fine scena
    if cue.sting_timing == "middle": piazza a metà scena
    
    Volume: 0 dBFS, fade in 50ms, nessun ducking (è il momento di impatto)
    PAD: silence totale per durata sting + 500ms

OUTPUT Stage E: stereo WAV per capitolo (non normalizzato)
```

### 9.3 Parametri Broadcast per Ducking (calibrati BBC)

```
Ducking durante voce:  -10 dB (medio BBC Radio 4 standard)
Attack ducking:         20-30 ms (rapido, non copre consonanti iniziali)
Release ducking:        150-250 ms (naturale, non "pompa")
PAD in pausa >1.5s:    restore a 0 dB in 2.5s (musica emerge lentamente)
AMB sempre:            -18 dBFS sotto PAD e voce
STING:                  -3 dBFS peak (impatto ma non clipping)
```

---

## 10. Stage F/G: Mastering e Consegna

### 10.1 Stage F — Processor Audio

Applica catena di finalizzazione per capitolo:

```
Input: WAV stereo da Stage E (24-bit/48kHz, non normalizzato)

Catena:
  1. High-pass su voce: -80 Hz (rimuove rumble da floor noise)
  2. Multiband compressor leggero (bus master): 2:1, -20 dB threshold
  3. Loudness normalization: -16 LUFS (podcast/streaming)
                              -23 LUFS (broadcast ITU-R BS.1770-4)
  4. True Peak limiter: -1.0 dBTP

Output: WAV 16-bit/44.1kHz normalized
```

### 10.2 Stage G — Consegna

```
Formato: MP3 320kbps (distribuzione) + WAV master (archivio)
Metadata ID3: titolo, autore, numero capitolo, durata
Naming: {project_id}_cap{N:02d}_{title_slug}.mp3
```

### 10.3 FFmpeg Chain Completa

```bash
ffmpeg \
  -i mix_chapter_01.wav \
  -af "
    highpass=f=30,
    compand=attacks=0:points=-80/-80|-40/-30|-20/-15|-5/-5|0/0,
    loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json
  " \
  -ar 44100 -sample_fmt s16 \
  output_chapter_01_master.wav

# Poi MP3
ffmpeg -i output_chapter_01_master.wav \
  -codec:a libmp3lame -b:a 320k \
  -id3v2_version 3 \
  -metadata title="Uomini in Rosso - Capitolo 1" \
  -metadata artist="John Scalzi" \
  output_chapter_01.mp3
```

---

## 11. Protocollo Redis Completo

### 11.1 Naming Convention

```
# Code (DIAS → ARIA)
aria:q:mus:local:acestep-1.5-xl-sft:dias    # PAD, MUS
aria:q:voice                                 # TTS (Stage D)

# Callback (ARIA → DIAS)
aria:c:dias:{job_id}

# Master Registry DIAS (tracking interno)
dias:registry:{project_id}                   # Hash Redis

# State bus
dias:state:{project_id}:{stage}             # Stato pipeline
```

### 11.2 Flusso Completo per un PAD

```
DIAS Stage D2                     Redis (LXC 120)          ARIA (PC 139)
─────────────────────────────────────────────────────────────────────────
LPUSH task JSON ──────────────→  aria:q:mus:...:dias
                                                    BRPOP ←──────────────
                                                    Wrapper avvia relay
                                                    pt0: genera 120s
                                                    pt1: genera 120s (ref)
                                                    pt2: genera 120s (ref)
                                                    pt3: genera 110.4s
                                                    SoX stitch → 470.4s
                                                    LPUSH result ────────→
BRPOP (timeout=7200s) ←──────── aria:c:dias:{job_id}
Scarica WAV via HTTP 8082
Esegue HTDemucs (bass/melody/drums)
Scrive manifest.json
```

### 11.3 Formato Task Completo per Queue

```python
# src/stages/stage_d2_sound_factory.py — build_aria_task()
def build_aria_task(item: SoundShoppingItem, task_desc: dict) -> dict:
    lyrics = build_lyrics_string(task_desc["prompt"]["lyrics_arc"])
    
    return {
        "job_id": f"{item.canonical_id}_{project_id}",
        "client_id": "dias",
        "model_type": "mus",
        "model_id": "acestep-1.5-xl-sft",
        "callback_key": f"aria:c:dias:{item.canonical_id}_{project_id}",
        "timeout_seconds": 7200,
        "payload": {
            "job_id": f"{item.canonical_id}_{project_id}",
            "prompt": task_desc["prompt"]["tags"],
            "lyrics": lyrics,
            "duration": item.duration_s or task_desc["generation"]["duration_s"],
            "seed": task_desc["generation"].get("seed", 42),
            "guidance_scale": item.guidance_scale,
            "inference_steps": item.inference_steps,
            "output_style": item.type,  # "pad", "amb", "sfx", "sting"
            "thinking": task_desc["generation"]["thinking"],
        }
    }
```

---

## 12. Esempio End-to-End: Scalzi "Uomini in Rosso"

### 12.1 Input B2-Macro (da Stage B + preproduction.json)

```json
{
  "palette_choice": "Retro-Futurismo Orchestrale",
  "block_analysis": {
    "primary_emotion": "tension",
    "secondary_emotion": "fear",
    "setting": "enclosed_spaceship_interior",
    "arousal": 0.7,
    "valence": -0.6
  },
  "estimated_duration_s": 470.4
}
```

### 12.2 Output B2-Macro → ACE-Step Task Descriptor

```json
{
  "asset": {
    "canonical_id": "pad_retro_sci_fi_tension_01",
    "type": "pad",
    "stem_role": "A"
  },
  "generation": {
    "thinking": true,
    "use_cot_caption": false,
    "inference_steps": 60,
    "guidance_scale": 4.5,
    "duration_s": 470.4,
    "seed": 42
  },
  "prompt": {
    "tags": "cinematic underscore, no vocals, instrumental, dark suspense, tension rising, ominous, low strings cello section, vintage analog synthesizer, distant brass ensemble, 1970s analog warmth, retro cinematic, sustained drone, F minor, 65bpm",
    "negative_prompt": "upbeat, happy, major key, pop, bright, modern production, generic ai music, vocals",
    "lyrics_arc": [
      {"tag": "[Intro]",  "description": "Sub-bass drone, sparse texture, isolated synth shimmer.",          "duration_s": 90},
      {"tag": "[Verse]",  "description": "Brass enters gradually, low strings swell, tension builds.",        "duration_s": 240},
      {"tag": "[Chorus]", "description": "Full orchestral climax, dissonant brass, metallic percussion.",     "duration_s": 100},
      {"tag": "[Outro]",  "description": "Decay to minimal drone, metallic textures fade to silence.",        "duration_s": 40}
    ]
  },
  "tonal_lock": {"bpm": 65, "keyscale": "F minor", "timesignature": "4"},
  "relay": {"enabled": true, "chunk_s": 120.0, "tail_s": 15.0, "crossfade_s": 3.0},
  "post_processing": {"htdemucs": true, "stems_requested": ["bass", "melody", "drums"]}
}
```

### 12.3 Output B2-Micro (per scena di dialogo intensa)

```json
{
  "scene_id": "chunk-000-micro-002-scene-004",
  "pad_volume_automation": "ducking",
  "pad_duck_depth": "deep",
  "pad_fade_speed": "snap",
  "amb_id": "amb_enclosed_spaceship_01",
  "amb_offset_s": 0.0,
  "sfx_id": "sfx_alarm_alert_01",
  "sfx_timing": "start",
  "sfx_offset_s": 0.5,
  "sting_id": null,
  "reasoning": "Dialogo di crisi, massimo spazio per la voce, allarme sincronizzato all'inizio scena"
}
```

### 12.4 Stage E usa i dati così

```
Timeline 0s ────────────────────────────── 470.4s
                                             
[PAD bass]  ══════════════════════════════  (sempre attivo)
[PAD melody]    ════════════════════════    (da t=90s, intensity≥mid)
[PAD drums]                 ══════════      (da t=330s, intensity=high)
                                             
[Voce]      ─[s1]──[s2]──[s3]──[s4]──...   (da Master Timing Grid)
[Ducking]   ▼▼▼    ▼▼▼    ▼▼▼    ▼▼▼        (da IntegratedCueSheet)
                                             
[AMB ship]  ════════════════════════════    (-18 dBFS, sotto tutto)
[SFX alarm]                 ●               (a t=scene4.start+0.5s)
```

---

## 13. Checklist Implementativa

### Lato DIAS (da implementare)

- [ ] **`stage_d2_sound_factory.py`**: aggiungere metodo `build_ace_step_task_descriptor()` che converte `SoundShoppingItem` + `PadRequest` nel Task Descriptor v1.0
- [ ] **`stage_d2_sound_factory.py`**: aggiungere `build_lyrics_string()` che converte `lyrics_arc` in stringa ACE-Step
- [ ] **`stage_d2_sound_factory.py`**: aggiungere `build_prompt_tags()` che traduce `production_tags` B2 in vocabolario ACE-Step (dizionario di traduzione)
- [ ] **`stage_d2_sound_factory.py`**: integrazione HTDemucs post-download per PAD
- [ ] **`src/common/models.py`**: aggiungere `AceStepTaskDescriptor` come Pydantic model con tutti i campi del contratto
- [ ] **`stage_b2_macro.py`**: aggiornare il prompt Gemini per usare il dizionario di traduzione sound-designer → musicista
- [ ] **`stage_e_mixer.py`**: implementare da zero usando il design di questo documento
- [ ] **`config/sound_taxonomy.yaml`**: aggiungere mapping `intensity → stems_attivi`

### Lato ARIA (già implementato nel wrapper v2.0)

- [x] Relay chunking (120s, tail 15s, crossfade 3s)
- [x] `use_cot_caption=False` nel TOML generato
- [x] stdin bypass (`b"\n"*20`) per eliminare blocco interattivo
- [x] Cleanup `instruction.txt` dopo ogni chunk
- [x] Tonal lock via `lm_metadata` da `score.json`
- [x] `os.replace()` per Windows-safe file operations
- [x] `output_name` via TOML (non CLI arg)

### Verifica Cross-System

- [ ] Test con PAD 470.4s: verificare che DIAS riceva un singolo WAV (non 4 chunk separati)
- [ ] Verificare che `tonal_lock` dal Task Descriptor sia propagato correttamente al TOML pt0
- [ ] Verificare struttura `score.json` (`lm_metadata.bpm`, `lm_metadata.keyscale`)
- [ ] Test HTDemucs: verificare che i 3 stem siano coerenti con il master
- [ ] Test Stage E (mock): sovrapporre manualmente bass+melody+ducking su una scena e ascoltare

---

*Documento generato in sessione ARIA/DIAS integrazione — Aprile 2026*
*Riferimenti: `aria_wrapper_server.py` (v2.0), DIAS `docs/blueprint.md` (v6.3), `docs/stage_b2_d2_sound_on_demand_v4.md` (v4.0), `docs/pad_music_architecture_analysis.md`*
