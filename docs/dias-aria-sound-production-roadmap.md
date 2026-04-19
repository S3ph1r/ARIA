# DIAS × ARIA — Sound Production Architecture & Roadmap
## Versione 1.0 — Aprile 2026
**Autori**: Roberto + Claude Sonnet 4.6  
**Scope**: Architettura completa del sistema di produzione sonora DIAS×ARIA, principi di sound design, roadmap di sviluppo separata per DIAS e ARIA.

---

## 1. Principi Fondamentali

### 1.1 Separazione delle responsabilità

**DIAS non conosce l'interno di ARIA.** DIAS invia richieste in code Redis con tipo e parametri, aspetta il callback, scarica il risultato e avanza negli stage. Cosa fa ARIA internamente (quale modello usa, come gestisce la VRAM, quanti chunk genera) è irrilevante per DIAS.

**ARIA non conosce la narrativa di DIAS.** ARIA riceve una specifica tecnica (prompt, durata, tipo, seed) e restituisce un WAV. Non sa che quel PAD è il capitolo 3 di un audiolibro, non sa chi è il personaggio del leitmotif.

**I stage di DIAS sono sequenziali.** Ogni stage parte solo quando il precedente è completato. Questo semplifica la gestione delle dipendenze ed è sufficiente per la produzione di audiolibri.

### 1.2 Sound design — glossario e ruoli

| Tipo | Natura | Chi sente | Durata | Funzione |
|---|---|---|---|---|
| **PAD** | Musicale, non-diegetico | Solo pubblico | = capitolo | Tessuto emotivo continuo del capitolo |
| **STING** | Musicale, non-diegetico | Solo pubblico | 3-8s | Accento drammatico puntuale (rivelazione, morte, shock) |
| **Leitmotif base** | Musicale, non-diegetico | Solo pubblico | 20-30s | Tema fisso di un personaggio, generato una volta |
| **Leitmotif variation** | Musicale, non-diegetico | Solo pubblico | 8-20s | Variazione orchestrale del tema base adattata all'emozione della scena |
| **AMB bed** | Naturalistico, diegetico | Anche personaggi | Durata scena/location | Texture sonora di luogo (caverna, foresta, navicella) — quasi impercettibile |
| **AMB event** | Naturalistico, diegetico | Anche personaggi | 1-5s | Evento atmosferico puntuale (tuono, folla, vento improvviso) |
| **SFX** | Naturalistico, diegetico | Anche personaggi | 0.3-3s | Effetto sonoro preciso (spada, porta, passi) |

**Distinzione chiave diegetico/non-diegetico:**
- Diegetico = esiste nel mondo della storia (AMB, SFX). I personaggi lo percepiscono.
- Non-diegetico = esiste solo per il pubblico (PAD, STING, leitmotif). I personaggi non lo sentono.

Questa distinzione guida le decisioni di mix: diegetico e non-diegetico convivono senza competere perché operano su piani semantici diversi.

### 1.3 Livelli di mix di riferimento (EBU R128 target -23 LUFS)

| Track | Volume base | Duck durante voce | Note |
|---|---|---|---|
| Voce | 0 dB | — | Reference assoluta |
| PAD bass stem | -20 dB | mai duckato | Fondamenta armoniche sempre presenti |
| PAD drums stem | -23 dB | -14 dB aggiuntivi | Più aggressivo, disturba la voce |
| PAD other stem | -20 dB | -8/-14 dB aggiuntivi | Melodia/armonia, ducking moderato |
| AMB bed | -28/-32 dB | -3 dB aggiuntivi | Quasi impercettibile, si "sente" solo se manca |
| AMB event | -18 dB | — | Puntuale, non duckato |
| SFX | -12 dB | — | Nitido, puntuale |
| STING | -6 dB | — | Impatto massimo, breve |
| Leitmotif variation | -9 dB | -6 dB aggiuntivi | Sotto la voce ma riconoscibile |

### 1.4 Comportamento AMB bed — rilevamento automatico

AMB bed vs AMB event non richiede un campo esplicito. Stage E lo inferisce dal pattern di assegnazione di B2:
- **Stessa `amb_id` in scene consecutive** = bed → Stage E loopa il clip per la durata cumulativa delle scene
- **`amb_id` unica in una singola scena** = event → Stage E piazza il clip una volta sola

B2 assegna naturalmente lo stesso `amb_id` a tutte le scene in un'unica location, producendo il comportamento corretto senza metadati aggiuntivi.

---

## 2. Architettura del Pipeline Completo

```
Stage 0    → preproduction.json
               palette_choice, characters_dossier, theatrical_standard

Stage 0.5  → Theme Factory [DA IMPLEMENTARE — solo chiamate LLM, nessuna ARIA]
               Per ogni personaggio primary/secondary:
                 1. Chiama Gemini con profilo personaggio + palette
                 2. Ottiene: musical_profile + prompt ACE-Step + tags + seed
               Aggiorna preproduction.json con project_sound_palette
               Output: solo testo/JSON — NESSUN WAV, NESSUNA chiamata ARIA
               
Stage A    → Text Ingestion
Stage B    → Semantic Analysis  
Stage C    → Scene Direction (chi c'è in ogni scena, timing)

Stage B2   → Copione sonoro completo del libro [SCHEMA DA AGGIORNARE]
  B2-Macro:  PAD request per ogni capitolo (prompt, arc, durata)
             Legge project_sound_palette → può usare leitmotif come reference_audio
  B2-Micro:  IntegratedCueSheet per ogni scena:
             - Ducking automation PAD
             - AMB assignment (bed auto-rilevato da pattern consecutive scenes)
             - sfx_events: lista con offset_s preciso
             - sting_events: lista con offset_s
             - leitmotif_events: lista con character_id, base_id, variation_emotion, offset_s
  Output:    shopping_list_aggregata.json (intero libro)

Stage D    → Voice WAV per ogni scena (chiama ARIA TTS — già completo)

Stage D2   → Sound Factory Client [DA COMPLETARE]
               Unico caller ARIA per musica/suono. Ordine batch fisso:
               1. Leitmotif bases (ACE-Step, da project_sound_palette — se non già in locale)
               2. PAD per capitolo (ACE-Step, da B2-Macro)
               3. STING (MusicGen, da B2-Micro)
               4. Leitmotif variations (MusicGen melody-conditioned, usa base WAV locale)
               5. AMB (AudioGen, da B2-Micro)
               6. SFX (AudioGen, da B2-Micro)
               Per ogni callback: SCARICA WAV → stage_d2/assets/{type}/{canonical_id}.wav
               Segnala completamento quando queue esaurita

Stage E    → Mixdown Engine [IMPLEMENTATO, in attesa di asset]
               Legge IntegratedCueSheet (partitura di B2)
               Assembla canvas: PAD + voce + AMB (con loop bed) + SFX + STING + leitmotif
               Applica ducking automation per scena
               Normalizza e produce WAV master
               Multi-mode: voice / music / music+fx / full / all
```

### Principio chiave: ARIA calls solo via D2

Nessuno stage prima di D2 chiama la parte music/sound di ARIA.
Stage D chiama ARIA per la voce (TTS, coda separata) — indipendente.
Stage 0.5 chiama solo Gemini (LLM gateway) — nessun servizio audio.

---

## 3. Contratto Dati B2 → Stage E (IntegratedCueSheet v2)

Lo schema attuale è da aggiornare per supportare multi-evento per scena e leitmotif variation.

### Schema attuale (v1):
```json
{
  "scene_id": "chunk-000-micro-001-scene-003",
  "pad_duck_depth": "medium",
  "pad_fade_speed": "smooth",
  "amb_id": "amb_enclosed_cave_01",
  "sfx_id": "sfx_impact_stone_01",
  "sfx_timing": "middle",
  "sting_id": null
}
```

### Schema target (v2):
```json
{
  "scene_id": "chunk-000-micro-001-scene-003",
  "pad_volume_automation": "ducking",
  "pad_duck_depth": "medium",
  "pad_fade_speed": "smooth",
  "amb_events": [
    {"amb_id": "amb_enclosed_cave_01", "offset_s": 0.0}
  ],
  "sfx_events": [
    {"sfx_id": "sfx_impact_stone_01", "offset_s": 12.5},
    {"sfx_id": "sfx_body_fall_01", "offset_s": 28.0}
  ],
  "sting_events": [
    {"sting_id": "sting_tragedy_01", "offset_s": -2.0}
  ],
  "leitmotif_events": [
    {
      "leitmotif_base_id": "leitmotif_kovacs_base",
      "variation_emotion": "fear",
      "offset_s": 5.0
    }
  ]
}
```

Dove `offset_s` è in secondi dall'inizio della scena (negativo = dalla fine).

---

## 4. Roadmap DIAS

### Fase 1 — Sblocca il primo mixdown completo (URGENTE)

**Stage 0.5 — Theme Factory** ← PRIMO DA IMPLEMENTARE
- File: `src/stages/stage_0_5_theme_factory.py`
- Runner: `tests/stages/run_theme_factory.py`
- Prompt: `config/prompts/stage_0_5/leitmotif_prompt_v1.0.yaml`
- Input: `preproduction.json` (characters_dossier, palette_choice, theatrical_standard)
- Per ogni personaggio `primary` (e opzionalmente `secondary`):
  - Chiama Gemini con profilo personaggio + palette
  - Riceve: musical_profile, generation_prompt, generation_tags, seed, duration_s
- Aggiorna `preproduction.json` con `project_sound_palette`
- Idempotente: se `project_sound_palette` già popolato, salta
- Nessuna chiamata ARIA, nessun WAV

**D2 download step**
- Dopo ogni callback ARIA, D2 scarica il WAV da `audio_url` e lo salva in `stage_d2/assets/{type}/{canonical_id}.wav`
- Senza questo, Stage E non trova nessun asset su disco
- File: `src/stages/stage_d2_sound_factory.py`
- Complessità: bassa (requests.get + save)

**Stage E — AMB loop engine**
- Se stessa `amb_id` in scene consecutive → Stage E loopa il clip per la durata cumulativa
- File: `src/stages/stage_e_mixdown.py` → metodo `_layer_fx()`
- Complessità: media

**Test end-to-end voice-only**
- `python tests/stages/run_stage_e.py urania_... --mode voice`
- Verifica che i WAV di Stage D siano posizionati correttamente
- Non richiede ARIA, solo Stage D output

### Fase 2 — Qualità del copione sonoro

**Stage 0.5 — Theme Factory**
- File: `src/stages/stage_0_5_theme_factory.py`
- Runner: `tests/stages/run_theme_factory.py`
- Genera 1 tema base per ogni personaggio principale (da `characters_dossier`)
- Invia ad ARIA (tipo `leitmotif_creation`) → WAV permanenti in `stage_d2/assets/leitmotif/`
- Aggiorna `preproduction.json` con `project_sound_palette`
- Idempotente: se `project_sound_palette` già popolato, salta

**B2-Micro schema v2**
- Aggiorna prompt YAML `b2_micro_v2.yaml` per produrre schema v2
- Multi-slot per AMB/SFX/STING
- Campo `leitmotif_events` con `variation_emotion`
- Offset in secondi al posto di stringhe "start/middle/end"
- Richiede Stage C come input (ha i personaggi per scena)

**Stage E — multi-evento e leitmotif**
- Aggiorna `_layer_fx()` per iterare liste `sfx_events`, `sting_events`, `leitmotif_events`
- Aggiunge `_layer_leitmotif()` per variazioni melody-conditioned

### Fase 3 — Arc alignment

**B2-Macro arc alignment ai micro-chunk boundaries**
- Modifica `_get_chunk_total_duration()` → `_get_chunk_timing_data()`
- Ritorna durata + lista boundaries micro-chunk
- Aggiorna prompt v4.1: segmenti arc devono coincidere con boundaries
- File: `src/stages/stage_b2_macro.py`

### Fase 4 — Ottimizzazioni future

- Asset library cross-progetto: se `sfx_impact_stone_01` esiste già da un progetto precedente, non rigenerarlo
- Resampling automatico in Stage E (oggi assume stesso SR)
- Export MP3 da Stage E
- EBU R128 loudness normalization vera (oggi peak normalization)

---

## 5. Roadmap ARIA

### Fase 1 — Routing interno per tipo asset (PRIORITARIA)

**Integrazione MusicGen in ARIA**
- Nuovo wrapper: `backends/musicgen/musicgen_wrapper.py` (HTTP service)
- O: routing interno nel connector ACE-Step esistente
- Gestisce: STING (3-8s), leitmotif_creation (20-30s), leitmotif_variation (melody-conditioned)
- Env: `audiocraft-env` (già installato con MusicGen)

**Integrazione AudioGen in ARIA**
- Nuovo wrapper: `backends/audiogen/audiogen_wrapper.py`
- Gestisce: AMB (5-60s), SFX (0.3-5s)
- Env: `audiocraft-env` (AudioGen già presente)

**Router nel node controller**
- `aria_node_controller/backends/` aggiunge routing per tipo:
  ```
  pad              → ACE-Step
  sting            → MusicGen
  leitmotif_creation  → ACE-Step (qualità alta, una volta sola)
  leitmotif_variation → MusicGen melody-conditioned
  amb              → AudioGen
  sfx              → AudioGen
  ```
- File: `aria_node_controller/backends/router.py` (nuovo) o estende `orchestrator.py`

**Batch processing per tipo (shopping list)**
- ARIA processa tutti i job PAD prima di caricare MusicGen
- Poi tutti STING + leitmotif, poi tutti AMB + SFX
- Non serve concorrenza — staging sequenziale per tipo riduce swap VRAM
- Richiede code Redis separate per tipo (già parzialmente in atto con la naming convention `aria:q:mus:local:acestep-1.5-xl-sft:dias`)

### Fase 2 — Consolidamento environment

**Valutazione merge audiocraft-env → dias-sound-engine**
- Pro: un env unico, gestione semplificata
- Pro: batching sequenziale significa no conflict VRAM
- Contro: rischio rottura dipendenze CUDA (sm_120 già risolto ma fragile)
- Decisione: testare su branch separato prima di mergiare in produzione
- Prerequisito: Fase 1 funzionante e testata con env separati

**HTDemucs — già funzionante**
- `demucs 4.0.1` installato in `dias-sound-engine`
- Pesi `htdemucs_6s` (53MB) ora in cache: `~/.cache/torch/hub/checkpoints/5c90dfd2-34c22ccb.th`
- La prossima run con `run_demucs: true` funzionerà senza download
- Nessuna azione richiesta

### Fase 3 — Qualità e leitmotif variation

**MusicGen melody conditioning**
- Implementazione del meccanismo: dato un WAV base (leitmotif_base), MusicGen genera variazione che segue la stessa melodia ma con orchestrazione diversa
- Input: `reference_audio` (WAV base), `prompt` (emozione/contesto), durata
- Output: WAV 8-20s con melodia riconoscibile ma colore orchestrale adattato
- Questo è il cuore dei leitmotif cinematografici

**Cache asset per progetto**
- ARIA mantiene un indice degli asset già generati
- Se `sfx_impact_stone_01` esiste nel sound library, serve il file esistente invece di rigenerarlo
- Riduce drasticamente i tempi di produzione per progetti successivi

---

## 6. Matrice di dipendenze — cosa blocca cosa

```
D2 download step
  └─→ sblocca Stage E test con asset reali
  └─→ sblocca verifica qualità AMB/SFX/STING

ARIA MusicGen integration
  └─→ sblocca produzione STING di qualità
  └─→ sblocca leitmotif variation

ARIA AudioGen integration  
  └─→ sblocca produzione AMB/SFX rapida (secondi vs minuti)

Stage 0.5 Theme Factory
  └─→ richiede ARIA MusicGen (per leitmotif_variation in futuro)
  └─→ richiede ACE-Step (per leitmotif_creation base)
  └─→ sblocca continuità timbrica inter-capitolo
  └─→ sblocca B2 leitmotif_events

B2-Micro schema v2
  └─→ richiede Stage 0.5 completato (leitmotif_base_id noti)
  └─→ sblocca multi-SFX per scena
  └─→ sblocca offset precisi in secondi

Stage E AMB loop
  └─→ richiede D2 download step (serve il WAV su disco)
  └─→ sblocca AMB bed funzionante in Stage E
```

---

## 7. Stato corrente (18 Aprile 2026)

### ARIA — completato
| Componente | Stato |
|---|---|
| ACE-Step wrapper server | ✅ Funzionante, 4 fix qualità applicati |
| Relay chunking (120s chunks) | ✅ Crossfade 8s, threshold -45dB |
| HTDemucs pesi scaricati | ✅ `htdemucs_6s` in cache |
| PAD Urania chunk-000 (470s) | ✅ Generato, `d2-pad-1f227b79b8.wav` su ARIA |
| Symlink modelli ACE-Step | ✅ vae + Qwen3-Embedding senza duplicati |
| MusicGen integration | ❌ Da implementare |
| AudioGen integration | ❌ Da implementare |
| Routing per tipo | ❌ Da implementare |

### DIAS — completato
| Componente | Stato |
|---|---|
| master_timing_grid.json | ✅ Per-scena, misure WAV reali |
| Stage B2 (Macro + Micro) | ✅ Funzionante, schema v1 |
| Stage D2 dispatch | ✅ Funzionante, manca download post-callback |
| Stage E Mixdown Engine | ✅ Implementato, in attesa di asset su disco |
| Stage E multi-mode export | ✅ voice/music/music+fx/full/all |
| Stage 0.5 Theme Factory | ❌ Da implementare |
| D2 download step | ❌ Da implementare |
| B2-Micro schema v2 | ❌ Da implementare |
| Stage E AMB loop | ❌ Da implementare |

---

## 8. Prossimi 3 passi operativi concreti

**1. [ARIA] Scarica il PAD già generato**
```bash
# Su LXC 190 (DIAS)
wget http://192.168.1.139:8082/assets/sound_library/pad/d2-pad-1f227b79b8/d2-pad-1f227b79b8.wav \
  -O data/projects/urania_.../stages/stage_d2/assets/pad/d2-pad-1f227b79b8.wav
```
Sblocca il test `--mode music` e `--mode full` di Stage E.

**2. [DIAS] D2 download step**
Aggiungere in `stage_d2_sound_factory.py` il download post-callback:
```python
def _download_asset(self, audio_url: str, canonical_id: str, asset_type: str) -> Path:
    dest = self.assets_dir / asset_type / f"{canonical_id}.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(audio_url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest
```

**3. [DIAS] Test Stage E --mode voice**
```bash
python tests/stages/run_stage_e.py urania_... --mode voice
```
Non richiede ARIA. Verifica timing grid, offset assoluti, sovrapposizione WAV voice.
Ascoltare il risultato e validare la traccia vocale prima di procedere con il mix completo.

---

*Documento cristallizzato il 18 Aprile 2026*  
*Copia in: `aria/docs/dias-aria-sound-production-roadmap.md`*  
*Copia in: `dias/docs/dias-aria-sound-production-roadmap.md`*
