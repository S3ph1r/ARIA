# DIAS → ARIA: Integration Specification
## Cosa arriva da DIAS, cosa deve produrre ARIA
**Versione**: 1.0 — 17 Aprile 2026
**Scope**: Protocollo Redis, formato payload, output attesi, gap implementativi aperti.

---

## 1. Panoramica del flusso

```
DIAS (LXC 120)                          ARIA (PC 139 — Win11, RTX 5060 Ti 16GB)
─────────────────────────────────────   ──────────────────────────────────────────
Stage B2-Macro   →  MacroCue (PAD)
Stage B2-Director → SoundEventScore
Stage B2-Engineer → IntegratedCueSheet
Stage D2          → sound_shopping_list_aggregata.json
                 ──────── Redis (LXC 120:6379) ────────→
                    LPUSH aria:q:mus:local:acestep-1.5-xl-sft:dias
                                                         NodeOrchestrator
                                                           ACEStepBackend
                                                           aria_wrapper_server.py
                                                           cli.py (ACE-Step)
                 ←─────── Redis callback ────────────────
                    BRPOP aria:c:dias:{job_id}
Stage D2         ←  HTTP download 192.168.1.139:8082
Stage E          →  mixdown finale
```

---

## 2. Protocollo Redis

| Parametro | Valore |
|---|---|
| Redis host | 192.168.1.120:6379 |
| Coda task | `aria:q:mus:local:acestep-1.5-xl-sft:dias` |
| Callback | `aria:c:dias:{job_id}` |
| Comando push | `LPUSH` (task JSON serializzato) |
| Comando wait | `BRPOP` con timeout |

---

## 3. Struttura del task Redis (inviato da D2)

```json
{
  "job_id":          "d2-pad-a1b2c3d4e5",
  "client_id":       "dias",
  "model_type":      "mus",
  "model_id":        "acestep-1.5-xl-sft",
  "callback_key":    "aria:c:dias:d2-pad-a1b2c3d4e5",
  "timeout_seconds": 7200,
  "payload": {
    "job_id":          "d2-pad-a1b2c3d4e5",
    "prompt":          "cinematic underscore, no vocals, instrumental, ...",
    "lyrics":          "[00:01] [Intro] Minimal bass drone ...\n[02:01] [Verse] ...",
    "duration":        620.0,
    "seed":            42,
    "guidance_scale":  4.5,
    "inference_steps": 60,
    "output_style":    "pad",
    "thinking":        true,
    "run_demucs":      true
  }
}
```

---

## 4. I quattro tipi di asset — cosa riceve ARIA, cosa deve produrre

### 4.1 PAD (tappeto musicale)

**Chi lo genera**: ACE-Step 1.5 XL SFT — **già implementato e funzionante**.

**Payload caratteristico**:
```json
{
  "output_style":    "pad",
  "duration":        600.0,
  "guidance_scale":  4.5,
  "inference_steps": 60,
  "thinking":        true,
  "run_demucs":      true,
  "prompt":          "cinematic underscore, no vocals, instrumental, 1970s orchestral, retro sci-fi, dark suspense, low strings, vintage analog synthesizer, slow tempo, F minor, ominous",
  "lyrics":          "[00:01] [Intro] Solo cello drone, sparse strings\n[02:01] [Verse] Bass and strings building tension\n[08:01] [Pre-Chorus] Strings densify, brass enters\n[10:01] [Chorus] Full orchestra, maximum intensity\n[14:01] [Outro] Fade to bass drone"
}
```

**Output atteso da ARIA**:
- `audio_url`: master WAV (HTTP 8082)
- `stems.bass`: WAV stem basse frequenze
- `stems.drums`: WAV stem percussioni
- `stems.other`: WAV stem melodia/armonia

**Durata tipica**: 5-20 minuti. Relay automatico ogni 120s con SoX crossfade.
**Timeout D2**: 7200s.
**Stato**: ✅ Completamente implementato (relay, tonal lock, HTDemucs).

---

### 4.2 AMB (cue ambientale transitionale)

**Chi lo genera attualmente**: ACE-Step 1.5 XL SFT.
**Chi potrebbe generarlo**: modello alternativo ottimizzato per sound effects (da decidere — vedi §6).

**Payload caratteristico**:
```json
{
  "output_style":    "amb",
  "duration":        4.0,
  "guidance_scale":  7.0,
  "inference_steps": 60,
  "thinking":        false,
  "run_demucs":      false,
  "prompt":          "cinematic underscore, no vocals, instrumental, short transitional cue, establishing ambience, urban exterior, city street, light vehicle traffic, distant pedestrian voices, fade in 0.5s, fade out 1.5s, total 4s, non-looping, no music, no melody, no rhythm, no drums",
  "lyrics":          ""
}
```

**Output atteso da ARIA**:
- `audio_url`: master WAV (HTTP 8082), 3-5s, già sfumato in/out

**Durata tipica**: 3-5s. Single-shot (< RELAY_CHUNK_S=120s).
**Timeout D2**: 900s.
**Stato**: ⚠️ Funziona ma con 2 gap:
  1. **Fade non garantite**: il prompt chiede fade ma ACE-Step può ignorarle. Serve post-proc SoX lato ARIA dopo generazione.
  2. **Tempo di generazione**: ~4.5 min per 4s di output (overhead LM fisso di ACE-Step).

---

### 4.3 SFX (effetto puntuale)

**Chi lo genera attualmente**: ACE-Step 1.5 XL SFT.
**Chi potrebbe generarlo**: modello alternativo ottimizzato per sound effects (da decidere — vedi §6).

**Payload caratteristico**:
```json
{
  "output_style":    "sfx",
  "duration":        0.4,
  "guidance_scale":  7.0,
  "inference_steps": 60,
  "thinking":        false,
  "run_demucs":      false,
  "prompt":          "cinematic underscore, no vocals, instrumental, sound effect, isolated, sharp, metal pistol firing, single gunshot, dry crack, no music, no reverb, no ambient, no echo, mono",
  "lyrics":          ""
}
```

**Output atteso da ARIA**:
- `audio_url`: master WAV (HTTP 8082), durata = `duration` richiesto

**Durata tipica**: 0.3-2.0s. Single-shot.
**Timeout D2**: 900s.
**Stato**: ⚠️ Funziona ma con 2 gap:
  1. **Durata non trimmed**: ACE-Step genera spesso più lungo del richiesto. Serve trim post-generazione a `duration_s` esatto.
  2. **Tempo di generazione**: ~4.5 min per 0.4s di output. Inefficiente ma accettato in questa fase.

---

### 4.4 STING (accento orchestrale drammatico)

**Chi lo genera attualmente**: ACE-Step 1.5 XL SFT.
**Chi potrebbe generarlo**: ACE-Step rimane il candidato migliore (è musicale, non un semplice effetto).

**Payload caratteristico**:
```json
{
  "output_style":    "sting",
  "duration":        3.0,
  "guidance_scale":  4.5,
  "inference_steps": 60,
  "thinking":        false,
  "run_demucs":      false,
  "prompt":          "cinematic underscore, no vocals, instrumental, dramatic sting, orchestral accent, sharp attack, short, dark revelation, low strings, deep brass, dissonant harmony, no sustained pad, 1970s orchestral, Bernard Herrmann influence",
  "lyrics":          ""
}
```

**Output atteso da ARIA**:
- `audio_url`: master WAV (HTTP 8082), 2-4s

**Durata tipica**: 2-4s. Single-shot.
**Timeout D2**: 900s.
**Stato**: ✅ Funziona. ACE-Step gestisce bene i brevi accenti orchestrali. Nessun post-proc necessario.

---

## 5. Struttura callback da ARIA verso DIAS

### Callback PAD (con stems):
```json
{
  "job_id":    "d2-pad-a1b2c3d4e5",
  "client_id": "dias",
  "status":    "done",
  "output": {
    "audio_url":        "http://192.168.1.139:8082/assets/sound_library/pad/d2-pad-a1b2c3d4e5/d2-pad-a1b2c3d4e5.wav",
    "duration_seconds": 618.4,
    "stems": {
      "bass":  "http://192.168.1.139:8082/assets/sound_library/pad/d2-pad-a1b2c3d4e5/stems/bass.wav",
      "drums": "http://192.168.1.139:8082/assets/sound_library/pad/d2-pad-a1b2c3d4e5/stems/drums.wav",
      "other": "http://192.168.1.139:8082/assets/sound_library/pad/d2-pad-a1b2c3d4e5/stems/other.wav"
    }
  }
}
```

### Callback AMB/SFX/STING (solo master):
```json
{
  "job_id":    "d2-sfx-b2c3d4e5f6",
  "client_id": "dias",
  "status":    "done",
  "output": {
    "audio_url":        "http://192.168.1.139:8082/assets/sound_library/sfx/d2-sfx-b2c3d4e5f6/d2-sfx-b2c3d4e5f6.wav",
    "duration_seconds": 0.43
  }
}
```

### Callback errore:
```json
{
  "job_id":    "d2-pad-a1b2c3d4e5",
  "client_id": "dias",
  "status":    "error",
  "error":     "cli.py exited with code 1"
}
```

---

## 6. Decisione architetturale aperta: modello per AMB/SFX

ACE-Step è eccellente per musica e STING orchestrali. Per AMB (4s di rumore urbano) e SFX (0.4s colpo di pistola) l'overhead di ~240s LM è sproporzionato.

**Alternativa da valutare**: modello dedicato sound effects (es. AudioCraft/AudioGen, Stable Audio Open, EzAudio) con inferenza in 5-15s per clip brevi.

### Impatto sull'integrazione ARIA se si introduce un secondo modello:

| Aspetto | ACE-Step (attuale) | Modello SFX alternativo |
|---|---|---|
| Queue Redis | `aria:q:mus:local:acestep-1.5-xl-sft:dias` | nuova coda `aria:q:sfx:local:{model}:dias` |
| output_style | pad / amb / sfx / sting | amb / sfx |
| Payload | production_tags come `prompt` | prompt diverso (no Qwen3 vocab) |
| Tempo generazione | ~4.5 min/asset | ~15-30s/asset |
| STING | resta su ACE-Step | resta su ACE-Step |

**Decisione**: non cambia nulla in DIAS (B2 produce già i prompt corretti).
Cambia solo il routing in ARIA (quale modello riceve quale `output_style`).
D2 potrebbe usare due queue Redis distinte o un campo `model_override` nel task.

---

## 7. Gap implementativi aperti lato ARIA

### Gap A — Fade AMB (priorità: alta)
**Dove**: `backends/acestep/aria_wrapper_server.py`, dopo `_run_single_shot()` per `output_style=amb`
**Cosa fare**: applicare fade in/out via SoX dopo generazione
```python
if style == "amb":
    _apply_fades(audio_path, fade_in=0.5, fade_out=1.5)
```
```python
def _apply_fades(wav: Path, fade_in: float, fade_out: float):
    sox = shutil.which("sox")
    duration = _get_wav_duration(wav)
    tmp = wav.with_suffix(".fade.wav")
    subprocess.run([sox, str(wav), str(tmp), "fade", "t",
                    str(fade_in), str(duration), str(fade_out)], check=True)
    os.replace(str(tmp), str(wav))
```

### Gap B — Trim SFX a durata esatta (priorità: media)
**Dove**: `backends/acestep/aria_wrapper_server.py`, dopo `_run_single_shot()` per `output_style=sfx`
**Cosa fare**: trim a `req.duration` secondi esatti via SoX
```python
if style == "sfx":
    _trim_to_duration(audio_path, req.duration)
```
```python
def _trim_to_duration(wav: Path, duration_s: float):
    sox = shutil.which("sox")
    tmp = wav.with_suffix(".trim.wav")
    subprocess.run([sox, str(wav), str(tmp), "trim", "0",
                    str(duration_s)], check=True)
    os.replace(str(tmp), str(wav))
```

### Gap C — Modello alternativo per AMB/SFX (priorità: bassa, fase 2)
**Dove**: nuovo backend `aria_node_controller/backends/{sfx_model}.py`
**Cosa fare**: da decidere quale modello, poi implementare il connettore seguendo il pattern di `acestep.py`

---

## 8. File locali prodotti da D2 (struttura manifest)

Dopo download da ARIA, D2 salva in:
```
data/projects/{project_id}/stages/stage_d2/assets/
  pad/
    {canonical_id}.wav           ← master PAD
    stems/{canonical_id}/
      bass.wav
      drums.wav
      other.wav
  amb/
    {canonical_id}.wav
  sfx/
    {canonical_id}.wav
  sting/
    {canonical_id}.wav
```

Il manifest `stages/stage_d2/manifest.json` mappa `canonical_id → local_paths`:
```json
{
  "project_id": "urania_n_1610_...",
  "assets": {
    "pad_retro_scifi_tension_01": {
      "type": "pad",
      "master_path": "/.../assets/pad/pad_retro_scifi_tension_01.wav",
      "stems": {
        "bass":  "/.../assets/pad/stems/pad_retro_scifi_tension_01/bass.wav",
        "drums": "/.../assets/pad/stems/pad_retro_scifi_tension_01/drums.wav",
        "other": "/.../assets/pad/stems/pad_retro_scifi_tension_01/other.wav"
      },
      "duration_s": 618.4,
      "status": "ready"
    },
    "sfx_impact_gunshot_01": {
      "type": "sfx",
      "master_path": "/.../assets/sfx/sfx_impact_gunshot_01.wav",
      "duration_s": 0.4,
      "status": "ready"
    }
  }
}
```

---

## 9. Checklist prima del primo invio reale

- [ ] ARIA wrapper server attivo su PC 139 (porta 8084)
- [ ] ARIA asset server attivo su PC 139 (porta 8082)
- [ ] Redis LXC 120 raggiungibile da entrambi i nodi
- [ ] Modelli ACE-Step scaricati: `data/assets/models/acestep-5Hz-lm-1.7B` e `acestep-v15-xl-sft`
- [ ] HTDemucs installato nell'env `dias-sound-engine`
- [ ] SoX installato su PC 139 (per relay crossfade + Gap A/B)
- [ ] Gap A (fade AMB) implementato nel wrapper
- [ ] Gap B (trim SFX) implementato nel wrapper
- [ ] Test con 1 PAD (chunk piccolo, ~60s) prima del run completo

*Ultimo aggiornamento: 17 Aprile 2026 — v1.0*
