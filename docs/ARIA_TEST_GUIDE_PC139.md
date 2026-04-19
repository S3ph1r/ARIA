# ARIA — Guida Test su PC 139
## Da eseguire su PC 139 (Win11, RTX 5060 Ti 16GB)
**Progetto di test**: Uomini in Rosso — John Scalzi
**Data preparazione**: 17 Aprile 2026
**Status DIAS**: payload pronti, non ancora inviati a Redis.

---

## 1. Contesto — Cosa arriva da DIAS

DIAS ha prodotto 6 asset da generare. I payload Redis sono già costruiti e validati.

### Asset da produrre

| # | Tipo | canonical_id | Durata | Priorità test |
|---|---|---|---|---|
| 1 | PAD | `pad_retro_scifi_tension_01` | 470s (~7.8 min) | Test DOPO gli altri |
| 2 | AMB | `amb_enclosed_cave_01` | 4.0s | Test 1° (breve, verifica pipeline) |
| 3 | SFX | `sfx_impact_stone_01` | 0.4s | Test 2° |
| 4 | SFX | `sfx_impact_body_fall_01` | 0.3s | Test 3° |
| 5 | SFX | `sfx_impact_explosion_01` | 1.5s | Test 4° |
| 6 | STING | `sting_tragedy_01` | 2.0s | Test 5° |

**Ordine consigliato**: AMB → SFX × 3 → STING → PAD.
Motivo: gli asset brevi (3-60s) confermano la pipeline in ~5 min ciascuno.
Il PAD (470s) usa il relay e HTDemucs — va testato per ultimo.

---

## 2. Prerequisiti ARIA da verificare prima dei test

```
[ ] aria_wrapper_server.py in ascolto su 127.0.0.1:8084
    → GET http://127.0.0.1:8084/health deve rispondere {"status": "ok"}

[ ] Asset HTTP server in ascolto su 0.0.0.0:8082
    → GET http://192.168.1.139:8082/ deve rispondere (anche 404 va bene)

[ ] NodeOrchestrator attivo e in ascolto su Redis 192.168.1.120:6379
    → coda: aria:q:mus:local:acestep-1.5-xl-sft:dias

[ ] Modelli presenti:
    → data/assets/models/acestep-5Hz-lm-1.7B/
    → data/assets/models/acestep-v15-xl-sft/

[ ] Env Python dias-sound-engine attivo:
    → envs/dias-sound-engine/python.exe deve esistere

[ ] SoX installato (usato dal relay per crossfade e dai fix fade/trim):
    → where sox  (deve trovare il path)

[ ] HTDemucs nel env dias-sound-engine (solo per PAD):
    → envs/dias-sound-engine/python.exe -m demucs --help
```

---

## 3. Due fix da implementare nel wrapper PRIMA dei test

Entrambi vanno in `backends/acestep/aria_wrapper_server.py`.

### Fix A — Fade garantite per AMB

Aggiungere queste due funzioni e chiamarle in `_run_single_shot()`:

```python
def _apply_fades(wav: Path, fade_in: float, fade_out: float):
    """Applica fade in/out via SoX. Garantisce sfumatura indipendentemente da ACE-Step."""
    sox = shutil.which("sox")
    if not sox:
        print("[ARIA Wrapper] ⚠  SoX non trovato — fade AMB non applicato", flush=True)
        return
    duration = _get_wav_duration(wav)
    tmp = wav.with_suffix(".fade.wav")
    result = subprocess.run(
        [sox, str(wav), str(tmp), "fade", "t",
         str(fade_in), str(duration), str(fade_out)],
        capture_output=True
    )
    if result.returncode == 0:
        os.replace(str(tmp), str(wav))
        print(f"[ARIA Wrapper] ✓  Fade AMB applicato: in={fade_in}s out={fade_out}s", flush=True)
    else:
        print(f"[ARIA Wrapper] ⚠  Fade SoX fallito: {result.stderr.decode()}", flush=True)
        if tmp.exists():
            tmp.unlink()
```

```python
def _trim_to_duration(wav: Path, duration_s: float):
    """Taglia il WAV alla durata esatta richiesta. Necessario per SFX brevi (<1s)."""
    actual = _get_wav_duration(wav)
    if actual <= duration_s + 0.05:  # già nella tolleranza
        return
    sox = shutil.which("sox")
    if not sox:
        print("[ARIA Wrapper] ⚠  SoX non trovato — trim SFX non applicato", flush=True)
        return
    tmp = wav.with_suffix(".trim.wav")
    result = subprocess.run(
        [sox, str(wav), str(tmp), "trim", "0", str(duration_s)],
        capture_output=True
    )
    if result.returncode == 0:
        os.replace(str(tmp), str(wav))
        print(f"[ARIA Wrapper] ✓  Trim SFX: {actual:.2f}s → {duration_s}s", flush=True)
    else:
        print(f"[ARIA Wrapper] ⚠  Trim SoX fallito: {result.stderr.decode()}", flush=True)
        if tmp.exists():
            tmp.unlink()
```

**Dove chiamarle** — in `_run_single_shot()`, dopo aver trovato `audio_path` e prima del `return`:

```python
    audio_path = _find_output_audio(warehouse_dir, job_id, req.audio_format)
    if not audio_path:
        ...

    # ── Post-processing per tipo ──────────────────────────────────────
    if req.output_style == "amb":
        _apply_fades(audio_path, fade_in=0.5, fade_out=1.5)
    elif req.output_style == "sfx":
        _trim_to_duration(audio_path, req.duration)
    # ─────────────────────────────────────────────────────────────────

    score_path = _find_score_json(warehouse_dir, job_id)
    ...
```

### Fix B — Considerazione futura: modello alternativo per AMB/SFX

ACE-Step impiega ~4.5 min per qualsiasi durata (overhead LM fisso).
Per AMB 4s e SFX 0.3s questo è sproporzionato.

Modelli alternativi da valutare (fase 2):
- **EzAudio** — sound effects, 10-30s di inferenza
- **AudioGen** (Meta) — environmental sounds
- **Stable Audio Open** — general audio

Per ora ACE-Step va bene per validare la qualità. La velocità si ottimizza dopo.

---

## 4. Payload pronti per Redis

I payload sono già costruiti da DIAS. Puoi inviarli manualmente con lo script
Python qui sotto, oppure aspettare che DIAS li invii via Stage D2.

### Script test manuale (da eseguire su PC 139)

Salva questo file come `test_single_asset.py` nella root di ARIA:

```python
"""
Test manuale: invia UN singolo payload ad ARIA e aspetta il callback.
Usa uno dei payload dal file d2_dry_run_payloads.json.

Eseguire con: python test_single_asset.py <indice>
  indice 0 = PAD (attenzione: 7200s timeout, usa relay)
  indice 1 = AMB (consigliato come primo test)
  indice 2 = SFX stone
  indice 3 = STING
  indice 4 = SFX body fall
  indice 5 = SFX explosion
"""
import json, sys, redis, time

REDIS_HOST = "192.168.1.120"
REDIS_PORT = 6379

# Incollare qui il contenuto di d2_dry_run_payloads.json
PAYLOADS = """ <<INCOLLARE QUI IL JSON DEI PAYLOAD>> """

def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    items = json.loads(PAYLOADS)
    item = items[idx]

    cid     = item["canonical_id"]
    queue   = item["redis_queue"]
    cbk     = item["callback_key"]
    timeout = item["timeout_s"]
    task    = json.dumps(item["redis_task"])

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    print(f"\n{'='*60}")
    print(f"  Asset     : {cid}")
    print(f"  Tipo      : {item['type']}")
    print(f"  Durata    : {item['redis_task']['payload']['duration']}s")
    print(f"  Queue     : {queue}")
    print(f"  Callback  : {cbk}")
    print(f"  Timeout   : {timeout}s")
    print(f"{'='*60}\n")

    print("📤 Invio task a Redis...")
    r.lpush(queue, task)

    print(f"⏳ Attesa callback (max {timeout}s)...")
    t_start = time.time()
    result = r.brpop(cbk, timeout=timeout)
    elapsed = time.time() - t_start

    if not result:
        print(f"\n❌ Timeout ({timeout}s) — nessun callback ricevuto.")
        return

    _, data = result
    resp = json.loads(data)
    print(f"\n✅ Callback ricevuto in {elapsed:.1f}s")
    print(json.dumps(resp, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
```

---

## 5. Payload JSON completi

### PAD — pad_retro_scifi_tension_01
```json
{
  "job_id": "d2-pad-1f227b79b8",
  "client_id": "dias",
  "model_type": "mus",
  "model_id": "acestep-1.5-xl-sft",
  "callback_key": "aria:c:dias:d2-pad-1f227b79b8",
  "timeout_seconds": 7200,
  "payload": {
    "job_id": "d2-pad-1f227b79b8",
    "prompt": "cinematic underscore, no vocals, instrumental, 1970s orchestral, retro sci-fi, dark suspense, low strings, vintage analog synthesizer, dissonant brass ensemble, analog warmth, slow tempo, minor key, ominous, heavy metallic percussion, Bernard Herrmann influence, tense atmosphere",
    "lyrics": "[00:01.00] [Intro] Sparse cello drone, quiet vintage synthesizer textures, lonely atmosphere, no percussion\n[02:31.00] [Pre-Chorus] Layered analog strings, subtle dissonant brass swells, increasing tension, steady slow tempo\n[05:51.00] [Chorus] Rapidly thickening brass clusters, metallic industrial percussion builds, intense ominous harmony\n[07:01.00] [Outro] Fading metallic strikes, lingering low synthesizer note, decaying analog reverb, silence",
    "duration": 470.4,
    "seed": 42,
    "guidance_scale": 4.5,
    "inference_steps": 60,
    "output_style": "pad",
    "thinking": true,
    "run_demucs": true
  }
}
```
**Contesto artistico**: Capitolo 1 di Uomini in Rosso — Davis Perry isolato su una roccia marziana, minacciato dai vermi borgoviani sotterranei. Archi gravi retro-sci-fi, climax all'attacco. PAD arc: low (isolamento) → mid (tensione crescente) → high (attacco vermi) → low (shock silenzioso).

---

### AMB — amb_enclosed_cave_01
```json
{
  "job_id": "d2-amb-8d5a86bbfb",
  "client_id": "dias",
  "model_type": "mus",
  "model_id": "acestep-1.5-xl-sft",
  "callback_key": "aria:c:dias:d2-amb-8d5a86bbfb",
  "timeout_seconds": 900,
  "payload": {
    "job_id": "d2-amb-8d5a86bbfb",
    "prompt": "short transitional cue, establishing ambience, enclosed cave, stone walls, distant water dripping, low reverb, cold damp air, fade in 0.5s, fade out 1.5s, total 4s, non-looping, no music, no melody",
    "lyrics": "",
    "duration": 4.0,
    "seed": 42,
    "guidance_scale": 7.0,
    "inference_steps": 60,
    "output_style": "amb",
    "thinking": false,
    "run_demucs": false
  }
}
```
**Post-proc richiesto**: applicare fade in 0.5s / fade out 1.5s via SoX (Fix A).

---

### SFX — sfx_impact_stone_01
```json
{
  "job_id": "d2-sfx-74068b90c3",
  "client_id": "dias",
  "model_type": "mus",
  "model_id": "acestep-1.5-xl-sft",
  "callback_key": "aria:c:dias:d2-sfx-74068b90c3",
  "timeout_seconds": 900,
  "payload": {
    "job_id": "d2-sfx-74068b90c3",
    "prompt": "sound effect, isolated, sharp, hand hitting solid rock, stone impact, deep resonance, no music, no reverb, no ambient, no echo, mono",
    "lyrics": "",
    "duration": 0.4,
    "seed": 42,
    "guidance_scale": 7.0,
    "inference_steps": 60,
    "output_style": "sfx",
    "thinking": false,
    "run_demucs": false
  }
}
```
**Post-proc richiesto**: trim a 0.4s esatti via SoX (Fix B).

---

### SFX — sfx_impact_body_fall_01
```json
{
  "job_id": "d2-sfx-5cc9973ff9",
  "client_id": "dias",
  "model_type": "mus",
  "model_id": "acestep-1.5-xl-sft",
  "callback_key": "aria:c:dias:d2-sfx-5cc9973ff9",
  "timeout_seconds": 900,
  "payload": {
    "job_id": "d2-sfx-5cc9973ff9",
    "prompt": "sound effect, isolated, sharp, hard impact of skull against rock, dull thud, no music, no reverb, no ambient, no echo, mono",
    "lyrics": "",
    "duration": 0.3,
    "seed": 42,
    "guidance_scale": 7.0,
    "inference_steps": 60,
    "output_style": "sfx",
    "thinking": false,
    "run_demucs": false
  }
}
```
**Post-proc richiesto**: trim a 0.3s esatti via SoX (Fix B).

---

### SFX — sfx_impact_explosion_01
```json
{
  "job_id": "d2-sfx-9e3bc3929e",
  "client_id": "dias",
  "model_type": "mus",
  "model_id": "acestep-1.5-xl-sft",
  "callback_key": "aria:c:dias:d2-sfx-9e3bc3929e",
  "timeout_seconds": 900,
  "payload": {
    "job_id": "d2-sfx-9e3bc3929e",
    "prompt": "sound effect, isolated, sharp, ground breaking open, muffled deep boom, creature hissing and skittering, no music, no reverb, no ambient, no echo, mono",
    "lyrics": "",
    "duration": 1.5,
    "seed": 42,
    "guidance_scale": 7.0,
    "inference_steps": 60,
    "output_style": "sfx",
    "thinking": false,
    "run_demucs": false
  }
}
```
**Post-proc richiesto**: trim a 1.5s via SoX (Fix B).

---

### STING — sting_tragedy_01
```json
{
  "job_id": "d2-sti-dd45a09fb5",
  "client_id": "dias",
  "model_type": "mus",
  "model_id": "acestep-1.5-xl-sft",
  "callback_key": "aria:c:dias:d2-sti-dd45a09fb5",
  "timeout_seconds": 900,
  "payload": {
    "job_id": "d2-sti-dd45a09fb5",
    "prompt": "dramatic sting, orchestral accent, sharp attack, short, tragic revelation, low strings, ominous brass, dark, no vocals, no singing, no sustained pad, retro sci-fi",
    "lyrics": "",
    "duration": 2.0,
    "seed": 42,
    "guidance_scale": 4.5,
    "inference_steps": 60,
    "output_style": "sting",
    "thinking": false,
    "run_demucs": false
  }
}
```
**Post-proc**: nessuno richiesto. ACE-Step gestisce bene stings orchestrali brevi.

---

## 6. Output attesi dopo generazione

### Per AMB/SFX/STING (callback Redis):
```json
{
  "job_id": "d2-amb-8d5a86bbfb",
  "status": "done",
  "output": {
    "audio_url": "http://192.168.1.139:8082/assets/sound_library/amb/d2-amb-8d5a86bbfb/d2-amb-8d5a86bbfb.wav",
    "duration_seconds": 4.0
  }
}
```

### Per PAD (callback Redis con stems):
```json
{
  "job_id": "d2-pad-1f227b79b8",
  "status": "done",
  "output": {
    "audio_url": "http://192.168.1.139:8082/assets/sound_library/pad/d2-pad-1f227b79b8/d2-pad-1f227b79b8.wav",
    "duration_seconds": 468.5,
    "stems": {
      "bass":  "http://192.168.1.139:8082/assets/sound_library/pad/d2-pad-1f227b79b8/stems/bass.wav",
      "drums": "http://192.168.1.139:8082/assets/sound_library/pad/d2-pad-1f227b79b8/stems/drums.wav",
      "other": "http://192.168.1.139:8082/assets/sound_library/pad/d2-pad-1f227b79b8/stems/other.wav"
    }
  }
}
```

### Dove trovi i file fisici su PC 139:
```
C:\Users\roberto\aria\data\assets\sound_library\
  amb\d2-amb-8d5a86bbfb\
    d2-amb-8d5a86bbfb.wav        ← 4s, con fade
  sfx\d2-sfx-74068b90c3\
    d2-sfx-74068b90c3.wav        ← 0.4s, trimmed
  sting\d2-sti-dd45a09fb5\
    d2-sti-dd45a09fb5.wav        ← 2s orchestral
  pad\d2-pad-1f227b79b8\
    d2-pad-1f227b79b8.wav        ← master 470s
    stems\
      bass.wav
      drums.wav
      other.wav
```

---

## 7. Criteri di valutazione qualità

### AMB (amb_enclosed_cave_01)
- [ ] Si sente dripping acqua lontano
- [ ] Riverbero da spazio chiuso in pietra
- [ ] Fade in udibile (non attacco secco)
- [ ] Fade out udibile (non taglio netto)
- [ ] Durata ~4s
- [ ] Nessun elemento melodico o ritmico

### SFX impatto pietra (sfx_impact_stone_01)
- [ ] Suono secco di impatto fisico
- [ ] Nessuna coda di riverbero artificiale
- [ ] Nessun elemento musicale
- [ ] Durata ~0.4s (non > 1s)

### SFX caduta corpo (sfx_impact_body_fall_01)
- [ ] Suono sordo, basso di massa che cade
- [ ] Durata ~0.3s

### SFX esplosione/creatura (sfx_impact_explosion_01)
- [ ] Boom sordo grave + presenza biologica/organica
- [ ] Non deve sembrare una bomba classica — è terreno che si apre
- [ ] Durata ~1.5s

### STING (sting_tragedy_01)
- [ ] Attacco netto, non sfumato
- [ ] Timbro orchestrale scuro (archi + ottoni)
- [ ] NO sustain prolungato (deve finire in ~2s)
- [ ] Nessuna voce

### PAD (pad_retro_scifi_tension_01)
- [ ] Coerenza timbrica retro-sci-fi per tutti i ~7.8 min
- [ ] Percezione dell'arco: intro quieto → tensione crescente → picco → dissolvenza
- [ ] Transizioni tra chunk non percepibili (crossfade SoX)
- [ ] Stems HTDemucs presenti e separati correttamente
- [ ] Bass: solo frequenze basse, niente melodia
- [ ] Drums: solo percussioni
- [ ] Other: melodia, archi, synth

---

## 8. Decisione architetturale da prendere durante i test

Dopo aver ascoltato i risultati AMB e SFX, decidere:

**Opzione A** — ACE-Step per tutto (attuale): qualità buona ma ~4.5 min per ogni asset breve.
**Opzione B** — Modello dedicato SFX/AMB per clip < 10s: 15-30s inferenza. Richiede nuovo backend ARIA.

La decisione impatta sul routing in ARIA ma non cambia nulla in DIAS (i payload sono già pronti).

---

*Preparato da DIAS (LXC 120) — 17 Aprile 2026*
*Workspace DIAS: `/home/Projects/NH-Mini/sviluppi/dias`*
*Prossimo step dopo i test: implementare Stage E (mixdown finale)*
