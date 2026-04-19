# Audiocraft Backend — AudioGen & MusicGen
## ARIA Sound Engine per AMB, SFX, STING

**Porta**: 8086  
**Ambiente**: `envs/dias-sound-engine` (env unificato con ACE-Step e Demucs)  
**Script**: `backends/audiocraft/aria_audiocraft_server.py`  
**Stato**: ✅ Operativo (Aprile 2026)  
**Documenti correlati**: [ARIA Service Registry](../ARIA-Service-Registry.md) · [DIAS Sound Integration](../DIAS-ARIA-ACEStep-Integration.md)

---

## Panoramica

Il backend Audiocraft gestisce la generazione di suoni ambientali, effetti sonori e stacchi musicali tramite due modelli Meta AI in-process:

| Modello | Asset type | Caratteristiche |
|---|---|---|
| **AudioGen medium** | `amb`, `sfx` | Generazione audio non-musicale da testo: ambienti, texture, effetti fisici |
| **MusicGen large** | `sting` | Generazione musicale da testo: brevi sequenze melodiche, transizioni |

Il wrapper fa routing interno in base al campo `output_style` nel payload — il client (DIAS Stage D2) non deve sapere quale modello viene usato.

---

## Architettura

```
DIAS Stage D2
    │  model_id: "audiocraft-medium"
    │  output_style: "amb" | "sfx" | "sting"
    ▼
Redis: aria:q:mus:local:acestep-1.5-xl-sft:dias
    ▼
ARIA Orchestrator
    │  dispatch: _process_audiocraft_task()
    │  JIT: ensure_running("audiocraft-medium") → porta 8086
    ▼
AudiocraftBackend.run()  [aria_node_controller/backends/audiocraft.py]
    │  POST http://127.0.0.1:8086/generate
    ▼
aria_audiocraft_server.py  [FastAPI, porta 8086]
    ├─ output_style ∈ {amb, sfx}  → AudioGen.get_pretrained("facebook/audiogen-medium")
    └─ output_style = sting        → MusicGen.get_pretrained("facebook/musicgen-large")
         │  generate([prompt])
         │  resample → 44100 Hz stereo
         └─ save: data/assets/sound_library/{style}/{job_id}/{job_id}.wav
```

---

## Payload (D2 → Redis)

```json
{
  "job_id":       "d2-amb-a1b2c3d4e5",
  "client_id":    "dias",
  "model_type":   "mus",
  "model_id":     "audiocraft-medium",
  "callback_key": "aria:c:dias:d2-amb-a1b2c3d4e5",
  "timeout_seconds": 900,
  "payload": {
    "job_id":       "d2-amb-a1b2c3d4e5",
    "prompt":       "deep space station hum, metallic resonance, low frequency drone",
    "duration":     4.0,
    "seed":         42,
    "output_style": "amb"
  }
}
```

### Campi payload

| Campo | Tipo | Default | Note |
|---|---|---|---|
| `prompt` | string | obbligatorio | Descrizione semantica del suono in inglese |
| `duration` | float | 5.0 | Durata in secondi (AudioGen max ~30s, MusicGen max ~30s) |
| `seed` | int | 42 | -1 = casuale |
| `output_style` | string | "amb" | `amb` \| `sfx` \| `sting` — determina il modello |

---

## Output e Callback

```json
{
  "status": "done",
  "job_id": "d2-amb-a1b2c3d4e5",
  "output": {
    "audio_url": "http://192.168.1.139:8082/assets/sound_library/amb/d2-amb-a1b2c3d4e5/d2-amb-a1b2c3d4e5.wav",
    "duration_seconds": 4.0
  },
  "processing_time_seconds": 12.3
}
```

L'audio è salvato in: `ARIA_ROOT/data/assets/sound_library/{style}/{job_id}/{job_id}.wav`  
Formato: **WAV stereo 44100 Hz** (normalizzato dal wrapper da qualsiasi sample rate nativo).

---

## Dettagli Tecnici

### Caricamento Modelli (JIT)
I modelli vengono caricati **per ogni request** e scaricati subito dopo per liberare VRAM:

```python
model = AudioGen.get_pretrained("facebook/audiogen-medium")
model.set_generation_params(duration=req.duration)
wav = model.generate([req.prompt])   # [1, channels, T]
del model
torch.cuda.empty_cache()
```

Questo garantisce che 8084 (ACE-Step) e 8086 (Audiocraft) non competano per VRAM — sono avviati in momenti diversi e ogni task libera la GPU al completamento.

### Cache Modelli
La variabile `HF_HUB_CACHE` è impostata a `ARIA_ROOT/data/assets/models/audiocraft/` all'avvio del wrapper. I modelli scaricati vengono mantenuti in quella directory.

- MusicGen Large: già presente (`models--facebook--musicgen-large/`)
- AudioGen Medium: scaricato da HuggingFace al primo avvio (~1.5 GB)

### Output Audio
- AudioGen output nativo: mono 16kHz
- MusicGen output nativo: stereo 32kHz  
- Output normalizzato dal wrapper: **stereo 44100 Hz** (compatibile Stage E)

### Lock concorrenza
Il wrapper usa un `asyncio.Lock()` per garantire un solo task alla volta — necessario perché il modello occupa GPU.

---

## Manifest Entry (`backends_manifest.json`)

```json
"audiocraft-medium": {
  "port": 8086,
  "health_url": "http://localhost:8086/health",
  "startup_wait": 30,
  "env_prefix": "envs/dias-sound-engine",
  "working_dir": "backends/audiocraft",
  "script": "backends/audiocraft/aria_audiocraft_server.py",
  "args": ["--host", "127.0.0.1", "--port", "8086"]
}
```

---

## Linee Guida Prompt

### AMB (AudioGen)
Descrizioni di ambienti fisici, texture sonore, atmosfere:
```
"deep space station ambient, low mechanical hum, metallic resonance, distant ventilation"
"alien forest at night, strange insects chirping, low wind, eerie silence"
"enclosed metallic vessel, engine drone, subtle water dripping"
```

### SFX (AudioGen)
Effetti sonori discreti, eventi fisici, suoni di oggetti:
```
"heavy stone impact, sharp crack, low rumble decay"
"large bio-creature movement, wet organic sounds, heavy footsteps"
"mechanical weapon fire, sharp metallic click, brief recoil"
```

### STING (MusicGen)
Brevi sequenze musicali emotivamente caratterizzate:
```
"dramatic orchestral sting, brass and strings, sudden revelation, 2 seconds"
"mysterious sci-fi sting, synthesizer pulse, dark ambiance, short"
"tense cinematic hit, percussion accent, suspense build"
```

---

## Note Implementative

- **Env unificato**: `dias-sound-engine` ospita sia ACE-Step (tramite wrapper) che Audiocraft (in-process). Non è necessario un env separato.
- **`audiocraft-env` deprecato**: il vecchio ambiente `audiocraft-env` è stato ritirato in Aprile 2026 — tutti i modelli audiocraft sono ora in `dias-sound-engine`.
- **flash_attn rimosso**: `flash_attn 2.8.2` era installato in `dias-sound-engine` e causava DLL failure su sm_120 (`ImportError: DLL load failed`). Rimosso in Aprile 2026. xformers 0.0.35 funziona correttamente senza flash_attn (margine di velocità trascurabile su generazioni brevi).
