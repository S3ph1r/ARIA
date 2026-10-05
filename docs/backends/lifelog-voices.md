# Backend `lifelog_voices` — voci e «riprodotto» per Lifelog (dal 2026-10-05)

**Stato:** in prova, modalità ombra. Lifelog salva le uscite, C1 e D non le usano ancora.
**Perché e misure:** Lifelog2 `docs/lifelog3-roadmap-modelli-ascolto.md` (test 5–9); flusso e fasi in `docs/lifelog3-pipeline-voci.md`.
**Identità:** nessuna. Le voci sono etichette anonime del segmento (S0, S1…): ARIA misura, Lifelog decide (come `lifelog_whisperx_v2`).

## Cosa fa

| passo | come | perché (misura) |
|---|---|---|
| Voci e sovrapposizioni | DiariZen (BUT) `wavlm-large-s80-md-v2`: segmentazione WavLM + embedding ResNet34 + clustering VBx | su carabinieri/TG Putin: voce giusta 77/99% contro 71/94% di pyannote. Usato da C1 come veto: falsi utente 48 → 0 sul TG Putin |
| «Riprodotto» per turno | WavLM base-plus, strato 3, media e dev. std sui fotogrammi del turno → standardizzazione → PCA 64 → regressione logistica (`replay_wavlm-base-plus_L3_bench_v1`) | AUC 0,88 su segmenti mai visti (banco A, verità provvisoria); sul formato nuovo i turni dell'utente risultano dal vivo, la TV riprodotta |

## Richiesta e risposta

`POST /analyze {wav_url, segment_id, replay=true, min_turn_s=1.0}`: WAV 16 kHz mono (lo stesso di Stage B).

```json
{"output": {"version": "voices-v1", "status": "ok | no_speech", "segment_id": "...", "duration_s": 299.9,
            "models": {"diarization": "...", "replay": {"name": "...", "layer": 3, "train": "..."}},
            "tracks": [[start_s, end_s, "S1"]],          // intervalli per voce, possono sovrapporsi
            "overlaps": [[start_s, end_s, n_voci]],      // tratti con ≥2 voci
            "turns": [{"start": s, "end": e, "voice": "S1", "overlap": false, "p_replay": 0.93}],
            "timing_s": {"diarization": 17.0, "replay": 2.0, "total": 19.5}}}
```

- **`status: no_speech`:** DiariZen va in errore sui segmenti senza parlato («negative dimensions» in `reconstruct`). Il server lo intercetta e risponde vuoto. Lifelog dovrebbe comunque non mandarli (filtro Silero in Stage B).
- **Coda:** `aria:q:voices:local:lifelog-voices:lifelog`. Nel manifest è la voce `lifelog-voices`, porta **8094**. La 8093 si evita per il port-forwarding di Firebase Studio (vedi `ARIA-Service-Registry.md`).

## Ambiente e pesi (convenzioni ARIA)

- **Env:** `envs\lifelog-voices`, nuovo, Python 3.10.
  1. `conda create -p envs\lifelog-voices python=3.10`
  2. `pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128`. Non il 2.1.1+cu121 ufficiale di DiariZen: non supporta sm_120.
  3. `pip install -r docs\backends\lifelog-voices.requirements.txt -c (torch==2.8.0, numpy==1.26.4)`
  4. DiariZen in `envs\diarizen-src`, commit `844f5555b0a98acd0931511fc641a8c5b8ba92c7`: `pip install --no-deps -e .`, poi `pip install -e pyannote-audio` (il suo fork di pyannote 3.1.1).
- **Pesi** in `data\assets\models`:
  - `diarizen\wavlm-large-s80-md-v2\` (config.toml, pytorch_model.bin, plda/; licenza **CC BY-NC 4.0**, uso personale);
  - `pyannote\wespeaker-voxceleb-resnet34-LM\pytorch_model.bin`, già presente per il v1;
  - `wavlm\wavlm-base-plus\`;
  - `lifelog-voices\replay_wavlm-base-plus_L3_bench_v1.npz`: classificatore in numpy puro, niente scikit-learn nell'env.
- **Variabili:** `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (impostata dal server: i checkpoint sono salvati con torch 2.1); arch spoof sm_120 → (9, 0) come il v2.

## Memoria e tempi (PC 139, 5 minuti di audio)

Picco 11 GiB (DiariZen); diarizzazione 17–40 s, WavLM ~2 s. Non può stare in GPU insieme a WhisperX v2 (11 GiB): l'orchestratore li alterna, Lifelog manda i task a lotti (≥6 segmenti o 30 minuti, poi drena).
