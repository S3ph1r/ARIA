# Backend `lifelog_whisperx_v2` — STT di Lifelog2 dal 2026-09-26

**Stato**: in produzione. La voce `whisperx-large-v3` del manifest punta a questo backend (ARIA 484071e);
il v1 (`lifelog-whisperx.md`) resta come voce di riserva `whisperx-large-v3-v1-riserva`, mai avviata.
**Contratto con il client**: Lifelog2 `docs/lifelog2-asr-v2-contract.md` (approvato da Roberto il 2026-09-26).
**Perché**: laboratorio e misure in Lifelog2 `docs/lifelog2-audit-completo.md`, blocchi dal 2026-09-25.

## Cosa fa

| passo | come | perché (misura) |
|---|---|---|
| Trascrizione | faster-whisper large-v3 **non batched**, beam 5, condizionamento sul testo precedente, fallback di temperatura, `word_timestamps=True`, `vad_filter=True`, lingua `it` se il client manda `None` | il batched di whisperx perdeva il 15–18% delle parole e non ha fallback; senza condizionamento il testo peggiora (radio dei carabinieri «giulietto…papà») |
| Tempi delle parole | ctc-forced-aligner (MMS 300m, `MahmoudAshraf/mms-300m-1130-forced-aligner`) su tutto il testo, durata massima 1.2 s per parola; se il numero di token non torna → tempi di Whisper (`align_fallback`) | wav2vec2 dentro i blocchi del batched metteva il 18–27% delle parole fuori posto di oltre 1 s; confermato a orecchio |
| Diarizzazione | pyannote community-1 come nel v1, esposta **grezza** (`intervals`) con i tratti a ≥2 voci (`overlaps`) | Lifelog non usa le etichette pyannote per l'identità (le riusa per persone diverse) |
| Impronte | WeSpeaker ResNet293 256d sull'audio delle sole parole (±50 ms) senza i tratti sovrapposti: per **frase** Whisper e su **finestre** 1.5 s ogni 0.5 s; float16 base64; minimo 0.5 s di voce | l'etichetta per parola viene dal voto delle finestre (55/2/14 su verità a orecchio contro 6/2/63 della parola singola) |
| Identità | **nessuna**: il backend non conosce voci di persone | decisione Roberto: ARIA misura, Lifelog decide |

## Richiesta

`POST /transcribe` `{wav_url, segment_id, language?, min_speakers?, max_speakers?, exclusive_diarization?,
compat_v1=false, embed_words=false}`.
- `compat_v1=true`: aggiunge anche `speaker_turns` e impronte per turno del formato v1 (solo confronti; +10–20 s).
- `embed_words=true`: impronta anche per ogni parola (solo misure di laboratorio).

## Risposta (modalità snella, default)

`output = {transcript, language, duration_ms, speaker_turns: [], diarization_stats,
transcription_quality: {avg_logprob_mean}, v2: {version, timing_s, asr{segments}, align_fallback, words[],
diarization{intervals, overlaps}, embeddings{phrases[], windows{size_ms, step_ms, items[]}}}}` — schema
completo nel contratto. Circa 400–500 KB per 5 minuti.

## Ambiente e pesi

- Env `envs\lifelog-whisperx-v2` = `conda create --clone envs\lifelog-whisperx` + `pip install
  git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git` (0.3.0; la 1.0.2 di PyPI non linka su Windows:
  `LNK2001 PyInit_align_ops`). Nient'altro di diverso (verificato con `pip freeze`).
- Pesi: faster-whisper, pyannote e ResNet293 come il v1 in `data\assets\models`; il modello ctc MMS si scarica
  in `data\assets\models` (HF_HOME).
- Porta 8091 (la stessa del v1: i due non girano mai insieme).

## Memoria GPU (misura 2026-09-26)

Senza interventi la cache di torch cresceva a ogni segmento: 5.9 GB a modelli caricati, 10 GB dopo il primo
ctc, poi 12 → 14 → 15.4 GB con le impronte (16 GB totali). Vicino al limite Windows sposta memoria nella RAM di
sistema e il segmento rallenta 4–5× (657b: 230 s invece di 50). Correzione: `torch.cuda.empty_cache()` dopo ctc
e dopo le impronte, impronte a blocchi da 64. Dopo: picco 11.0 GB stabile, impronte identiche (coseno 1.0).

## Tempi (PC 139, 5 minuti di audio, esclusi i caricamenti)

Caricamento modelli ~30 s una volta per batch. Per segmento 46–109 s (mediana ~60 s): trascrizione 24–84 s
(varia tra giri a testo identico, probabile contesa di CPU con l'uso interattivo), ctc 1–2 s, pyannote ~7 s,
impronte 2.5–5 s. Il v1 aveva mediana 26 s: sul batch da 6 segmenti Stage C passa da ~3 a ~6.5 min ogni 30 min
di audio (dentro il tempo reale; decisione Roberto: precisione prima del tempo).

## Tornare al v1

Nel manifest copiare `env_prefix` e `script` della voce `whisperx-large-v3-v1-riserva` nella voce
`whisperx-large-v3` e fare `git pull` sul PC 139: l'orchestratore rilegge il manifest a ogni avvio di backend.
Lifelog riconosce da solo i grezzi senza blocco `v2` (percorso C1 v1).
