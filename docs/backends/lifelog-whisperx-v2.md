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

## [2026-09-27] Filtro del parlato: `silero_recupero` di default

Il VAD Silero ai valori di default di faster-whisper scartava TV e radio in sottofondo (dry run su 138 segmenti:
Dark Matter 215 → 4 parole, radio dopo il controllo 131 → 9, notturni → 0; ~7.750 s con parole solo nel v1).
Prova su 17 segmenti (12 di media + 5 con verità a orecchio), `vad_mode`:

| modalità | parole (v1 4.406) | s solo-v1 | frasi inventate | ripetute | ASR s | verità a orecchio |
|---|---|---|---|---|---|---|
| silero | 4.372 | 2.850 | 8 | 11 | 319 | 59/4/9 |
| silero_basso (0.25) | 4.495 | 2.812 | 9 | 16 | 388 | 61/4/5 |
| nessuno | 5.124 | 2.325 | 50 | 148 | 711 | – (Amara.org, «Grazie per la visione») |
| pyannote (solo zone pyannote) | 4.896 | 2.382 | 43 | 95 | 645 | 47/7/9 (cambia il testo delle conversazioni) |
| **silero_recupero** | **5.111** | **2.404** | 61 | 91 | 720 | **59/4/9 (identico a silero)** |

`silero_recupero`: prima passata Silero (conversazioni invariate), seconda passata SOLO sui pezzi dove pyannote sente
voce e la prima non ha parole (≥1 s), segmenti marcati `recovered`. Le allucinazioni note nei pezzi recuperati o
ripetute sono tolte da Lifelog (asr_rebuild r3, `hallucinated_segments`: 55 frasi su 870). Costo: trascrizione
~2× (circa 42 s invece di 19 s per segmento da 5 minuti).

## [2026-09-27] Default: lingua `auto_it_en`, allineamento `frase`

Decisione di Roberto dopo la prova su 22 segmenti e l'ascolto (Lifelog2 `docs/lifelog2-audit-completo.md`, blocco
13:30 del 2026-09-27):
- `lang_mode=auto_it_en`: lingua riconosciuta per finestra ma ristretta a italiano/inglese, con `initial_prompt`
  italiano di contesto; i termini inglesi restano inglesi, ritrovate frasi perse (chiamata delle 03:52).
- `align_mode=frase`: ctc per frase sulla finestra delle parole di Whisper (±0.5 s), ricaduta sul tempo di Whisper
  per parola se lo scarto supera 1.5 s. Scarto: p99 1.28 s, 0% oltre 2 s (testo intero: p99 68 s).
- Il backend manda anche i **pezzi** (frase tagliata ai cambi di etichetta pyannote, con impronta): l'identità la
  decide Lifelog su questi (asr_rebuild r4).
Le vecchie modalità restano disponibili per richiesta o variabile d'ambiente (`V2_LANG_MODE`, `V2_ALIGN_MODE`).

## [2026-09-28] Seconda passata in italiano sui pezzi brevi; protezioni del ctc

- **Lingua nella seconda passata** (3bd37f5): con `auto_it_en` i pezzi recuperati (mediana 1.1 s) venivano
  riconosciuti come inglese e riempiti di frasi fatte («I'll tell the police» ×9): 583 su 139 segmenti. Ora i pezzi
  recuperati <`V2_RECOVER_AUTO_MIN_S` (10 s) vanno in `V2_RECOVER_SHORT_LANG` (it); sui 22 segmenti di prova le
  recuperate in inglese passano da 140 a 19, le parole restano (6.709 contro 6.705). Costo: l'inglese vero
  brevissimo nei pezzi recuperati («Get on with the meeting», 0.8 s) diventa italiano inventato.
- **Crash nativo del ctc** (9df1c13 e seguenti): ctc-forced-aligner 0.3.0 corrompe la memoria (Windows
  0xc0000374, processo morto senza errore Python, rilevato secondi dopo in un'altra frase) su due input che non
  controlla: una parola che dopo la romanizzazione resta vuota (cifre: «8 ore al giorno») e frasi con quasi un
  fotogramma per token (ripetizioni di Whisper compresse: 37 token in 1.2 s, errore «r != u»). Ora queste frasi
  usano i tempi di Whisper: parole senza lettere → errore; meno di 2 fotogrammi (40 ms) per token → errore
  (parlato normale ≈ 3–4 per lettera). Riprodotto e verificato sul segmento 78af6078 (crash 3 volte su 3 → 0).
- **Lanci lunghi sul PC**: da SSH il processo muore con la sessione; si lancia con un'attività pianificata
  una tantum che esegue `pythonw.exe` (nessuna finestra da chiudere per sbaglio), log scritto dallo script.

## [2026-10-11] Allucinazioni: opzioni di prova `asr_opts` (default invariato)

Contesto: Lifelog2 `docs/lifelog2-audit-tratti-2026-10.md`, blocchi dell'11/10.
- Sul turno di notte 08/10 (26 segmenti, 02:36-04:41) la 2ª passata scrive 396 frasi: Lifelog ne butta 300 come
  inventate e ne tiene 96 (577 parole, il 14% del testo tenuto, in buona parte vere: voci deboli nel rumore).
- Le invenzioni sono soprattutto formule brevi: «Grazie.», «Grazie a tutti.», chiusure dei video.
- **La frase di contesto di `auto_it_en` viene ricopiata alla lettera** nei silenzi («Ho visto il meeting su YouTube,
  l'app funziona e il feedback del team è ok.», 33 frasi in 123 segmenti, tolte da Lifelog) e ne escono pezzi nei
  turni («Come was the meeting? It was good.», «la vostra team è ok»).

**Controlli fatti prima di cambiare.**
- Un controllo sul livello prima della 2ª passata non serve: le frasi tenute e quelle inventate stanno allo stesso
  livello sopra il fondo (mediana 2,3 e 2,6 dB). Qualunque soglia toglie le une e le altre in proporzione (a 8 dB:
  231/300 inventate e 71/96 tenute); Silero fa lo stesso.
- Le inventate hanno `avg_logprob` più alto (−0,23 contro −0,53) e `no_speech_prob` più alto (0,16 contro 0,05).
- `condition_on_previous_text=False` resta escluso: senza condizionamento il testo peggiorava (prova di settembre,
  sopra).
- faster-whisper sul PC 139: 1.2.1, ha `hallucination_silence_threshold` (non usato finora).

**Opzioni** (`TranscribeRequest.asr_opts`, ARIA `86489b6`): senza il campo il comportamento è quello di prima.
- `hst` / `hst1` / `hst2`: `hallucination_silence_threshold` in secondi, per tutte e due le passate o per una.
- `prompt`: frase di contesto al posto di `ITALIAN_CONTEXT_PROMPT` (`""` = nessuna).
- `temp2`: temperatura della 2ª passata (`0.0` = niente fallback a temperature alte sul rumore).

Le opzioni usate finiscono nell'uscita (`v2.asr.asr_opts`, `v2.asr.initial_prompt`).

**Client dell'orchestratore** (`aria_node_controller/backends/lifelog_whisperx.py`).
- Ricostruiva la richiesta con pochi campi e lasciava cadere in silenzio `vad_mode`, `lang_mode`, `align_mode` (e
  ora `asr_opts`).
- Aggiunti alla lista: hanno effetto **solo dal prossimo riavvio dell'orchestratore** (il modulo STT non viene
  ricaricato per task).
- Fino ad allora le prove chiamano il backend direttamente su :8091 mentre è acceso.

**Prova** (Lifelog2 `scripts/exp/aria_allucinazioni.py`, confronto `scripts/exp/allucinazioni_confronto.py`).
- WAV rifatti come Stage B.
- Varianti V1 (`hst` 2 s), V2 (frase di contesto neutra: «Allora, ci sentiamo dopo. Ho controllato il server, il backup
  è partito e il ticket è chiuso.»), V3 (V1 + V2 + `temp2` 0), più ripetizioni dalla coda con la richiesta di
  produzione (`V0rep`) per misurare la variabilità.
- Risultati sotto, appena finita.

**Risultati (2026-10-11 mattina).**
- Turno di notte 08/10, 22 segmenti comuni a tutte le varianti; parole tenute = non tolte dai filtri di Lifelog;
  «ritrovate» = stessa parola entro ±1 s.
- Frasi di controllo: 13 frasi vere e 4 invenzioni note, dalla lettura integrale.

| variante | parole della produzione ritrovate | parole tenute | frasi vere | chiusure video / eco della frase di contesto prodotte |
|---|---|---|---|---|
| produzione | — | 3145 | 13/13 | 6 / 3 |
| V0rep (produzione ripetuta dalla coda) | 96,7% | 3133 | 13/13 | 5 / 3 |
| V1 `hst` 2 s | 89,3% | 3599 | 12/13 | 1 / 0 |
| V2 frase di contesto neutra | 69,5% | 3331 | 8/13 | 4 / 3 (copia anche la frase nuova) |
| V3 `hst` + frase neutra + `temp2` 0 | 69,4% | 3573 | 8/13 | 2 / 0 |

- **Variabilità**: la stessa richiesta sullo stesso WAV cambia ~3% delle parole (fallback di temperatura).
- **La frase di contesto guida tutta la decodifica.** Cambiarla rimescola il 30% del testo e perde frasi vere
  («Ok, il collega ti ha mandato l'IP, no?» → «Ok, collegati alla IP, no?»; «Dimmi, dimmi.» sparisce).
- Con la frase neutra il segmento `ee6ea9f3` fa **morire il backend nella prima passata**, due volte su due: crash
  nativo di ctranslate2, senza errore Python. Lo stesso segmento passa con la frase attuale e in V3.
- **`temp2` 0 è dannoso.** Sul parlato debole Whisper ripiega su «Grazie.» (17 volte dove la produzione aveva «Ha
  mandato la mail perché comunque in realtà doveva dire il centro.»): il fallback di temperatura recupera parlato
  vero.
- **`hst` (V1)** riduce le chiusure dei video inventate (6 → 1). Cambia però il percorso di decodifica della prima
  passata: 7 punti di testo diverso oltre la variabilità, +14% di parole tenute (miste: «Senti, c'è la telefonata.»
  ma anche cicli come «la schedulazione, la schedulazione.»), e perde frasi vere («Non ci capisco, non riusciamo a
  lavorare…», «Soletto non ti ha detto niente»). Le invenzioni della 2ª passata restano le stesse.

**Decisione proposta:** produzione invariata.
- Le invenzioni nascono soprattutto nella 2ª passata come formule fisse e Lifelog le toglie già.
- Il residuo che conta (formule brevi, chiusure) si gestisce meglio a valle:
  - Lifelog C1 #20;
  - il testo JUNK non mostrato a Stage D, in prova.
- `hst` va rivalutato solo con un ascolto delle frasi che cambia. Le opzioni restano disponibili per prove future.
