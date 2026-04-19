# Diagnostica Pipeline ACE-Step: Analisi Audio "Scalzi"

## Punto 1 — Il JSON di Partenza (Task Redis)

Il JSON iniettato in Redis (`inject_b2_test.py`) contiene:

| Campo | Valore | Note |
|-------|--------|------|
| `duration` | 470.4s | Totale richiesto |
| `guidance_scale` | 4.5 | **⚠️ Basso** (standard 7.0) |
| `inference_steps` | 60 | Corretto |
| `seed` | 42 | Fisso |
| `prompt` | "70s retro-futuristic...dissonant brass..." | Semanticamente aggressivo |
| `lyrics` | 4 anchor temporali: 00:00 / 01:30 / 05:00 / 06:40 | Costruiti per 470s totali |

**Problema #1**: Le lyrics sono scritte per l'intera durata di 470 secondi (l'evento a 06:40 = minuto 6, secondo 40). Vengono però passate identiche a ogni chunk da 120s, creando un'incoerenza temporale grave.

---

## Punto 2 — Conversione TOML e Segmentazione

### Come avviene la segmentazione

Il Wrapper divide 470.4s in chunk da 120s:
- pt0: 0s → 120s (120s)
- pt1: 120s → 240s (120s)
- pt2: 240s → 360s (120s)
- pt3: 360s → 470.4s (110.4s)

### Come vengono distribuite le informazioni

> [!CAUTION]
> **Errore Critico: Le lyrics sono identiche in tutti i 4 TOML.**
>
> Ogni chunk riceve lo stesso testo di 4 marker: `[00:00]`, `[01:30]`, `[05:00]`, `[06:40]`.
> Questo è sbagliato. Il chunk `pt1` comincia a 120s nella realtà, ma l'LM crede di stare generando da zero. Gli viene detto che a `[05:00]` dovrà eseguire il "chorus"... ma il chunk dura solo 120s. L'LM non capisce e collassa.

### Cosa è uguale tra i chunk (corretto):
- `prompt`, `caption`, `guidance_scale`, `inference_steps`, `save_dir` — **contestuale globale, va bene che sia uguale**

### Cosa cambia tra i chunk (parzialmente corretto):
- `duration`: corretto (120.0 / 120.0 / 120.0 / 110.4)
- `seed`: incrementato di 1 per chunk (42, 43, 44, 45) — design corretto per evitare duplicati
- `reference_audio`: presente da pt1 in poi con il file `relay_tail_N.wav` — meccanismo corretto
- `audio_cover_strength`: 1.0 per pt0, 0.4 per pt1-pt3 — corretto

### Cosa NON cambia e dovrebbe cambiare (errore grave):
- `lyrics`: **IDENTICHE per tutti i 4 chunk** — vedi sopra

---

## Punto 3 — Cosa passa cli.py per ogni segmento

### Per `pt0` (il più "pulito"):
```toml
duration = 120.0
lyrics = "[00:00...][01:30...][05:00...][06:40...]"   # TUTTI GLI ANCHOR
audio_cover_strength = 1.0  # genera da zero, corretto
```
L'LM ACE-Step deve capire il "timing globale" dell'opera in 120 secondi. Nel json prodotto vediamo:
- Ha riscalato autonomamente i tempi: `[00:38]`, `[00:60]`, `[01:30]`, `[01:55]`
- Ha ignorato `[05:00]` e `[06:40]` (troppo lontani per 120s)

Ma questo "riscalamento autonomo" è non deterministico e crea incoerenza tra i chunk.

### Per `pt1` (il problema si aggrava):
```toml
duration = 120.0
lyrics = "[00:00...][01:30...][05:00...][06:40...]"   # IDENTICHE a pt0
reference_audio = "relay_tail_0.wav"                   # corretto
audio_cover_strength = 0.4
```
L'LM riceve **esattamente le stesse istruzioni temporali** ma deve produrre musica diversa. Il `reference_audio` gli dice "continua da qui timbricamente" ma le lyrics gli dicono "ricomincia dall'inizio narrativamente". Conflitto diretto.

### Per `pt2` e `pt3`: stesso problema, amplificato.

---

## Punto 4 — Il flusso interno di cli.py (senza toccare nulla)

ACE-Step usa una pipeline a 4 stadi. Basandosi su `cli.py`, `inference.py` e `llm_inference.py`:

```
[Input TOML]
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  FASE 1: LM (Qwen3 1.7B) — "Il Compositore"        │
│  Input:  prompt + lyrics + duration + thinking       │
│  Output: score.json (bpm, key, timesig, audio_codes)│
│  Lavoro: scrive la "partitura" testuale dell'audio   │
└──────────────────────┬──────────────────────────────┘
                       │ audio_codes (sequenza token)
                       ▼
┌─────────────────────────────────────────────────────┐
│  FASE 2: T5 Encoder — "Il Traduttore del Testo"     │
│  Input:  prompt + caption (testo libero)             │
│  Output: embedding vettoriale del testo (cross-attn) │
│  Lavoro: converte il prompt in vettori semantici     │
└──────────────────────┬──────────────────────────────┘
                       │ text_embeddings
                       ▼
┌─────────────────────────────────────────────────────┐
│  FASE 3: DiT XL — "Il Direttore d'Orchestra"        │
│  Input:  audio_codes (da LM) + text_embeddings (da  │
│          T5) + reference_audio latents (se presente) │
│  Output: latenti audio raffinati (spazio latente)    │
│  Lavoro: denoising diffusion in 60 steps (CFG=4.5)  │
└──────────────────────┬──────────────────────────────┘
                       │ refined_latents
                       ▼
┌─────────────────────────────────────────────────────┐
│  FASE 4: VAE Decoder — "Il Musicista"               │
│  Input:  latenti raffinati                           │
│  Output: waveform audio (PCM float32)                │
│  Lavoro: decodifica i latenti in suono reale         │
└──────────────────────┬──────────────────────────────┘
                       │ .wav file
                       ▼
               [Output WAV + score.json]
```

### Dove si localizza il problema osservato

Il **collasso ripetitivo** (`<|audio_code_35847|>` per centinaia di token) avviene in **FASE 1 (LM)**.

Lo score.json mostra chiaramente:
1. Struttura normale per i primi 200-300 token
2. Poi "loop lock": lo stesso pattern `[1868|12019|36219|13259]` si ripete identico decine di volte
3. Poi caduta totale: `<|audio_code_35847|>` per 40+ token = **padding/fine forzata**

Questo vuol dire che il T5, il DiT e il VAE ricevono in ingresso una partitura già compromessa. Il problema nasce **a monte**, nell'LM.

---

## Punto 5 — Come salvare gli output intermedi per diagnosi

Attualmente salviamo già:
- ✅ `score.json` (output Fase 1 LM) — **salvato da cli.py automatically**
- ✅ TOML di ogni chunk (input completo della pipeline)
- ❌ T5 embeddings — **non esposti** da cli.py (interni a inference.py)
- ❌ DiT latents — **non esposti** (salvarne uno richiederebbe una modifica minima)
- ❌ VAE pre-decode — **non esposto**

### Cosa possiamo fare SENZA toccare cli.py

Il `score.json` è la nostra "TAC" più preziosa. Contiene:
- **bpm, key, timesig**: dice chi ha "deciso" la struttura musicale l'LM
- **audio_codes**: la partitura token per token — possiamo analizzarla per rilevare loop
- **lyrics riscritte dall'LM**: vediamo come ha "interpretato" le nostre istruzioni

**Proposta**: Creare uno script di analisi post-generazione `diagnose_score.py` che:
1. Legge tutti i `score.json` della sessione
2. Calcola il tasso di ripetizione dei token (se token X appare >10 volte consecutive → loop detected)
3. Stampa un report: quale chunk è collassato, a che token, percentuale di unicità
4. Non tocca nulla, solo legge e valuta

---

## Diagnosi Finale

| Problema | Localizzazione | Gravità |
|----------|---------------|---------|
| Lyrics identiche in tutti i chunk | Wrapper (segmentazione) | 🔴 Critica |
| Anchor temporali non riscalati per chunk | Wrapper (time-mapping) | 🔴 Critica |
| Loop dell'LM nella Fase 1 | Probabilmente causato dai 2 problemi sopra | 🟠 Conseguenza |
| guidance_scale = 4.5 | JSON di ingresso | 🟡 Contribuisce |
| `instrumental = false` con lyrics non vocalizzabili | TOML generato | 🟡 Contribuisce |

**La causa radice**: l'LM riceve lyrics "impossibili" (anchor a 06:40 in un chunk da 120s) e si "confonde", generando prima un pattern loop poi collassando nel padding token `35847`.

I componenti T5, DiT e VAE funzionano probabilmente correttamente. Il problema è tutto nell'input che diamo all'LM.
