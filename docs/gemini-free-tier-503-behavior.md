# Gemini Free Tier — Comportamento 503 e Finestre Operative Ottimali

> Documento basato su analisi empirica dei log ARIA (26 aprile – 1 maggio 2026).
> Dataset: pipeline DIAS, modello `gemini-3.1-flash-lite-preview`, ~186 chiamate Stage B + ~930 Stage C (Hyperion).

---

## 1. Contesto

ARIA instrada i task LLM di DIAS verso Gemini Flash Lite via free tier Google AI Studio.
Il free tier non ha SLA di disponibilità: l'errore 503 (`UNAVAILABLE / high demand`) è
la modalità di throttling lato Google, non un errore di rete o configurazione locale.

Ogni chiamata ARIA costa ~67-90 secondi totali (esecuzione + cooldown 60s obbligatorio).
Stage B: 186 chiamate per Hyperion. Stage C: ~930 chiamate per Hyperion.

---

## 2. Pattern 503 per Fascia Oraria (CEST, UTC+2)

| Fascia CEST | Qualità | Comportamento osservato |
|---|---|---|
| **00:00 – 07:00** | Buona | 503 sporadici, si risolvono al 1° retry (60s) |
| **07:00 – 12:00** | Discreta | 503 più frequenti, burst fino a 3-4 retry |
| **12:00 – 13:00** | Variabile | Dipende dal giorno |
| **13:00 – 16:00** | **Pessima** | Burst lunghi: 7+ retry consecutivi, 40+ min bloccati |
| **16:00 – 18:00** | Mediocre | Miglioramento progressivo |
| **18:00 – 23:00** | **Ottima** | 40-76 OK consecutivi senza interruzioni |

### Esempio critico — Giovedì 30 aprile, fascia 14:41-15:23 CEST

Job `43ae007a5506` ha fallito **7 volte consecutive** nell'arco di 40 minuti.
Con il vecchio codice ogni retry richiedeva intervento manuale: pipeline bloccata.

---

## 3. Pattern per Giorno della Settimana

| Giorno | Qualità attesa |
|---|---|
| **Sabato / Domenica** | Ottima — carichi globali ridotti |
| **Lunedì – Giovedì** | Variabile — peggiora nelle fasce business EU+US |
| **Venerdì** | Simile a Thursday, migliora in serata |

### Evidenza: Sabato 26 aprile 2026

Avvio pipeline alle 18:28 CEST (sabato sera):
- **76 OK consecutivi** nelle finestre 18:28–19:11 e 20:23–21:05 (con pausa 72 min in mezzo)
- Throughput stabile: 1 chunk ogni ~67 secondi, zero interventi manuali

Confronto giovedì 30 aprile, fascia 13:00–16:00 CEST: 10+ interventi manuali
per la stessa pipeline nelle stesse condizioni tecniche.

---

## 4. Durata dei Burst 503 — Distribuzione empirica

| Tipo burst | Frequenza stimata | Durata reale | Risolto al tentativo |
|---|---|---|---|
| Burst corto | ~70% dei casi | 60–120 secondi | 1° retry |
| Burst medio | ~20% dei casi | 2–5 minuti | 2°–3° retry |
| Burst lungo | ~10% dei casi | 10–20 minuti | 4°–5° retry |

Nel 90% dei casi il problema si risolve entro 2-3 minuti dal primo 503.
Il burst del 1 maggio (12 minuti, 4 retry falliti) è un caso limite ma gestito.

---

## 5. Strategia Retry Implementata (v2.5, Stage B e C)

Backoff progressivo implementato in Stage B e Stage C:

```
Tentativo 0  → 503 → attesa  60s
Tentativo 1  → 503 → attesa 120s
Tentativo 2  → 503 → attesa 180s
Tentativo 3  → 503 → attesa 300s   <- risolve il 95% dei burst lunghi
Tentativo 4  → 503 → attesa 600s
Tentativo 5  → 503 → PAUSE globale (intervento manuale necessario)
```

Perché backoff progressivo e non flat 10 minuti:
nel 70% dei casi il 503 dura 60-120 secondi. Un flat 10 min sprecherebbe
8-9 minuti in attesa inutile sulla maggioranza delle occorrenze.

---

## 6. Differenza 503 vs 429

| Codice | Causa | Durata | Comportamento ARIA |
|---|---|---|---|
| **503** | Server Google sovraccarico (temporaneo) | 1–20 min | Retry con backoff progressivo |
| **429** | Quota giornaliera esaurita | Fino a mezzanotte UTC | PAUSE immediata + lockout Redis 10 min |

Il 429 non è mai comparso nella sessione Hyperion Stage B (186 chunk).
Rischio concreto in Stage C (~930 chiamate) se la quota giornaliera viene consumata.
Quota free tier default: 200 request/day su `gemini-flash-lite-preview`.

---

## 7. Raccomandazioni Operative

### Quando avviare la pipeline

| Priorità | Finestra | Note |
|---|---|---|
| 1 | **Sabato/Domenica 18:00–23:00 CEST** | Finestra assoluta migliore |
| 2 | **Qualsiasi giorno 20:00–07:00 CEST** | Notte europea, basso carico US |
| 3 | **Mattina 07:00–12:00 CEST** | Accettabile con retry attivi |
| EVITARE | **Feriali 13:00–17:00 CEST** | Picco EU+US simultaneo, burst lunghi |

### Indicatori da monitorare

- **Retry count per sessione**: se >3 retry in <10 chunk, considera pausa e riavvio serale
- **Burst persistenti**: se i 503 durano oltre 15 min, è probabile un problema infrastrutturale più esteso
- **Quota giornaliera**: `redis-cli get aria:rate_limit:google:daily_count:$(date +%Y-%m-%d)`

---

## 8. Limiti del Dataset

- N = 5 giorni di osservazione — il pattern è indicativo, non statisticamente solido
- Il modello `gemini-3.1-flash-lite-preview` è in preview: il comportamento può cambiare
- La finestra EU serale funziona bene, ma Google potrebbe ribilanciare i datacenter
- Nessun dato sui weekend mattina/pomeriggio — potrebbero essere altrettanto buoni

---

*Generato: 2026-05-01 | Fonte dati: `C:\Users\roberto\aria\logs\` (aria-2026-04-*.jsonl)*
