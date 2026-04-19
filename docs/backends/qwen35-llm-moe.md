# ARIA Hybrid LLM Gateway — Technical Guide

Questa documentazione descrive l'architettura ibrida di ARIA per l'inferenza testuale, che permette di alternare dinamicamente tra **modelli locali ad alte prestazioni (GPU)** e **modelli cloud ultra-rapidi (Google Gemini)**.

---

## 🚀 1. Architettura di Routing
ARIA agisce come un proxy intelligente. Lo sviluppatore invia il task alla coda Redis e ARIA decide il percorso in base al `model_id` e al `provider`.

| Tipo | Model ID | Provider | Ambiente | Caratteristiche |
| :--- | :--- | :--- | :--- | :--- |
| **Locale** | `qwen3.5-35b-moe-q3ks` | `llama-cpp` | `envs/nh-qwen35-llm` | Massimo ragionamento, No Privacy Leak, Thinking Mode. |
| **Cloud** | `gemini-flash-lite` | `google` | `envs/aria-cloud` | Velocità istantanea, No carico GPU, Ottimo per task brevi. |

---

## 🧠 2. Inferenza Locale: Qwen 3.5 MoE
Utilizzato per task che richiedono profonda comprensione o quando la privacy è assoluta.

- **Engine**: `llama-server.exe` (OpenAI Compatible API).
- **VRAM**: ~14.5 GB (richiede RTX 5060 Ti 16GB).
- **Path Pesi**: `data/assets/models/Qwen3.5-35B-A3B-GGUF/`
- **Coda Redis**: `aria:q:llm:local:qwen3.5-35b-moe-q3ks:dias`

### Esempio Payload (Local)
```json
{
  "model_id": "qwen3.5-35b-moe-q3ks",
  "payload": {
    "messages": [
      {"role": "system", "content": "Sei un direttore artistico."},
      {"role": "user", "content": "Analizza la coerenza di questa scena."}
    ],
    "temperature": 0.2,
    "thinking": true
  }
}
```

---

## ☁️ 3. Inferenza Cloud: Google Gemini Gateway
Utilizzato come fallback quando la GPU è occupata (Semaforo RED) o per analisi veloci.

- **Worker**: `gemini_worker.py` (One-shot process).
- **Ambiente**: `envs/aria-cloud` (SDK Google GenAI).
- **Pacing**: Gestito da `GeminiRateLimiter` per evitare errori 429.
- **Coda Redis**: `aria:q:cloud:google:gemini-flash-lite:dias`

### Esempio Payload (Cloud)
```json
{
  "model_id": "gemini-flash-lite",
  "provider": "google",
  "payload": {
    "text": "Riassumi questo capitolo in 3 punti.",
    "config": {
       "temperature": 0.7,
       "response_mime_type": "application/json"
    }
  }
}
```

---

## 🛠️ 4. Note per lo Sviluppatore

### Thinking Mode (Ragionamento)
Il modello Qwen locale supporta il tag `<thought>`. ARIA estrae automaticamente questo blocco e lo restituisce isolato nel campo `thought` del risultato, lasciando la risposta pulita nel campo `response`.

### Gestione degli Asset (Warehouse-First)
Tutti i modelli GGUF e i pesi devono risiedere in `data/assets/models/`. La mappatura fisica è definita in `model_registry.json`. Se sposti i file, esegui `scripts/sync_junctions.ps1` per aggiornare i collegamenti.

---
*Ultimo aggiornamento: Aprile 2026 — ARIA Golden Standard*
