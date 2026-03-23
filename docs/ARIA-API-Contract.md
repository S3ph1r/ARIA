# ARIA API Contract (v1.0) — Redis Communication Specification

Questo documento definisce il "Contratto" formale di comunicazione tra ARIA (Server di Inferenza) e i suoi Client (DIAS, Mobile, Dashboard, etc.). È l'unica fonte di verità per la nomenclatura delle code e la struttura dei payload.

---

## 1. Nomenclatura delle Code (Standard Universale)

### 1.1 Code di Inferenza (ARIA Global)
Queste code sono gestite dall'Orchestratore di ARIA e sono accessibili a tutti i client autorizzati.

**Pattern**: `aria:q:{env}:{provider}:{model_id}:{client_id}`

*   **`env`**: `cloud` (API esterne) | `local` (GPU locale).
*   **`provider`**: `google`, `openai`, `local` (aria-native).
*   **`model_id`**: Identificativo preciso del modello (es. `gemini-1.5-flash`, `qwen3-tts-1.7b`).
*   **`client_id`**: Identificativo del client (es. `dias`, `mob`).

### 1.2 Code di Risposta (Callback)
Dove il client attende il risultato del task.

**Pattern**: `aria:c:{client_id}:{job_id}`

---

## 2. Schema dei Payload

### 2.1 Richiesta (Task Request)
Da inviare tramite `LPUSH` sulla coda di inferenza.

```json
{
  "job_id": "string (unico)",
  "client_id": "string",
  "model_type": "string (tts|llm|vision)",
  "provider": "string",
  "model_id": "string",
  "callback_key": "aria:c:{client_id}:{job_id}",
  "timeout_seconds": "int",
  "payload": {
    "text": "string (optional)",
    "contents": "list (optional, for Gemini)",
    "config": "dict (params)"
  }
}
```

### 2.2 Risposta (Task Result)
Ricevuto tramite `BRPOP` sulla coda di callback.

```json
{
  "status": "done | error | timeout",
  "job_id": "string",
  "output": {
    "text": "string",
    "audio_url": "string (URL HTTP)",
    "duration_seconds": "float"
  },
  "error": "string (se status=error)",
  "error_code": "string",
  "processing_time": "float"
}
```

---

## 3. Identificativi Modelli (Backend Registry)

Questi sono i nomi **MANDATORI** da usare nel campo `model_id`. Riferimento originale: `aria_node_controller/config/backends_manifest.json`.

| Env | Provider | Model ID | Descrizione |
| :--- | :--- | :--- | :--- |
| `cloud` | `google` | `gemini-1.5-flash-lite` | Modello Cloud per Regia e Analisi. |
| `local` | `local` | `qwen3-tts-1.7b` | Sintesi Vocale standard (VRAM 5GB). |
| `local` | `local` | `fish-s1-mini` | Sintesi Vocale alta qualità / cloning. |
| `local` | `local` | `qwen3.5-35b-moe-q3ks` | LLM locale ad alte prestazioni (VRAM 13GB). |

---

## 4. Code Interne DIAS (Private)
Queste code **NON** fanno parte del contratto ARIA e sono gestite privatamente da DIAS.

**Pattern**: `dias:q:{stage_num}:{name}`
*   `dias:q:1:ingest`
*   `dias:q:2:semantic`
*   `dias:q:4:voice`

---
*Ultimo Aggiornamento: 2026-03-23*
