# ARIA Network Interface Specification (v2.1)

## 1. Overview
This document defines the official "Service Contract" between an **ARIA Worker Node** and any **Client Application** (e.g., DIAS). ARIA operates as a decoupled inference provider over a **Redis Mesh**.

### Connectivity Principles
- **Asynchronous**: Clients submit tasks and continue; ARIA processes when resources are available.
- **Location-Agnostic**: All communication happens via Redis; IPs are only for asset retrieval.
- **Stateless Workers**: ARIA does not store project-level state; it only knows tasks and local cache.

---

## 2. Node Discovery & Heartbeat
Every ARIA node signals its presence on Redis to allow for central monitoring.

- **Key**: `aria:global:node:{node_ip}:status`
- **Type**: String (JSON)
- **TTL**: 60 seconds (updated every 20s)

### Heartbeat Schema
```json
{
  "node_ip": "192.168.1.139",
  "status": "online | paused",
  "last_seen": "ISO-8601-Timestamp",
  "active_backends": ["qwen3-tts-1.7b", "fish-s1-mini"],
  "available_voices": ["angelo", "luca", "narratore"]
}
```

---

## 3. Task Submission (Input)
Tasks are submitted to model-specific FIFO queues.

- **Queue Pattern**: `gpu:queue:{model_type}:{model_id}`
- **Operation**: `LPUSH` (Client) / `BRPOP` (ARIA)

### Task Payload Schema (v2.1)
```json
{
  "job_id": "unique-uuid",
  "client_id": "app-name",
  "model_type": "tts | llm | music",
  "model_id": "specific-model-name",  // Identificativo per il caricamento JIT (es. qwen3-tts-custom)

  "payload": {
      "text": "Content to process",
      "voice_id": "optional-intent",
      "voice_override": "luca",
      "...": "model-specific params"
  },
  "callback_key": "redis-key-for-result",
  "timeout_seconds": 300
}
```

---

## 4. Idempotency & Persistence
ARIA implements a **Double-Check Idempotency** rule to prevent redundant GPU usage:

1. **Local Disk Check**: Before starting inference, ARIA checks if `{job_id}.wav` (or specific extension) already exists in its local `outputs/` directory.
2. **Registry Check (External)**: Independent of ARIA, the Client (DIAS) maintains a Master Registry.
3. **If Hit**: ARIA skips the GPU task and immediately returns the URL of the existing file to the `callback_key`.

---

## 5. Result Callback (Output)
When a task is completed (or skipped via idempotency), ARIA posts a results.

- **Key**: `{callback_key}` (provided in task)
- **Operation**: `LPUSH` + `EXPIRE` (24h)

### Result Schema
```json
{
  "job_id": "unique-uuid",
  "status": "done | error",
  "processing_time_seconds": 12.5,
  "output": {
    "audio_url": "http://{node_ip}:8082/{filename}",
    "duration_seconds": 45.2,
    "cached": true | false
  },
  "error": "Error message if failed"
}
```

---

## 6. Asset Serving Standards
All generated files are served via a local HTTP server on the Worker Node.

- **Asset Port**: `8082` (Standard)
- **Root**: `C:\Users\{User}\aria\data\outputs\`
- **URL Format**: `http://{node_ip}:8082/{job_id}.wav` (Fish might use `_scene-001` suffix).

---
*ARIA Interface v2.1 — March 2026*
