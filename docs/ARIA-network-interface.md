# ARIA Network Interface (v2.1)

Questo documento definisce gli standard di comunicazione tra ARIA (Worker Node) e le applicazioni client (es. DIAS) tramite il Bus Redis.

---

## 1. Architettura di Comunicazione
ARIA opera come un set di worker distribuiti che consumano task da code Redis e pubblicano i risultati su chiavi di callback specifiche.

## 2. Invio Task (Input)

### 2.1 Patterns delle Code
ARIA monitora le seguenti code (`LIST` Redis) in base al tipo di modello:

- **TTS Locale**: `global:queue:tts:local:{model_id}:{client_id}`
- **LLM Locale**: `global:queue:llm:local:{model_id}:{client_id}`
- **Cloud Gateway**: `global:queue:cloud:{provider}:{model_id}:{client_id}`

### 2.2 Schema del Payload
Il payload deve essere un JSON pushato tramite `LPUSH`.

```json
{
  "job_id": "unique-uuid",
  "client_id": "identificativo-client",
  "model_type": "tts | llm | vision",
  "provider": "local | google | openai",
  "model_id": "nome-modello-specifico",
  "callback_key": "global:callback:{client_id}:{job_id}",
  "timeout_seconds": 300,
  "payload": {
    "messages": [ ... ],
    "text": "Contenuto principale",
    "config": {
      "temperature": 0.2,
      "max_tokens": 4096
    }
  }
}
```

---

## 3. Ricezione Risultati (Output)
Al completamento, ARIA pubblica il risultato sulla `callback_key` fornita.

- **Operazione**: `LPUSH` + `EXPIRE` (24h)
- **Schema Risultato**:
```json
{
  "job_id": "unique-uuid",
  "status": "done | error",
  "processing_time_seconds": 1.25,
  "output": {
    "text": "Risposta generata",
    "thinking": "Ragionamento interno (se presente)",
    "audio_url": "http://{node_ip}:8082/{filename}"
  },
  "error": "Messaggio di errore in caso di failure"
}
```

---

## 4. Idempotenza e Cache
Se `job_id` è già stato processato con successo nelle ultime 24 ore, ARIA potrebbe restituire il risultato dalla cache saltando l'inferenza.

---
*Documentazione Interfaccia ARIA — Marzo 2026*
