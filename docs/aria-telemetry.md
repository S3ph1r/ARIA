# ARIA — Sistema di Telemetria Locale

> Documento tecnico: `TelemetryDB` — registrazione persistente di ogni task elaborato da ARIA.

---

## Indice

1. [Perché la telemetria](#1-perché-la-telemetria)
2. [Architettura](#2-architettura)
3. [Schema SQLite](#3-schema-sqlite)
4. [Campi per backend](#4-campi-per-backend)
5. [Dove si trova il file](#5-dove-si-trova-il-file)
6. [Come interrogare i dati](#6-come-interrogare-i-dati)
7. [Garanzie di robustezza](#7-garanzie-di-robustezza)

---

## 1. Perché la telemetria

Prima dell'implementazione (Maggio 2026), ARIA non aveva un registro persistente delle sue operazioni. Era possibile analizzare i log JSONL giornalieri per capire cosa era successo, ma:

- I log non erano strutturati per query aggregate
- Non c'era modo di confrontare le performance tra modelli nel tempo
- I task cloud (Gemini) non tracciavano il consumo di token
- L'analisi dei pattern di errore (503, 429, OOM) richiedeva parsing manuale dei log

`TelemetryDB` risolve questo: ogni task completato (o fallito) da ARIA viene scritto in un database SQLite locale, interrogabile con qualsiasi strumento SQL.

---

## 2. Architettura

### Hook unico: `post_result()`

La telemetria è agganciata a un **singolo punto** nel codice: `AriaQueueManager.post_result()`.

```
Task completato
      │
      ▼
NodeOrchestrator._process_*_task()   ← 5 metodi (fish, qwen3, llm, acestep, audiocraft)
      │
      ▼
CloudManager._run_cloud_loop()       ← 1 metodo (gemini + future provider)
      │
      ▼
AriaQueueManager.post_result()       ◄── HOOK TELEMETRIA (unico punto)
      │         │
      │         └── self.telemetry.log(task, result)  se iniettato
      │
      ▼
Redis LPUSH → callback_key           ← risultato consegnato al client
```

### Injection pattern

`AriaQueueManager` espone `self.telemetry = None`. `NodeOrchestrator` lo inietta dopo la creazione:

```python
# In NodeOrchestrator.__init__:
self.qm = AriaQueueManager(redis_client)
self.telemetry = TelemetryDB(ARIA_ROOT / "logs" / "aria-telemetry.db")
self.qm.telemetry = self.telemetry
```

Questo significa che se per qualsiasi motivo la telemetria non viene iniettata (es. futuro scenario di test), ARIA continua a funzionare normalmente senza scrivere nulla.

### Thread safety

`TelemetryDB` usa un `threading.Lock()` interno. I task locali (orchestratore) e cloud (CloudManager) possono chiamare `log()` da thread diversi senza race condition.

### WAL mode

Il database SQLite è aperto in **WAL (Write-Ahead Logging)** mode. Questo permette:
- Letture concorrenti senza bloccare le scritture
- Lettura del `.db` con DB Browser o script Python mentre ARIA è in esecuzione
- Resistenza ai crash: le scritture incomplete non corrompono il database

---

## 3. Schema SQLite

File: `logs/aria-telemetry.db`  
Tabella: `task_log`

| Colonna | Tipo | Fonte | Note |
|---------|------|-------|------|
| `id` | INTEGER PK | auto | rowid SQLite |
| `ts` | TEXT | `result.completed_at` | ISO 8601 UTC |
| `job_id` | TEXT | `task.job_id` | UUID del task |
| `client_id` | TEXT | `task.client_id` | es. `dias-brain` |
| `provider` | TEXT | `task.provider` | `local` o `google` |
| `model_id` | TEXT | `task.model_id` | es. `fish-s1-mini` |
| `model_type` | TEXT | `task.model_type` | `tts`, `llm`, `mus`, `cloud` |
| `queued_at` | TEXT | `task.queued_at` | ISO 8601 UTC |
| `queue_wait_s` | REAL | calcolato | `completed_at - processing_s - queued_at` |
| `processing_s` | REAL | `result.processing_time_seconds` | tempo inferenza |
| `status` | TEXT | `result.status` | `done`, `error`, `timeout` |
| `error_code` | TEXT | `result.error_code` | `OOM`, `429`, `503`, `INFERENCE_FAILED`, … |
| `input_tokens` | INTEGER | `result.usage["input_tokens"]` | solo provider cloud |
| `output_tokens` | INTEGER | `result.usage["output_tokens"]` | solo provider cloud |
| `audio_duration_s` | REAL | `result.output["duration_seconds"]` | task TTS e MUS |
| `rtf` | REAL | `result.output["metrics"]["rtf"]` | Real-Time Factor (solo Qwen3) |
| `vram_peak_gb` | REAL | `result.output["metrics"]["vram_peak_gb"]` | solo Qwen3 |

**Indici:**
- `idx_ts` su `ts` — per query temporali
- `idx_model_status` su `(model_id, status)` — per analisi per modello

### Calcolo `queue_wait_s`

```
queue_wait_s = completed_at_epoch - processing_time_seconds - queued_at_epoch
```

Rappresenta quanto tempo il task ha trascorso in coda Redis prima che l'orchestratore lo prelevasse. È un indicatore di carico del sistema.

---

## 4. Campi per backend

| Backend | `audio_duration_s` | `rtf` | `vram_peak_gb` | `input_tokens` | `output_tokens` |
|---------|--------------------|-------|----------------|----------------|-----------------|
| fish-s1-mini | ✓ | — | — | — | — |
| qwen3-tts-1.7b | ✓ | ✓ | ✓ | — | — |
| qwen3.5-35b-moe-q3ks | — | — | — | — | — |
| acestep-1.5-xl-sft | ✓ | — | — | — | — |
| audiocraft-medium | ✓ | — | — | — | — |
| gemini (cloud) | — | — | — | ✓ | ✓ |

I campi non applicabili vengono scritti come `NULL` in SQLite.

---

## 5. Dove si trova il file

```
C:\Users\roberto\aria\
└── logs\
    ├── aria-YYYY-MM-DD.log     ← log JSONL strutturati (esistenti)
    └── aria-telemetry.db       ← database SQLite telemetria (NUOVO)
```

**Tool consigliati per ispezione:**
- [DB Browser for SQLite](https://sqlitebrowser.org/) — GUI, zero setup
- Python `sqlite3` stdlib — per script di analisi
- DBeaver — se già installato

---

## 6. Come interrogare i dati

### Task per modello nell'ultima settimana

```sql
SELECT model_id, COUNT(*) as tasks, AVG(processing_s) as avg_s, 
       SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as errors
FROM task_log
WHERE ts > datetime('now', '-7 days')
GROUP BY model_id
ORDER BY tasks DESC;
```

### Errori Gemini con dettaglio codice

```sql
SELECT date(ts) as giorno, error_code, COUNT(*) as count
FROM task_log
WHERE provider='google' AND status='error'
GROUP BY giorno, error_code
ORDER BY giorno DESC;
```

### Token consumati da Gemini (per stimare costi)

```sql
SELECT date(ts) as giorno, 
       SUM(input_tokens) as tot_input, 
       SUM(output_tokens) as tot_output,
       COUNT(*) as tasks
FROM task_log
WHERE provider='google' AND status='done'
GROUP BY giorno
ORDER BY giorno DESC;
```

### Performance Qwen3 TTS (RTF per durata audio)

```sql
SELECT date(ts) as giorno,
       AVG(rtf) as avg_rtf,
       AVG(audio_duration_s) as avg_dur_s,
       AVG(vram_peak_gb) as avg_vram
FROM task_log
WHERE model_id LIKE 'qwen3-tts%' AND status='done'
GROUP BY giorno;
```

### Tempi di attesa in coda (carico sistema)

```sql
SELECT model_id,
       AVG(queue_wait_s) as avg_wait_s,
       MAX(queue_wait_s) as max_wait_s,
       COUNT(*) as tasks
FROM task_log
WHERE status='done'
GROUP BY model_id
ORDER BY avg_wait_s DESC;
```

### Task falliti nelle ultime 24h

```sql
SELECT ts, job_id, client_id, model_id, error_code, 
       substr(status, 1, 50) as status
FROM task_log
WHERE status != 'done' AND ts > datetime('now', '-1 day')
ORDER BY ts DESC;
```

---

## 7. Garanzie di robustezza

`TelemetryDB.log()` è progettato per **non crashare mai ARIA**:

```python
def log(self, task, result) -> None:
    try:
        self._write(task, result)
    except Exception as e:
        logger.warning(f"Telemetry write skipped: {e}")
```

Se il disco è pieno, il file è corrotto, o il lock fallisce, ARIA continua a funzionare normalmente. La telemetria è un observer passivo — non è nel percorso critico di consegna del risultato al client.

Il risultato viene consegnato a Redis **prima** che venga chiamata `telemetry.log()`:

```python
# In post_result():
pipeline.execute()           # ← risultato consegnato al client
logger.info(...)
if self.telemetry:
    self.telemetry.log(...)  # ← telemetria dopo, mai bloccante
```

---

*ARIA Telemetry — Maggio 2026*
