# Qwen3-14B — backend ARIA: spec deployata + handoff Lifelog2

**Data**: 2026-09-09 · **Stato**: ARIA lato **deployato e verificato** su PC139. Resta la parte Lifelog2 (`core/llm.py`).
**Contesto**: `qwen3-llm-wrapper-investigation-2026-09-08.md` (indagine) · `qwen3-llm-wrapper-redesign-2026-09-08.md` (design + risultati §8).
**Consumatore unico**: Lifelog2 (`AriaLLMClient`). DIAS non usa questo backend.

---

## 1. Cosa è deployato

| Elemento | Stato |
|---|---|
| Binario | `llama-server` **b10819** / `cuda-13.3` in `C:\Users\roberto\aria\tools\llama\` (b9119 in `tools\llama.b9119\`, rollback = `ren` inverso) |
| Avvio server | dal manifest `server_args` (dict): `--ctx-size 32768 --cache-type-k q8_0 --cache-type-v q8_0 --parallel 1 --flash-attn on --jinja --reasoning-format deepseek --no-context-shift` |
| Wrapper | `aria_node_controller/backends/lifelog_llm.py` — riscritto (profili, passthrough, contratto) |
| Contratto | `aria_node_controller/config/backends_manifest.json` → entry `qwen3-14b-q4km` → blocco `llm_contract` |
| Orchestratore | `_build_cmd` ricarica il manifest e `_process_lifelog_llm_task` ricarica il modulo backend **ad ogni task** → un `git pull` è effettivo senza riavvio |

**Flusso** (invariato nella struttura):
```
Lifelog2 CT203 · core/llm.py::AriaLLMClient.generate_json
  → RPUSH task su  aria:q:llm:local:qwen3-14b-q4km:lifelog   (Redis CT120)
  → ARIA orchestrator PC139 · _process_lifelog_llm_task
      → ensure_running (JIT: avvia llama-server se serve, ~7s cold, spegne dopo 45min idle)
      → LifelogLLMBackend.run(payload)          ← QUI il redesign
      → POST http://192.168.1.139:8090/v1/chat/completions
  → RPUSH risultato su  aria:result:llm:{job_id}
  → Lifelog2 BRPOP
```

---

## 2. Protocollo payload — come si chiama

Il `payload` del task Redis (dentro `task["payload"]`). Struttura attuale di Lifelog2:
```jsonc
{
  "messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."}],
  "max_tokens": 1536,
  "temperature": 0.3,     // ⚠️ Lifelog2 lo forza oggi — va TOLTO (vedi §8)
  "thinking": false
}
```

### Selezione modalità (profilo)
| Campo nel payload | Effetto |
|---|---|
| `"profile": "thinking"` / `"non_thinking"` | sceglie il profilo (precedenza massima) |
| `"thinking": true` | → profilo `thinking` |
| `"thinking": false` | → profilo `non_thinking` |
| niente | → `non_thinking` (default del contratto) |

### Override per-campo
Qualunque parametro in §3 messo **esplicitamente** nel payload fa override del valore del
profilo, **campo per campo** (il resto del profilo resta). `chat_template_kwargs` è in
*deep-merge* col profilo.

Chiavi consumate dal wrapper e **non** inoltrate: `messages`, `prompt`, `text`, `job_id`,
`timeout_seconds`, `thinking`, `profile`.

Parametro non riconosciuto dal contratto → **inoltrato lo stesso + WARN** (`unknown_param_policy:
forward_warn`). Mai più scartato in silenzio.

---

## 3. Parametri configurabili (contratto `request_params`)

`req` = valido nel body richiesta. *default nt* / *default th* = valore nel profilo non_thinking / thinking.
Fuori range → **clamp + WARN**. Tipo sbagliato → **errore**.

| Parametro | Tipo | default nt | default th | Significato · note |
|---|---|---|---|---|
| `max_tokens` | int 1..32768 | 4096 | 4096 | Tetto token generati. In thinking il pensiero attinge dallo stesso tetto → per questo il cap sotto |
| `temperature` | 0.0..2.0 | 0.7 | 0.6 | Casualità. **0 vietato in thinking** (il wrapper rifiuta — Qwen: greedy → ripetizioni infinite) |
| `top_p` | 0.0..1.0 | 0.8 | 0.95 | Nucleus sampling |
| `top_k` | int ≥0 | 20 | 20 | Top-K. **1 vietato in thinking** (greedy) |
| `min_p` | 0.0..1.0 | 0.0 | 0.0 | Prob. minima relativa al token più probabile |
| `reasoning_budget_tokens` | int ≥-1 | **0** | **2048** | ✅ verificato: `0` = spegne il pensiero · `N>0` = cap sui token di pensiero (a 150 → ~150 tok poi risposta finale) · `-1` = illimitato. Alias accettato: `thinking_budget_tokens` |
| `chat_template_kwargs` | object | `{enable_thinking:false}` | `{enable_thinking:true}` | ✅ `enable_thinking:false` spegne il pensiero via template. Deep-merge col profilo |
| `presence_penalty` | -2.0..2.0 | — | — | Penalità presenza. Qwen: alto → *language mixing* (usare solo in non-thinking) |
| `frequency_penalty` | -2.0..2.0 | — | — | Penalità proporzionale alla frequenza |
| `repeat_penalty` | ≥0.0 | — | — | Penalità su sequenze ripetute (default server b10819 = 1.0) |
| `repeat_last_n` | int ≥-1 | — | — | Finestra token per `repeat_penalty` |
| `seed` | int | — | — | Seed RNG (riproducibilità) |
| `stop` | array[str] | — | — | Stringhe di arresto |
| `samplers` | array[str] | — | — | Ordine di applicazione dei sampler |
| `response_format` | object | — | — | `{"type":"json_object"}` → JSON valido **ma dentro fence ```json** (Lifelog2 le strippa già). Non affidarsi al "raw" |
| `json_schema` | object | — | — | Schema per grammar-based sampling (più stringente, non testato) |
| `grammar` | string | — | — | Grammatica BNF-like |
| `ignore_eos` | bool | — | — | Continua oltre il token di fine |
| `logit_bias` | object | — | — | Modifica probabilità di token specifici |

**Fuori dal contratto** (non funzionano su Qwen3-14B — verificato §8):
- `reasoning_effort` — il template non lo supporta (`chat_template_caps.supports_reasoning_effort=false`)
- `enable_thinking` top-level — ignorato (va dentro `chat_template_kwargs`)
- `reasoning_budget` (senza `_tokens`) — ignorato nel body

---

## 4. Profili — set completi

```jsonc
"non_thinking": {                       "thinking": {
  "temperature": 0.7,                     "temperature": 0.6,
  "top_p": 0.8,                           "top_p": 0.95,
  "top_k": 20,                            "top_k": 20,
  "min_p": 0.0,                           "min_p": 0.0,
  "chat_template_kwargs":                 "chat_template_kwargs":
      {"enable_thinking": false},             {"enable_thinking": true},
  "reasoning_budget_tokens": 0            "reasoning_budget_tokens": 2048,
}                                         "reasoning_reserve_tokens": 2048  // meta, non va al server
                                        }
```
Sampling = valori ufficiali Qwen3-14B (model-card + Unsloth). `non_thinking` spegne il pensiero
con **due leve indipendenti** (template + sampler). `thinking` cappa il pensiero a 2048 token di
default (overridabile) così non si mangia tutto `max_tokens`.

---

## 5. Risposta — cosa torna

`run()` ritorna, e l'orchestratore mette in `output`:
```jsonc
{
  "text":          "...",        // choices[0].message.content (risposta visibile)
  "thinking":      "...",        // choices[0].message.reasoning_content (pensiero isolato)
  "usage":         {...},        // completion_tokens, prompt_tokens, ...
  "finish_reason": "stop"|"length"   // NUOVO — "length" = troncato al tetto
}
```
`--reasoning-format deepseek` isola il pensiero in `reasoning_content`. Resta il fallback regex
`<think>...</think>` nel wrapper per sicurezza.

---

## 6. Limiti / cose da sapere

- **VRAM stretta**: llama-server ~10.8 GB (ctx 32768 + KV q8 + parallel 1). Con il desktop di
  Roberto attivo restano ~2.4 GB. Il semaforo GPU scarica il modello quando Roberto gioca.
- **`response_format: json_object`** → JSON valido ma con fence markdown. Lifelog2 le toglie già.
- **`psutil` assente** in `envs\lifelog-llm` → self-reporting PID del launcher saltato; la
  scoperta PID via titolo finestra dell'orchestratore è il fallback (fix 2026-09-05).
- **Perf** (b10819/cuda-13.3): generazione ~42 tok/s, prompt processing ~1600 tok/s su 8K token.
- **KV q8 qualità**: non ancora confrontata con f16 su un dump Stage D reale — vedi §7c.

---

## 7. Test da eseguire

### 7a. Diretto a llama-server (quando il backend è su)

Da PC139 localhost (CT190→:8090 bloccato dal firewall). Il backend è su solo se c'è stato un job
di recente; altrimenti si avvia a mano — **solo con ok di Roberto** — con i flag del §1.

Script pronto: `scratchpad/s8b.py` (nella sessione originale) o equivalente:
```python
# via:  ssh pc139 "C:\Users\roberto\aria\envs\lifelog-llm\python.exe - http://127.0.0.1:8090" < s8b.py
import json, urllib.request
B = "http://127.0.0.1:8090"
def call(p):
    r = urllib.request.Request(B+"/v1/chat/completions", data=json.dumps(p).encode(),
                               headers={"Content-Type":"application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=90)); c = d["choices"][0]
    return c["finish_reason"], (c["message"].get("reasoning_content") or ""), (c["message"].get("content") or "")

Q = [{"role":"user","content":"Quanto fa 17*23? Ragiona passo passo."}]
print(call({"messages":Q, "max_tokens":250, "chat_template_kwargs":{"enable_thinking":False}}))  # reasoning vuoto
print(call({"messages":Q, "max_tokens":250, "reasoning_budget_tokens":0}))                        # reasoning vuoto
print(call({"messages":Q, "max_tokens":700, "reasoning_budget_tokens":150,                        # pensiero ~150 tok
            "temperature":0.6, "top_p":0.95, "top_k":20}))
```
Atteso: le prime due con `reasoning` vuoto e `content` pieno; la terza con `reasoning` ~150 tok
poi `content` con la risposta finale, `finish_reason="stop"`.

### 7b. Via Redis (simula un task Lifelog2, path completo wrapper + orchestratore)

Da CT190 — **richiede che ARIA sia su e la coda scorra**. Non spinge job reali di produzione:
usa `client_id` distinto e un `callback_key` proprio.
```python
import json, uuid, redis
r = redis.from_url("redis://192.168.1.120:6379")
job = str(uuid.uuid4())
task = {
    "job_id": job, "client_id": "wrapper-test", "model_type": "llm",
    "model_id": "qwen3-14b-q4km", "callback_key": f"aria:result:llm:{job}",
    "payload": {
        "messages": [{"role":"user","content":"Estrai JSON {citta, paese}: sono stato a Oslo."}],
        "max_tokens": 200, "thinking": False,
        "response_format": {"type":"json_object"}
    },
}
r.rpush("aria:q:llm:local:qwen3-14b-q4km:lifelog", json.dumps(task))
_, res = r.brpop(f"aria:result:llm:{job}", timeout=600)
out = json.loads(res)
print("status:", out.get("status"))
print("output:", json.dumps(out.get("output"), indent=2, ensure_ascii=False))
# atteso: status=done, output.text JSON valido, output.thinking vuoto, output.finish_reason presente
```
Verifica anche: `thinking: true` + `reasoning_budget_tokens: 300` → `output.thinking` popolato
e limitato; parametro fuori contratto (es. `"pippo": 1`) → arriva lo stesso (WARN nei log ARIA).

### 7c. Spot-check qualità KV q8 (a mente fresca, con un dump reale)

Prendere un `llm_call_dumps/stage_d_*.json` reale di Lifelog2 (request completa: system + tabella
turni). Rilanciarlo diretto a llama-server 2 volte:
- così com'è (ctx 32768, KV q8 — la config live)
- con un llama-server temporaneo a `--ctx-size 16384 --cache-type-k f16 --cache-type-v f16` su
  porta diversa
Confrontare a campione i thread/confini prodotti. Se q8 degrada in modo visibile → ladder di
fallback (`cache-type-v f16` con K q8, poi ctx 16384) documentata nel redesign §5.

---

## 8. Modifiche lato Lifelog2 — `src/backend/lifelog2/core/llm.py`

Classe `AriaLLMClient`. Il wrapper ARIA ora fa tutto il lavoro di profilo/sampling — il client
deve **passare meno**, non di più.

### 8.1 `_generate_json_once` — il `task["payload"]`

Attuale:
```python
"payload": {
    "messages": messages,
    "max_tokens": max_tokens,
    "temperature": temperature,   # ← RIMUOVERE (lo forza a 0.3, sovrascrive il profilo)
    "thinking": thinking,
}
```
Nuovo:
```python
"payload": {
    "messages": messages,
    "max_tokens": max_tokens,
    "thinking": thinking,
    # temperature/top_p/top_k NON passati: decide il profilo del contratto ARIA.
    # Se un caller specifico li vuole diversi, li passa esplicitamente come override.
}
```
`generate_json` ha `temperature: float = 0.3` come parametro di firma: mantenerlo per
compatibilità ma **non inserirlo nel payload** se non esplicitamente richiesto da un caller
(valutare: aggiungere `temperature: Optional[float] = None` e passarlo solo se non-None).

### 8.2 Costanti

- `CONTEXT_WINDOW_TOKENS = 16384` → **`32768`**. Aggiungere commento: *"deve combaciare con
  `llm_contract.model.context_window_tokens` nel manifest ARIA — vedi
  `sviluppi/ARIA/docs/qwen3-14b-backend-spec-2026-09-09.md`"*.
- `_THINKING_RESERVE_TOKENS = 1024` → **`2048`** (= `reasoning_budget_tokens` del profilo
  `thinking`). Commento con lo stesso riferimento.

### 8.3 `finish_reason` — nuovo segnale

In `_generate_json_once`, dopo `output = result.get("output") or {}`:
```python
finish_reason = output.get("finish_reason")
```
Nella logica `near_ceiling` (retry su troncamento): oggi è un'euristica (`estimated_tokens_used
>= 0.9 * max_tokens`). Ora `finish_reason == "length"` è la **verità diretta**:
```python
near_ceiling = (finish_reason == "length") or (estimated_tokens_used >= 0.9 * max_tokens)
```
Loggarlo nella telemetria (`telemetry.record_stage(... extra={... "finish_reason": finish_reason})`).

### 8.4 Quali chiamate passano a `thinking: true`

Oggi **nessun** caller usa `thinking=True` (era rotto lato ARIA). Ora funziona. Da decidere,
caller per caller (Stage D thread builder, Stage E enrichment, Identity Detective, Day Digest,
Z6 consolidation):
- `thinking=True` migliora il ragionamento su compiti con **entità multiple / confini ambigui**
  (il limite del 14B trovato negli incidenti Stage D) — ma consuma `reasoning_budget_tokens`
  (2048 default) dal tetto `max_tokens`: **alzare `max_tokens` di conseguenza** dove si attiva.
- Partire conservativi: lasciare tutto `thinking=False`, poi provare `thinking=True` **solo su
  Stage D** (thread builder) con `max_tokens` aumentato e confrontare la qualità dei confini su
  un batch di verifica.
- La guardia overflow di `generate_json` (2026-09-02) già riserva `_THINKING_RESERVE_TOKENS`
  quando `thinking=True` — con la costante aggiornata a 2048 il conto torna.

### 8.5 Test lato Lifelog2

- Unit: `AriaLLMClient` costruisce il payload senza `temperature`, con `thinking` corretto.
- Integrazione (via Redis, ARIA su): una `generate_json` reale con `thinking=False` → risposta
  JSON, `finish_reason` letto; una con `thinking=True` → `output.thinking` popolato.
- Un giro di Stage D su un batch piccolo con la nuova config, confronto confini/troncamenti
  contro il baseline (i `llm_call_dumps` di prima del cambio).

---

## 9. Rollback (se serve)

- **Binario**: `ssh pc139` → `cd C:\Users\roberto\aria\tools` → `ren llama llama.b10819 && ren llama.b9119 llama`.
- **Codice**: `git revert baa917e 4bc78df` su CT190 + push + `git pull` su PC139. Gli hot-reload
  fanno sì che il revert sia effettivo al task successivo, senza riavvio.
- Il manifest col vecchio `args` e il vecchio `run()` tornano attivi automaticamente.
