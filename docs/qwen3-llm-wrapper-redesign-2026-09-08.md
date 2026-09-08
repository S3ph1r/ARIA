# Redesign wrapper LLM locale (Qwen3-14B ↔ llama-server) — DESIGN

**Data**: 2026-09-08 · **rev**: 2026-09-09 (§8 verifica empirica ESEGUITA — risultati in §8)
**Segue**: `qwen3-llm-wrapper-investigation-2026-09-08.md` (indagine — stato reale, bug, matematica KV)
**Stato**: **implementazione lato CT190 FATTA** (§9 passi 3-5, 7 — 2026-09-09). 16 test offline
passati. **ARIA su PC139 ancora intatta** (b9119, niente pushato, niente riavviato). Restano:
`core/llm.py` di Lifelog2 (Roberto), swap binario + deploy (§9 passi 6, 9).
**Non-goal**: cambio modello. Resta `Qwen3-14B-Q4_K_M` (unsloth GGUF).

Chiamante unico verificato: `AriaLLMClient` in `sviluppi/Lifelog2/src/backend/lifelog2/core/llm.py`
(grep su tutti i progetti — DIAS non tocca il backend locale). Roberto (2026-09-08): **Lifelog2
userà entrambe le modalità** — thinking acceso/spento va scelto da un parametro passato dal
client. Superficie da toccare: `aria_node_controller/backends/lifelog_llm.py`,
`config/backends_manifest.json`, `aria_node_controller/core/orchestrator.py::_build_cmd`,
`backends/lifelog_llm/launcher.py`, `scripts/install_lifelog_llm.ps1`, e lato client `core/llm.py`.

> **L'allineamento ARIA↔Lifelog2 durante il rollout NON è un vincolo** (Roberto, 2026-09-08):
> quando ARIA/Qwen3 è pronto, Lifelog2 viene aggiornato *prima* di fare nuove richieste al
> backend. Niente finestra di retrocompatibilità da gestire.

---

## 1. Principio: 3 livelli, risolti a ogni richiesta

```
request_body = merge(
    1. BASE      — default del backend, non legati alla modalità (max_tokens, stream, seed...)
    2. PROFILE   — set COERENTE e completo per la modalità scelta (thinking | non_thinking)
    3. OVERRIDE  — solo i campi che il chiamante ha ESPLICITAMENTE passato, campo per campo
)
```

Precedenza: `OVERRIDE > PROFILE > BASE`. Merge *shallow* tranne `chat_template_kwargs` (*deep*, §4).

### 1.1 "Esplicitamente passato" — come si distingue

Oggi `payload.get("temperature", 0.6)` non sa dire se il chiamante ha chiesto `0.6` o non ha
chiesto nulla. Nuovo contratto: **il chiamante mette nel payload SOLO ciò che vuole forzare**.
Il wrapper guarda le *chiavi presenti*, non i valori.

- Chiavi consumate dal wrapper (non inoltrate): `messages`, `prompt`, `text`, `job_id`,
  `timeout_seconds`, `thinking`, `profile`.
- Altra chiave presente e riconosciuta dal contratto (§3) → **override** di quel campo.
- Chiave presente ma non nel contratto → `unknown_param_policy` (default **inoltra + WARN**;
  alternativa `reject`). Mai più "scarta in silenzio".

## 2. Selezione del profilo

| Input dal chiamante | Profilo |
|---|---|
| `profile: "thinking"` / `"non_thinking"` | quello indicato (precedenza) |
| `thinking: true` | `thinking` |
| `thinking: false` | `non_thinking` |
| niente | `contract.defaults.profile` (proposto: `non_thinking`) |

Il profilo porta con sé **l'intero set coerente** (sampling + meccanismo thinking + budget
reasoning), non un flag isolato — risolve il difetto di fondo dell'indagine §4.

### 2.1 Valori dei profili

Sampling: valori ufficiali Qwen3-14B (model-card + Unsloth, concordano alla lettera).
Meccanismo thinking: vedi §4. **Nomi campo e leve confermati dal vivo il 2026-09-09 (§8).**

```jsonc
"non_thinking": {
  "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
  "chat_template_kwargs": { "enable_thinking": false },  // ✅ leva primaria (verificata)
  "reasoning_budget_tokens": 0                           // ✅ leva ridondante, sampler (verificata)
},
"thinking": {
  "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
  "chat_template_kwargs": { "enable_thinking": true },
  "reasoning_budget_tokens": 2048,  // ✅ cap REALE sul pensiero (verificato: a 150 il pensiero
                                    //    si è fermato ~150 tok e la risposta finale è arrivata).
                                    //    Overridabile dal client. Evita che il reasoning si
                                    //    mangi tutto il tetto max_tokens condiviso.
  "reasoning_reserve_tokens": 2048  // = reasoning_budget_tokens, informativo per il client (§6)
}
```

> **Rimosso**: `reasoning_effort` — `chat_template_caps.supports_reasoning_effort = false` per
> il template di Qwen3-14B (verificato in `/props`). Non è una leva disponibile.
> **Nome campo**: è `reasoning_budget_tokens` (o alias `thinking_budget_tokens`), **non**
> `reasoning_budget` — quest'ultimo nel body richiesta viene ignorato.

> **Nota greedy**: Qwen avverte "DO NOT use greedy decoding" in thinking mode (degrado +
> ripetizioni infinite). Il wrapper **rifiuta con errore** `temperature: 0` (o `top_k: 1`)
> quando il profilo risolto è `thinking` — un errore esplicito al chiamante, non una
> correzione silenziosa che maschera un bug nel caller (deciso 2026-09-08).

## 3. Il manifest self-describing (requisito 4)

**SOT unica**: blocco `llm_contract` nell'entry `qwen3-14b-q4km` di
`aria_node_controller/config/backends_manifest.json` — già caricato da `_load_manifest`, già
nel percorso di deploy CT190 → git pull → PC139.

```jsonc
"qwen3-14b-q4km": {
  "metadata": { /* invariato */ },
  "port": 8090,
  "health_url": "http://localhost:8090/health",
  "startup_wait": 600,
  "env_prefix": "envs/lifelog-llm",
  "script": "backends/lifelog_llm/launcher.py",

  // --- NUOVO: args di avvio strutturati (era un array piatto — vedi §5) ---
  "server_args": {
    "model": "data/assets/models/Qwen3-14B-Q4_K_M/Qwen3-14B-Q4_K_M.gguf",
    "host": "0.0.0.0", "port": 8090,
    "n-gpu-layers": -1,
    "ctx-size": 32768,               // nativo Qwen3-14B (§7). VRAM misurata: ~10.8 GB. Vedi §10
    "cache-type-k": "q8_0",          // KV a 8 bit (verificato §8)
    "cache-type-v": "q8_0",
    "parallel": 1,                   // 1 solo consumatore seriale — default 4 spreca ~3 GB VRAM (§8)
    "jinja": true,
    "reasoning-format": "deepseek",  // verificato: pensiero in reasoning_content
    "reasoning-budget": -1,          // globale illimitato; il contenimento è per-richiesta
    "flash-attn": "on",              // richiesto dalla KV cache quantizzata
    "no-context-shift": true         // evita corruzione su generazioni lunghe (esempio Qwen)
  },

  // --- NUOVO: contratto self-describing ---
  "llm_contract": {
    "contract_version": 1,
    "model": {
      "family": "qwen3", "name": "Qwen3-14B", "quant": "Q4_K_M",
      "hf_source": "unsloth/Qwen3-14B-GGUF",
      "llama_server_build": "b10819",      // §7 — bump da b9119, pin esatto
      "context_window_tokens": 32768,      // = server_args.ctx-size, tenerli in sync
      "kv_cache": "q8_0"
    },
    "request_params": { /* tabella §3.1 */ },
    "profiles": { /* §2.1 */ },
    "defaults": { "profile": "non_thinking", "max_tokens": 4096, "stream": false },
    "unknown_param_policy": "forward_warn"
  }
}
```

### 3.1 `request_params` — parametri inoltrabili (dalla doc ufficiale llama-server + Qwen3)

**Legenda**: `req` = per-richiesta · `srv` = anche flag di avvio · *default* = default llama-server.

| Parametro | Tipo | Default | Racc. Qwen3-14B | Significato |
|---|---|---|---|---|
| **Sampling** | | | | |
| `max_tokens` (`n_predict`) | int | 4096 | dare spazio | Tetto token generati. Con `thinking` il pensiero attinge dallo stesso tetto (per questo il cap `reasoning_budget_tokens`) |
| `temperature` | float | 0.8 | 0.6 / 0.7 | Casualità. **Mai 0 in thinking** (il wrapper rifiuta) |
| `top_p` | float | 0.95 | 0.95 / 0.8 | Nucleus sampling |
| `top_k` | int | 40 | 20 / 20 | Top-K |
| `min_p` | float | 0.05 | 0.0 | Prob. minima relativa al token top |
| `presence_penalty` | float | 0.0 | 0–2 opz. (solo non-think) | Penalità presenza. Qwen: alto → *language mixing* |
| `frequency_penalty` | float | 0.0 | — | Penalità proporzionale alla frequenza |
| `repeat_penalty` | float | 1.0 | — | Penalità su sequenze ripetute (default effettivo b10819 = 1.0, non 1.1) |
| `repeat_last_n` | int | 64 | — | Finestra token per `repeat_penalty` |
| `seed` | int | -1 (random) | — | Seed RNG, riproducibilità |
| `samplers` | list[str] | — | — | Ordine di applicazione dei sampler |
| **Thinking / reasoning** (nomi/leve verificati §8, 2026-09-09) | | | | |
| `reasoning_budget_tokens` (alias `thinking_budget_tokens`) | int | -1 | 0 / 2048 | ✅ `0` = spegne il pensiero · `N>0` = cap token di pensiero (verificato) · `-1` illimitato. **NON** `reasoning_budget` (ignorato nel body) |
| `chat_template_kwargs` | object | — | `{"enable_thinking": bool}` | ✅ `{enable_thinking:false}` spegne il pensiero. Kwargs al template Jinja. *Deep merge* col profilo |
| `reasoning_format` | enum(srv) | `deepseek` (nostro `--reasoning-format`) | `deepseek` | ✅ pensiero estratto in `reasoning_content` separato da `content` |
| `response_format` | object | — | — | `{"type":"json_object"}` → JSON valido ma **ancora dentro fence markdown** su Qwen3-14B (Lifelog2 le strippa già). Non affidarsi al "raw" |
| `json_schema` | object | — | — | Schema per grammar-based sampling (non testato — più stringente di `json_object`) |
| `grammar` | string | — | — | Grammatica BNF-like |
| **Altri** | | | | |
| `stop` | list[str] | — | — | Stringhe di arresto |
| `ignore_eos` | bool | false | — | Continua oltre il token di fine |
| `logit_bias` | object | — | — | Modifica probabilità di token specifici |

> ❌ **`reasoning_effort`** — NON nel contratto: il template di Qwen3-14B non lo supporta
> (`chat_template_caps.supports_reasoning_effort = false`).
> ❌ **`enable_thinking` top-level** — ignorato; va dentro `chat_template_kwargs`.

*Accettati da llama-server ma NON nel profilo e non raccomandati da Qwen — restano ai default se
il client non li tocca, il passthrough li lascia passare comunque (policy `forward_warn`):*
`typical_p`, `mirostat`/`mirostat_tau`/`mirostat_eta`, `dry_multiplier`/`dry_base`/`dry_allowed_length`/`dry_penalty_last_n`,
`xtc_probability`/`xtc_threshold`, `top_n_sigma`, `n_probs`.

### 3.2 Esposizione del contratto

1. `LifelogLLMBackend.contract()` — classmethod, ritorna il blocco `llm_contract` parsato.
2. `probe()` — estende `is_loaded()`: unisce il contratto statico con `GET /props` di
   llama-server (`default_generation_settings`, `chat_template`, `model_path`, build reale).
   Il self-describing riflette il runtime, non solo il file.
3. Endpoint dashboard `GET /api/backends/qwen3-14b-q4km/contract` → output di `probe()`.
   **Rimandato** (deciso 2026-09-08): `contract()` + `probe()` come metodi bastano per
   implementazione e test; l'endpoint HTTP è comodità per il client, si aggiunge dopo se serve.

## 4. Controllo del thinking — meccanismo (verificato dal vivo, §8)

Roberto vuole thinking on/off da parametro client. Leve **verificate su b10819 + Qwen3-14B** il 2026-09-09:

| Meccanismo | Livello | Esito | Uso nel profilo |
|---|---|---|---|
| `chat_template_kwargs: {enable_thinking: false}` | template Jinja | ✅ funziona | `non_thinking` — **primaria** |
| `reasoning_budget_tokens: 0` | sampler (forza `</think>` subito) | ✅ funziona | `non_thinking` — ridondanza, indipendente dal template |
| ` /no_think` in coda all'ultimo `user` | istruzione al modello | ✅ funziona | fallback |
| `reasoning_budget_tokens: N>0` | sampler | ✅ funziona (cap reale) | `thinking` — cap sul pensiero (default 2048, overridabile) |
| `reasoning_effort` | template | ❌ non supportato dal template | — rimosso |
| `enable_thinking` top-level (non nested) | — | ❌ ignorato | deve stare in `chat_template_kwargs` |
| `reasoning_budget` (senza `_tokens`) | — | ❌ ignorato nel body | usare `reasoning_budget_tokens` |

Prerequisiti di avvio (§5): `--jinja` + `--reasoning-format deepseek` (entrambi verificati — il
pensiero finisce in `reasoning_content` separato da `content`).

Il profilo `non_thinking` usa **due leve indipendenti** (`enable_thinking:false` a livello
template + `reasoning_budget_tokens:0` a livello sampler): se una build futura rompesse il
percorso template, il sampler chiude comunque il pensiero. Il ` /no_think` resta come terzo
fallback — **fatto bene** stavolta: append al contenuto dell'ultimo messaggio `user`, non
"solo se non c'è un system message" (il bug attuale `lifelog_llm.py:48-51`).

## 5. Args di avvio server — da array piatto a dict

`_build_cmd` (orchestrator.py:193) oggi itera un array `args`. Nuovo: `server_args` come dict →
genera `--chiave valore`; booleano `true` → flag nudo (`--jinja`), `false` → omesso; valori
non-bool passati come sono (`--flash-attn on`). Risoluzione path invariata (`data/` → `aria_root`).

| Flag | Perché |
|---|---|
| `--jinja` | Template Qwen3 nativo (prerequisito §4). README: "default enabled" — **non darlo per scontato, esplicitarlo** |
| `--reasoning-format deepseek` | Estrazione corretta `<think>`/`</think>` → `reasoning_content`. Raccomandato Qwen |
| `--reasoning-budget -1` | Esplicito = globale illimitato. Il contenimento è per-richiesta (profilo), non globale |
| `--flash-attn on` | **Richiesto** dalla KV cache quantizzata (llama.cpp non fa q8_0 KV senza FA). Riduce anche l'impronta attention. Esempio Qwen usa `-fa` |
| `--no-context-shift` | Disabilita la rotazione del contesto — senza, generazioni lunghe si corrompono. Nell'esempio ufficiale Qwen |
| `--ctx-size 32768` | Contesto nativo Qwen3-14B (era 16384, metà). Vedi §7 per la KV math |
| `--cache-type-k q8_0` / `--cache-type-v q8_0` | KV cache a 8 bit. 32768 con KV fp16 ≈ 13.9 GB (non entra nei 13.1 liberi); con q8_0 ≈ 11.4 GB. Ladder di fallback se la qualità cala: K=q8_0/V=f16 (~12.6 GB, entra appena) → torna a ctx 16384 fp16 |

## 6. Lato client (`AriaLLMClient`) — cosa cambia

Resta **thin**. Oggi manda `messages, max_tokens, temperature, thinking`.

- Continua a passare `thinking` (bool) → il wrapper lo mappa a profilo. Per le chiamate che
  useranno il reasoning, passerà `thinking: true`.
- **Smette di forzare `temperature`** (oggi `0.3` hardcoded in `generate_json`): decide il
  profilo. Un caller che vuole una temperatura specifica la passa come override esplicito.
- `CONTEXT_WINDOW_TOKENS` → **costante `32768` con commento "deve combaciare con
  `llm_contract.model.context_window_tokens`"** (deciso 2026-09-08). Il fetch dall'endpoint
  §3.2 si valuta dopo, se e quando l'endpoint esiste — cambia una volta all'anno, non vale una
  dipendenza runtime da ARIA-su.
- `_THINKING_RESERVE_TOKENS` → allineato a `contract.profiles.thinking.reasoning_budget` (2048).
  Ora che il reasoning si userà davvero, questo budget diventa reale nella guardia overflow di
  `core/llm.py`.

### 6.1 `finish_reason` — nuovo campo di ritorno

Il wrapper propaga `choices[0].finish_reason` (`"stop"` | `"length"`) in `output.finish_reason`.
Lifelog2 oggi *indovina* il troncamento con `near_ceiling` (≥90% `max_tokens`); `finish_reason
== "length"` è la verità diretta. L'euristica resta come backstop.

## 7. Aggiornamento build llama-server: b9119 → **b10819** (verificato §8)

- Build target: **`b10819` (2026-09-05)**, variante **`cuda-13.3`**, pin esatto.
- Contesto: **`--ctx-size 32768` + KV `q8_0` + `--flash-attn on` + `--parallel 1`**.
- `cuda-13.1` non esiste per b10819 (varianti x64: `12.4`, `13.3`). `13.3` verificata sana (§8).

**Variante CUDA — verificato dal vivo su PC139 (2026-09-08/09):**
- Driver `610.88`, CUDA UMD `13.3`. CUDA Toolkit installato a sistema: **v13.2**
  (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\`, `cudart64_13.dll` + `cublas64_13.dll` su PATH).
- `tools/llama/` contiene solo `ggml-cuda.dll` — nessun cudart/cublas bundle: il setup b9119
  usa il **runtime CUDA 13.x di sistema**.
- → Scelta: **`llama-b10819-bin-win-cuda-13.3-x64.zip`** (142 MB) — combacia con driver e
  toolkit 13.x, continuità con l'attuale 13.1. `cuda-12.4` richiederebbe anche il `cudart-*-12.4`
  (373 MB, `cudart64_12.dll` non presente a sistema).
- Fallback se `ggml-cuda.dll` lamenta simboli mancanti contro il toolkit 13.2: aggiungere
  `cudart-llama-bin-win-cuda-13.3-x64.zip`.

**La build attuale `b9119` è del 2026-05-12 — 4 mesi.** I controlli per-richiesta del thinking
che ci servono (§4) sono stati aggiunti *dopo*:

| Fix | PR | Merge | In b9119? |
|---|---|---|---|
| `reasoning_effort: "none"` nell'API OAI (spegni reasoning, canonico) | #26045 | 2026-07-24 | ❌ |
| Per-request `reasoning_budget_tokens` onorato in chat completions | #23116 | 2026-07-12 | ❌ |
| Reasoning leak con template `<think>` force-open | #24674 | 2026-07-13 | ❌ |
| `chat_template_kwargs: {enable_thinking}` da client | #13196 | 2025-06-29 | ✅ (ma fragile, §4) |

Su `b9119`, l'unica leva per-richiesta è `chat_template_kwargs.enable_thinking` — proprio quella
con i dubbi di affidabilità. **Aggiornare la build è l'abilitatore diretto del requisito di Roberto.**

**`b10819` (2026-09-05) vs `b9119`** (~1700 commit) — **verificato dal vivo il 2026-09-09 (§8):**
- I 3 fix sopra confermati funzionanti: `reasoning_budget_tokens` per-richiesta (off + cap),
  `chat_template_kwargs.enable_thinking`, nessun reasoning leak. (`reasoning_effort` non
  supportato dal template Qwen3-14B — non è un problema, ci sono già 2 leve.)
- CUDA `13.3` sana: 1646 tok/s prompt processing su 8K token, generazione 42 tok/s, GPU 94% —
  nessun fallback cuBLAS.

**Deploy del binario** (§10 punto 2 — decisione aperta): il binario `b10819`/`cuda-13.3` è già
scaricato in `tools\llama-b10819\` su PC139. Va o sostituito in `tools\llama\` (eredita la
regola firewall inbound, ma perde il rollback "rinomina file") o tenuto separato + regola
firewall nuova. `launcher.py` punta a `tools/llama/llama-server.exe` hardcoded → comunque da
toccare. Poi aggiornare `$LLAMA_BUILD`/`$LLAMA_ZIP` in `install_lifelog_llm.ps1` per coerenza.
- Timing di inferenza da ri-misurare (baseline b9119: ~17-21s warm; ctx 2× + KV q8 possono spostarlo).

## 8. Verifica empirica — ESEGUITA 2026-09-09 su PC139

**Setup**: `llama-b10819-bin-win-cuda-13.3-x64` estratto in `tools\llama-b10819\` (dir separata,
`tools\llama\` b9119 intatta). `llama-server.exe` avviato su :8090 con
`--ctx-size 32768 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --parallel 1 --jinja
--reasoning-format deepseek --no-context-shift`. Test da localhost su PC139 (CT190→:8090 bloccato
dal firewall — solo `tools\llama\` è nelle regole inbound). Server fermato a fine test, stato
ripristinato.

### Risultati

| Verifica | Esito |
|---|---|
| `--version` | ✅ `build 10819, commit 6a1a922d2` (0.4.0-dev) |
| Caricamento modello, ctx 32768, `--parallel 1` | ✅ ~7s, `n_slots=1 n_ctx_slot=32768` |
| `chat_template_kwargs:{enable_thinking:false}` | ✅ spegne il pensiero (`reasoning_content` vuoto, `content` pieno, `finish=stop`) |
| `reasoning_budget_tokens: 0` | ✅ spegne il pensiero |
| ` /no_think` in coda all'ultimo `user` | ✅ spegne il pensiero |
| `reasoning_budget_tokens: 150` (cap, thinking ON) | ✅ pensiero ~150 tok poi **risposta finale prodotta** — cap reale |
| `reasoning_effort` | ❌ template non lo supporta (`/props` caps) |
| `enable_thinking` top-level | ❌ ignorato |
| `reasoning_budget` (senza `_tokens`) | ❌ ignorato nel body |
| `--reasoning-format deepseek` | ✅ pensiero in `reasoning_content` separato |
| `finish_reason` | ✅ `stop` / `length` corretti |
| `response_format:{type:json_object}` | ⚠️ JSON valido ma dentro fence ```json (Lifelog2 le strippa già) |
| **Perf generazione** | ✅ **~42 tok/s** stabile (23.5 ms/tok) |
| **Perf prompt processing** | ✅ **1646 tok/s su prompt da 8024 token** — CUDA 13.3 sano, nessun fallback cuBLAS, GPU util 94% |
| Context 32768 reale | ✅ prompt da 8K processato senza problemi |
| **VRAM** | ⚠️ `llama-server` **~10.8 GB** (pesi + KV q8, --parallel 1). Con desktop di Roberto attivo (~3 GB) → ~13.7 / 16.3 GB, **~2.4 GB liberi**. Vedi §10 |

### Impatto sul design
- Profilo `non_thinking`: `chat_template_kwargs:{enable_thinking:false}` + `reasoning_budget_tokens:0` (2 leve verificate).
- Profilo `thinking`: `reasoning_budget_tokens: N` è un **cap reale** sul pensiero (leva contro i troncamenti).
- `reasoning_effort` **rimosso** dal contratto.
- `--parallel 1` **aggiunto** ai `server_args` (default 4 → ~+3 GB VRAM inutili, 1 solo consumatore).
- CUDA `13.3` confermata sana su Q4_K_M — nessun bisogno di valutare `12.4`.

### Non ancora verificato (a implementazione)
- Qualità KV `q8_0` vs `f16` su un dump reale Stage D/E (serve un prompt vero — non c'era a mano).
- `chat_template_kwargs` deep-merge vs replace lato llama-server.
- `json_schema` (più stringente di `json_object`).

## 9. Piano di implementazione — §8 fatta, si può scrivere il wrapper

1. **Binario** (già in `tools\llama-b10819\` su PC139): su ok di Roberto —
   `ren tools\llama\llama-server.exe llama-server.b9119.exe` → estrarre il pacchetto b10819
   completo in `tools\llama\` (sovrascrive le DLL) → aggiornare `$LLAMA_BUILD="b10819"` e
   `$LLAMA_ZIP="...cuda-13.3..."` in `install_lifelog_llm.ps1`. Rollback = rinominare indietro.
2. ~~Verifica §8~~ — **FATTA** (risultati sopra).
3. ✅ **FATTO** — `backends_manifest.json`: `server_args` (dict) + `llm_contract` (19 `request_params`,
   2 profili), rimosso `args`. Altri backend intatti. JSON valido.
4. ✅ **FATTO** — `orchestrator.py::_build_cmd`: gestisce `server_args` dict (bool→flag nudo,
   path risolti su aria_root); `args` list ancora supportata per gli altri backend.
5. ✅ **FATTO** — `lifelog_llm.py::run()` riscritto: profilo, merge BASE<PROFILE<OVERRIDE,
   deep-merge `chat_template_kwargs`, validazione+clamp dal contratto, alias
   `thinking_budget_tokens`, greedy guard, `unknown_param_policy`, `finish_reason` in output,
   `contract()` + `probe()`. `_process_lifelog_llm_task` propaga `finish_reason`.
6. ⏳ `core/llm.py` (Lifelog2) — Roberto: stop forzatura `temperature`; `CONTEXT_WINDOW_TOKENS` → 32768;
   `_THINKING_RESERVE_TOKENS` = `reasoning_budget_tokens` del profilo; consuma `finish_reason`;
   decidi quali chiamate passano a `thinking: true`.
7. ✅ **FATTO** — `tests/test_lifelog_llm_wrapper.py`: 16 test offline (no GPU), tutti passati
   (pytest + standalone). ⏳ spot-check qualità KV q8 su dump Stage D reale — a deploy.
8. ✅ `docs/ARIA-blueprint.md` tabella modelli aggiornata (14b + 35b scaffolding). Wiki NH-Mini
   già allineata (log `[2026-09-08] lint`). ⏳ `docs/backends/qwen35-llm-moe.md` "mai realizzato".
9. ⏳ **Deploy** (su ok di Roberto): swap binario `tools\llama\` (§7), push da CT190,
   `git pull` PC139, restart backend.

## 10. Decisioni

**Prese:**
- Build → **`b10819`** fissa, variante **`cuda-13.3`** (13.1 non esiste per b10819; verificata sana §8).
- `--ctx-size` → **32768** + KV `q8_0` + `--flash-attn on` + `--parallel 1`. Tutto verificato §8.
- Toggle thinking → `chat_template_kwargs:{enable_thinking:false}` + `reasoning_budget_tokens:0`
  + ` /no_think` fallback. `reasoning_effort` non disponibile. Tutto verificato §8.
- `thinking` profile: `reasoning_budget_tokens: 2048` come cap di default (verificato che cappa davvero).
- `temperature: 0` / `top_k: 1` con profilo `thinking` → **rifiuto con errore**.
- Parametro non nel contratto → **`forward_warn`**.
- Profili → **nel manifest** (`llm_contract`).
- `CONTEXT_WINDOW_TOKENS` Lifelog2 → **costante `32768` con commento**.
- Endpoint dashboard `/contract` → **rimandato**.
- Allineamento ARIA↔Lifelog2 nel rollout → non un vincolo.

**Chiuse (Roberto, 2026-09-09):**
- **VRAM headroom**: si tiene **`ctx 32768`**. ~2.4 GB liberi col desktop attivo — accettato:
  la KV math regge, il test è passato, il semaforo GPU scarica il modello quando Roberto gioca.
  Rischio residuo noto: OOM se il desktop spinge forte durante un job (ARIA ritenta).
- **Deploy binario**: **sostituire in `tools\llama\`**. `llama-server.exe` → `llama-server.b9119.exe`
  (backup), poi estrarre il pacchetto `b10819`/`cuda-13.3` completo lì (sovrascrive tutte le DLL).
  Eredita la regola firewall inbound; `launcher.py` non cambia path. Rollback = rinominare indietro.

## 11. Fuori scope (registrato per non riaprirlo)

- Cambio modello (Qwen3.8-27B ecc.) — indagine §7.
- Upgrade quant Q5_K_M — zero rischio ma non richiesto ora.
- Passaggio a `cuda-12.4` — CUDA 13.3 verificata sana (§8), nessun motivo.
- `--reasoning-budget` positivo come cap **globale** — scartato (il cap è per-richiesta, dal profilo).
