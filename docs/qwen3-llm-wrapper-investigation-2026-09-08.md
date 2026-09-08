# Indagine wrapper LLM locale (Qwen3-14B) — stato reale e requisiti nuovo design

**Data**: 2026-09-08
**Scopo**: handoff per una sessione dedicata al redesign del wrapper ARIA↔llama-server per
il backend LLM locale (`qwen3-14b-q4km`). Nato da un'indagine su Lifelog2/Stage D (troncamenti
e overlap nel thread builder) che ha portato a scoprire una serie di problemi strutturali nel
wrapper ARIA, non nel prompt — vedi `sviluppi/Lifelog2/docs/lifelog2-stage-d-prompt-iteration-2026-09-08.md`
e seguiti per il contesto Lifelog2 originale (quel filone resta aperto e riprende separatamente,
una volta che questo lavoro sblocca leve che oggi non abbiamo).

**Non-goal esplicito**: Roberto non vuole per ora sostituire Qwen3-14B con un modello diverso
(35B quantizzato o altro) — il lavoro di questa indagine riguarda SOLO il wrapper/config attorno
al modello attuale. Vedi sezione "Modello: versione e alternative" per perché.

**Regola operativa da rispettare**: mai avviare/fermare/riavviare processi ARIA su PC139
autonomamente — solo su ok esplicito di Roberto (vedi `restart_aria.py` in
`core-modules.mdc`, già marcato "non eseguire autonomamente"). Le verifiche fatte per
questa indagine sono state tutte in lettura (nvidia-smi, git log/status, dir, curl locale,
--version) — nessuna modifica allo stato di ARIA.

---

## 1. Come si arriva davvero al modello — flusso verificato

```
Lifelog2 (core/llm.py, AriaLLMClient.generate_json)
  → costruisce messages: [{"role":"system","content":system_prompt}, {"role":"user","content":prompt}]
  → payload Redis: {messages, max_tokens, temperature, thinking}
  → RPUSH su coda ARIA (Redis, CT120)
  → orchestratore ARIA (PC139) legge il job, dispatcha a backend qwen3-14b-q4km
  → aria_node_controller/backends/lifelog_llm.py :: run()
  → POST http://127.0.0.1:8090/v1/chat/completions (llama-server.exe)
  → risposta smontata in content/reasoning_content, torna su Redis (callback_key)
  → Lifelog2 fa BRPOP e legge il risultato
```

Sedi codice coinvolte:
- Lifelog2: `sviluppi/Lifelog2/src/backend/lifelog2/core/llm.py`
- ARIA: `sviluppi/ARIA/aria_node_controller/backends/lifelog_llm.py` (backend qwen3-14b)
- ARIA: `sviluppi/ARIA/aria_node_controller/config/backends_manifest.json` (config avvio server)
- ARIA: `sviluppi/ARIA/backends/lifelog_llm/launcher.py` (thin wrapper che esegue llama-server.exe)

---

## 2. Bug confermato: `thinking=False` non ha mai avuto effetto

File: `aria_node_controller/backends/lifelog_llm.py:48-51`, introdotto nel commit
`5849c6b` (12 maggio 2026, "fix(llm): parse reasoning_content ... disable thinking via /no_think"):

```python
# Disable thinking mode: inject /no_think system prompt for Qwen3
if payload.get("thinking") is False:
    if not any(m.get("role") == "system" for m in request_body["messages"]):
        request_body["messages"] = [{"role": "system", "content": "/no_think"}] + request_body["messages"]
```

L'iniezione di `/no_think` avviene SOLO se non esiste già un messaggio `system`. Ma
`llm.py:283-286` (lato Lifelog2) mette SEMPRE il proprio `system_prompt` come messaggio
`system` quando presente — vero per il 100% delle chiamate Stage D/D1/E (verificato:
tutti i caller passano `system_prompt=...`, mai `None`). Risultato: la guardia è sempre
`False`, `/no_think` non viene mai iniettato, e Qwen3 ha sempre ragionato (thinking ON di
default) su ogni chiamata da Lifelog2, in test come in produzione, dal 12 maggio 2026 a oggi.

Prima di quel commit, `thinking=False` impostava `temperature=0.6, top_k=1` — meccanismo
diverso, meno elegante, ma non condizionato dalla presenza di un system prompt.

**Conseguenza pratica**: nessun budget è mai stato riservato per il reasoning
(`_THINKING_RESERVE_TOKENS=1024` in `llm.py` scatta solo se il chiamante passa
`thinking=True`, cosa che nessun caller fa oggi) — il reasoning consuma token dallo stesso
tetto `max_tokens` della risposta finale, senza contabilità né controllo. Concorre (insieme
alla dimensione dell'output JSON, proporzionale al numero di thread prodotti) ai
troncamenti osservati sui casi Stage D più grandi (130-146 turni).

---

## 3. Whitelist rigida: ARIA non inoltra quello che gli mandiamo

`lifelog_llm.py::run()` costruisce `request_body` prendendo SOLO un set fisso di campi dal
payload: `model, messages, max_tokens, temperature, top_p, top_k, min_p, stream`, più il
caso speciale `thinking` (sopra, rotto). **Qualunque altro campo che Lifelog2 mettesse nel
payload verrebbe silenziosamente scartato** — non arriva a llama-server. Questo è il primo
requisito del nuovo design (sezione 6): passthrough esplicito, non whitelist chiusa.

Il backend gemello per un modello diverso, `qwen35_llm.py` (mai realmente in uso — vedi
sezione 5), prova a mandare `"thinking_budget_tokens": payload["thinking"] is False → 0`
a llama-server — **campo non documentato in nessuna fonte ufficiale trovata**, quasi
certamente ignorato silenziosamente dal server. Da non replicare/riusare come riferimento.

---

## 4. Cosa accetta REALMENTE llama-server (verificato su doc ufficiale, non congetture)

Fonti: [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
[Qwen3 su llama.cpp — doc ufficiale Qwen](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html),
[Qwen3-14B model card](https://huggingface.co/Qwen/Qwen3-14B).

**Parametri di richiesta (`/v1/chat/completions`) realmente esistenti, oggi non usati da noi:**
- `chat_template_kwargs` — oggetto libero passato al template Jinja; dentro può viaggiare
  `{"enable_thinking": false}` — è il meccanismo MODERNO per controllare il thinking di
  Qwen3, alternativo/migliore del vecchio `/no_think` testuale.
- `reasoning_effort` — livello (`minimal`/`low`/`medium`/`high`/`xhigh`/`max`/`default`)
  passato al template — supporto dipende dal modello specifico, da verificare se Qwen3-14B
  lo sfrutta.
- `reasoning_format` — come viene isolato il testo di pensiero (default `auto`).

**Flag di avvio server (CLI, non per-richiesta) oggi NON impostati nel manifest:**
- `--jinja` — necessario perché il chat-template nativo di Qwen3 (quello che gestisce
  `/no_think`, `<think>`, il soft-switch) venga applicato correttamente. Doc Qwen lo
  raccomanda esplicitamente insieme a `--reasoning-format deepseek`. La doc llama.cpp
  corrente dice che il default è "enabled" — ma non verificato per QUESTA build specifica
  (9119, vedi sezione 5), quindi da non dare per scontato.
- `--reasoning-format deepseek` — parsing corretto raccomandato per Qwen3 (stessi tag
  `<think>`/`</think>` di deepseek-r1). Oggi affidato al default implicito del server.
- `--reasoning-budget` — **risponde alla domanda "il modello ha un meccanismo che taglia
  il pensiero oltre un certo punto?"**: sì, esiste, ma è un flag di AVVIO SERVER (non di
  richiesta), default `-1` (illimitato). Un valore positivo forza la chiusura del
  ragionamento oltre quella soglia. Oggi non impostato — motivo diretto per cui la
  generazione può interrompersi a metà frase quando finisce `max_tokens`, senza nessun
  meccanismo che forzi una risposta finale valida.

**Attenzione — non tutto ciò che sembra documentato è affidabile al 100%:** esiste una
issue aperta su llama.cpp ([#20409](https://github.com/ggml-org/llama.cpp/issues/20409))
proprio su `enable_thinking` via `chat_template_kwargs` ignorato in certe build/shell su
Qwen3.5. Va verificato empiricamente sulla nostra build (9119) prima di fidarcene, non
assunto dalla doc da sola.

**Sampling params — scoperta indipendente**: i default usati oggi da `lifelog_llm.py`
(`temperature=0.6, top_p=0.95, top_k=20, min_p=0.0`) sono ESATTAMENTE i parametri
raccomandati dal model-card Qwen3-14B per la modalità THINKING. I raccomandati per
non-thinking sono diversi: `temperature=0.7, top_p=0.8, top_k=20, min_p=0`. Coerente col
fatto che il thinking non è mai stato davvero spento finora — ma un problema aggiuntivo
da correggere quando lo spegneremo per davvero: i due modi hanno bisogno di profili di
sampling diversi, non degli stessi default fissi indipendentemente da `thinking`.

---

## 5. Stato reale infrastruttura — verificato dal vivo il 2026-09-08

Accesso live confermato e funzionante: `ssh pc139` (alias in `~/.ssh/config` su CT190,
chiave già autorizzata, utente `roberto`). Nessun bisogno di tunnel — stessa LAN
(192.168.1.0/24), ping 0.7ms.

- **Build llama-server**: `9119 (ef93e98d0)`, Clang 19.1.5 per Windows x86_64 — combacia
  col manifest (`$LLAMA_BUILD = "b9119"` in `scripts/install_lifelog_llm.ps1`).
- **GPU**: NVIDIA RTX 5060 Ti, 16311 MiB VRAM totali. Snapshot 2026-09-08 (nulla caricato):
  2941 MiB usati (baseline Windows/altri processi), **13112 MiB liberi**.
- **Git PC139 == mirror CT190**: stesso HEAD (`8bcd1ec891b2cee4bebfa359fe071ac8fe1f3ec9`,
  2026-09-05), nessuna deriva rilevante per questo lavoro (solo modifiche locali non
  committate su acestep/file di backup, ininfluenti).
- **`qwen3.5-35b-moe-q3ks` è SOLO scaffolding, mai deployato**: il manifest
  (`backends_manifest.json`) e `model_registry.json` registrano l'entry (porta 8085, env
  `envs/nh-qwen35-llm` — esiste), e uno script wrapper FastAPI esiste
  (`backends/llm/server.py`, 14 aprile 2026, usa `llama_cpp` Python bindings diretti, non
  llama-server.exe) — MA la cartella pesi attesa
  (`data/assets/models/Qwen3.5-35B-A3B-GGUF/`) **non esiste sul disco**, nessuna traccia
  di download tentato (cache HF vuota, solo modelli pyannote). Lo script stesso solleva
  `FileNotFoundError` se lanciato. La doc interna `docs/backends/qwen35-llm-moe.md`
  (aprile 2026) descrive un piano mai realizzato, non lo stato reale — da correggere o
  marcare come tale se qualcuno la rilegge.
  Nota positiva: la config nello script abbandonato (`n_ctx=32768`, `flash_attn=True`,
  `type_k=8, type_v=8` — KV cache già quantizzata a 8 bit) mostra che il problema
  contesto-vs-VRAM era già stato ragionato correttamente, solo mai completato.

---

## 6. Requisiti per il nuovo design (da Roberto, 2026-09-08)

1. **Passthrough parametri completo**: il wrapper ARIA deve poter accettare ed effettivamente
   inoltrare a llama-server tutti i parametri realmente supportati dal server (vedi sezione 4),
   non una whitelist fissa arbitraria come oggi.
2. **Default lato ARIA quando il chiamante non specifica nulla** — non "niente succede in
   silenzio" come oggi, ma un default esplicito e documentato.
3. **Profili coerenti thinking/non-thinking**: quando il chiamante passa `thinking=True` o
   `thinking=False`, DEVE cambiare l'intero set di parametri coerenti con quella modalità
   (sampling, eventuale reasoning_format/effort, non solo l'iniezione/rimozione del
   `/no_think`) — non un singolo flag isolato che poi lascia sampling/altri parametri
   scollegati e sbagliati per quella modalità (esattamente il bug trovato in sezione 4).
   Un parametro esplicito passato dal chiamante deve poter fare override del default di
   quel profilo, campo per campo — non tutto o niente.
4. **Manifest self-describing lato backend**: il backend deve esporre (o avere un manifest
   associato) che dichiari modello, versione, e l'elenco dei parametri configurabili con i
   relativi default — non qualcosa che va dedotto leggendo il codice ogni volta (come
   abbiamo dovuto fare in questa indagine).

## 7. Modello: versione e alternative — quadro per non ripartire da zero

- Modello attuale: **Qwen3-14B-Q4_K_M** (dense, quant unsloth), generazione Qwen3 originale.
- **Non esistono pesi più recenti per la stessa taglia 14B**: la generazione 3.7 è stata
  saltata (mai rilasciata pubblicamente); le generazioni successive (3.5 Feb 2026, 3.6 Apr
  2026, 3.8 Ago 2026 — l'ultima) hanno spostato la taglia dense "di mezzo" a 27B, senza
  reintrodurre un 14B. Roberto ha esplicitamente detto di non voler cambiare modello ora —
  questa è solo l'informazione fattuale richiesta, non una proposta di azione.
- Per riferimento futuro, se/quando si vorrà riconsiderare: **Qwen3.8-27B** (27B dense,
  nativo multimodale, 262K contesto nativo, Apache-2.0, ago 2026) sta in 16-17GB via GGUF
  dinamici Unsloth — sarebbe l'unico vero "stesso ordine di grandezza" del 14B attuale,
  diverso dal salto a un MoE 30-35B quantizzato aggressivamente che era stato scartato.
  Non approfondito oltre in questa indagine — fuori scope per ora.
- **Matematica KV cache Qwen3-14B** (calcolo diretto, non da tabella online — formula
  standard GQA): 40 layer, 8 head KV, head_dim 128 →
  `2(K+V) × 40 × 8 × 128 × 2 byte(fp16) = 160 KiB/token`.
  - ctx 16384 (attuale): ~2.6 GB KV(fp16) + ~7.9 GB pesi Q4_K_M + ~1 GB overhead ≈ 11.5 GB
    — margine comodo sui 13.1 GB liberi reali.
  - ctx 32768 con KV fp16: ≈ 14.6 GB — **non ci sta** nei 13.1 GB liberi reali.
  - ctx 32768 con **KV cache a 8 bit** (`--cache-type-k q8_0 --cache-type-v q8_0`, flag
    reale e documentato di llama-server): ≈ 12 GB — **ci sta con margine**.
  - Upgrade quant a parità di architettura, se interessa più qualità senza cambiare
    contesto: Q5_K_M (~9.6 GB pesi) ci sta comodamente anche con ctx 16384 fp16 (~11.9 GB
    totali) — zero rischio, nessun cambio di modello.

---

## 8. Cosa NON è ancora stato fatto (per la nuova sessione)

- Nessuna modifica di codice applicata — solo lettura/indagine.
- Non verificato empiricamente se `--jinja` è già di fatto attivo di default sulla build
  9119 installata, né se `chat_template_kwargs.enable_thinking` funziona davvero su questa
  build (issue #20409 aperta upstream) — da testare con una chiamata reale prima di fidarsi
  della sola documentazione.
- Non deciso il design esatto del passthrough (schema payload, dove vivono i default per
  profilo, formato del manifest self-describing) — è il lavoro della prossima sessione.
- Non toccato il protocollo di deploy: modifiche vanno editate/pushate da CT190 (questo
  mirror), poi PC139 fa `git pull` — nessun avvio/riavvio di processi ARIA senza ok
  esplicito di Roberto.
