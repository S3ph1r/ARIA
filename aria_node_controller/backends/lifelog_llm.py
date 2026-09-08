"""
ARIA — Lifelog LLM Backend Client (Qwen3-14B-Q4_K_M via llama-server.exe :8090)

Consumatore unico: Lifelog2 (AriaLLMClient). Redesign 2026-09-08/09 — vedi
sviluppi/ARIA/docs/qwen3-llm-wrapper-redesign-2026-09-08.md.

Cosa fa `run()`:
  1. risolve un PROFILO (thinking | non_thinking) da payload["profile"] o payload["thinking"]
  2. costruisce i parametri di generazione a 3 livelli — BASE < PROFILE < OVERRIDE — dove
     OVERRIDE = solo le chiavi ESPLICITAMENTE presenti nel payload e note al contratto
  3. valida gli override contro lo schema del contratto (`llm_contract.request_params`)
  4. inoltra tutto a llama-server (niente più whitelist fissa che scartava in silenzio)
  5. propaga `finish_reason` nel risultato (troncamento certo, non stimato)

Il contratto (modello, versione, parametri configurabili, profili, default) vive nel
manifest — `aria_node_controller/config/backends_manifest.json`, blocco `llm_contract`
dell'entry `qwen3-14b-q4km` — ed è esposto da `contract()` / `probe()`.
"""

import json
import logging
import re
import requests
from pathlib import Path

logger = logging.getLogger("aria.backend.lifelog_llm")

LIFELOG_LLM_PORT = 8090

_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "config" / "backends_manifest.json"

# Chiavi che il wrapper consuma per sé — mai inoltrate come parametri di generazione.
_RESERVED_KEYS = {"messages", "prompt", "text", "job_id", "timeout_seconds", "thinking", "profile"}

# Chiavi di profilo che NON vanno a llama-server (meta per il client / il wrapper).
_PROFILE_META_KEYS = {"reasoning_reserve_tokens", "soft_switch"}


class LifelogLLMBackend:
    model_id   = "qwen3-14b-q4km"
    model_type = "llm"

    def __init__(self):
        self._contract_cache = None

    # ── Contratto self-describing ────────────────────────────────────────────
    def contract(self) -> dict:
        """Blocco `llm_contract` dell'entry del manifest. Caricato una volta e cachato
        (il manifest cambia solo su redeploy, che riavvia comunque l'orchestratore)."""
        if self._contract_cache is None:
            data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
            entry = data.get("backends", {}).get(self.model_id, {})
            c = entry.get("llm_contract")
            if not c:
                raise RuntimeError(
                    f"llm_contract mancante per '{self.model_id}' in {_MANIFEST_PATH}"
                )
            self._contract_cache = c
        return self._contract_cache

    def _alias_map(self) -> dict:
        """{alias -> nome canonico} dai `request_params` del contratto."""
        out = {}
        for name, spec in self.contract().get("request_params", {}).items():
            for alias in (spec.get("aliases") or []):
                out[alias] = name
        return out

    def is_loaded(self) -> bool:
        try:
            r = requests.get(f"http://127.0.0.1:{LIFELOG_LLM_PORT}/health", timeout=2)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            return False

    def probe(self, local_ip: str = "127.0.0.1") -> dict:
        """Contratto statico + ciò che il server dice davvero di sé via GET /props."""
        result = {"contract": self.contract(), "loaded": False, "server": None}
        try:
            r = requests.get(f"http://{local_ip}:{LIFELOG_LLM_PORT}/props", timeout=3)
            if r.ok:
                p = r.json()
                dg = p.get("default_generation_settings") or {}
                result["loaded"] = True
                result["server"] = {
                    "build_info": p.get("build_info"),
                    "n_ctx": dg.get("n_ctx"),
                    "model_path": p.get("model_path"),
                    "chat_template_caps": p.get("chat_template_caps"),
                }
        except Exception as e:  # server giù o non ancora pronto — non è un errore qui
            result["server_error"] = str(e)
        return result

    # ── Esecuzione ──────────────────────────────────────────────────────────
    def run(self, payload: dict, aria_root: Path, local_ip: str) -> dict:
        messages = payload.get("messages")
        if not messages:
            prompt = (payload.get("prompt") or payload.get("text") or "").strip()
            if not prompt:
                raise ValueError("Campo 'messages' o 'prompt' obbligatorio")
            messages = [{"role": "user", "content": prompt}]

        contract    = self.contract()
        profiles    = contract["profiles"]
        defaults    = contract.get("defaults", {})
        req_params  = contract.get("request_params", {})
        unknown_pol = contract.get("unknown_param_policy", "forward_warn")

        # 1. Profilo
        prof_name = self._resolve_profile_name(payload, profiles, defaults)
        profile   = profiles[prof_name]

        # 2. Override espliciti (solo chiavi presenti nel payload)
        overrides = self._collect_overrides(payload, req_params, unknown_pol)

        # 3. Validazione override contro lo schema del contratto
        for name, spec in list(req_params.items()):
            if name in overrides:
                overrides[name] = self._validate(name, overrides[name], spec)

        # 4. Greedy guard — Qwen: mai greedy in thinking (ripetizioni infinite)
        if prof_name == "thinking":
            eff_temp = overrides.get("temperature", profile.get("temperature"))
            eff_topk = overrides.get("top_k", profile.get("top_k"))
            if eff_temp == 0 or eff_topk == 1:
                raise ValueError(
                    "profilo 'thinking': temperature=0 / top_k=1 non ammessi "
                    "(Qwen: greedy decoding -> degrado + ripetizioni infinite)"
                )

        # 5. Merge BASE < PROFILE < OVERRIDE  (chat_template_kwargs in deep-merge)
        gen: dict = {
            "max_tokens": defaults.get("max_tokens", 4096),
            "stream": defaults.get("stream", False),
        }
        for k, v in profile.items():
            if k not in _PROFILE_META_KEYS and k != "chat_template_kwargs":
                gen[k] = v
        ctk = dict(profile.get("chat_template_kwargs") or {})

        over_ctk = overrides.pop("chat_template_kwargs", None)
        gen.update(overrides)
        if over_ctk:
            ctk.update(over_ctk)
        if ctk:
            gen["chat_template_kwargs"] = ctk

        # 6. soft-switch opzionale (profilo `soft_switch: " /no_think"` -> appende
        #    all'ultimo messaggio user). Non attivo nei profili di default: le leve
        #    template/sampler bastano (verificato §8). Rete per build future.
        soft = profile.get("soft_switch")
        if soft:
            messages = self._append_to_last_user(messages, soft)

        request_body = {"model": self.model_id, "messages": messages, **gen}

        timeout = payload.get("timeout_seconds", 300)
        url = f"http://{local_ip}:{LIFELOG_LLM_PORT}/v1/chat/completions"
        logger.info(
            "LifelogLLM -> %s | profilo=%s max_tokens=%s reasoning_budget_tokens=%s override=%s",
            url, prof_name, gen.get("max_tokens"),
            gen.get("reasoning_budget_tokens"), sorted(overrides.keys()),
        )

        try:
            resp = requests.post(url, json=request_body, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            msg = choice["message"]
            text = msg.get("content", "") or ""
            thinking = msg.get("reasoning_content", "") or ""

            # Fallback: <think> inline nel content (se --reasoning-format non è attivo)
            if not thinking and text:
                m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
                if m:
                    thinking = m.group(1).strip()
                    text = re.sub(r"<think>.*?</think>", "", text,
                                  flags=re.DOTALL | re.IGNORECASE).strip()

            return {
                "text":          text,
                "thinking":      thinking,
                "usage":         data.get("usage"),
                "finish_reason": choice.get("finish_reason"),
            }

        except Exception as e:
            logger.error("LifelogLLM call failed: %s", e)
            raise RuntimeError(f"LifelogLLM call failed: {e}")

    # ── Helper ──────────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_profile_name(payload: dict, profiles: dict, defaults: dict) -> str:
        name = payload.get("profile")
        if name is None:
            thinking = payload.get("thinking")
            if thinking is True:
                name = "thinking"
            elif thinking is False:
                name = "non_thinking"
            else:
                name = defaults.get("profile", "non_thinking")
        if name not in profiles:
            raise ValueError(f"Profilo '{name}' non definito nel contratto ({list(profiles)})")
        return name

    def _collect_overrides(self, payload: dict, req_params: dict, unknown_pol: str) -> dict:
        aliases = self._alias_map()
        overrides: dict = {}
        unknown: list = []
        for k, v in payload.items():
            if k in _RESERVED_KEYS:
                continue
            canonical = k if k in req_params else aliases.get(k)
            if canonical:
                overrides[canonical] = v
            else:
                unknown.append(k)
                if unknown_pol == "reject":
                    raise ValueError(f"parametro '{k}' non nel contratto (policy=reject)")
                overrides[k] = v  # forward_warn: inoltra comunque, mai scartare in silenzio
        if unknown:
            logger.warning(
                "LifelogLLM: parametri fuori contratto inoltrati (policy=%s): %s",
                unknown_pol, unknown,
            )
        return overrides

    @staticmethod
    def _validate(name: str, value, spec: dict):
        t = spec.get("type")
        type_ok = {
            "int":     lambda x: isinstance(x, int) and not isinstance(x, bool),
            "number":  lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
            "string":  lambda x: isinstance(x, str),
            "boolean": lambda x: isinstance(x, bool),
            "array":   lambda x: isinstance(x, list),
            "object":  lambda x: isinstance(x, dict),
        }.get(t)
        if type_ok and not type_ok(value):
            raise ValueError(f"parametro '{name}': atteso {t}, ricevuto {type(value).__name__}")

        if t in ("int", "number") and isinstance(value, (int, float)):
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None and value < lo:
                logger.warning("LifelogLLM: '%s'=%s < min %s -> clamp", name, value, lo)
                value = lo
            if hi is not None and value > hi:
                logger.warning("LifelogLLM: '%s'=%s > max %s -> clamp", name, value, hi)
                value = hi
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"parametro '{name}'={value!r} non in {spec['enum']}")
        return value

    @staticmethod
    def _append_to_last_user(messages: list, suffix: str) -> list:
        out = [dict(m) for m in messages]
        for m in reversed(out):
            if m.get("role") == "user":
                m["content"] = (m.get("content") or "") + suffix
                return out
        out.append({"role": "user", "content": suffix.strip()})
        return out
