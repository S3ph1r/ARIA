"""
Test offline (no GPU, no rete) del wrapper LifelogLLMBackend — redesign 2026-09-09.
Verifica: risoluzione profilo, merge BASE<PROFILE<OVERRIDE, deep-merge chat_template_kwargs,
validazione dal contratto, greedy guard, alias, unknown_param_policy, parsing risposta.

Run:  python3 tests/test_lifelog_llm_wrapper.py       (o: pytest tests/test_lifelog_llm_wrapper.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "aria_node_controller"))

from backends import lifelog_llm
from backends.lifelog_llm import LifelogLLMBackend


class _FakeResp:
    def __init__(self, payload):
        self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p


def _capture(monkey_payload=None):
    """Sostituisce requests.post, cattura il request_body, ritorna (backend, holder)."""
    holder = {}
    def fake_post(url, json=None, timeout=None):
        holder["url"] = url
        holder["body"] = json
        return _FakeResp(monkey_payload or {
            "choices": [{"finish_reason": "stop",
                         "message": {"content": "ciao", "reasoning_content": ""}}],
            "usage": {"completion_tokens": 2},
        })
    lifelog_llm.requests.post = fake_post
    return LifelogLLMBackend(), holder


def run(payload):
    b, h = _capture()
    out = b.run(payload, aria_root=Path("/x"), local_ip="127.0.0.1")
    return h["body"], out


CASES = []
def case(fn): CASES.append(fn); return fn


@case
def test_default_profile_non_thinking():
    body, _ = run({"messages": [{"role": "user", "content": "hi"}]})
    assert body["temperature"] == 0.7 and body["top_p"] == 0.8, body
    assert body["chat_template_kwargs"] == {"enable_thinking": False}, body
    assert body["reasoning_budget_tokens"] == 0, body
    assert body["max_tokens"] == 4096 and body["stream"] is False


@case
def test_thinking_true_selects_thinking_profile():
    body, _ = run({"messages": [{"role": "user", "content": "hi"}], "thinking": True})
    assert body["temperature"] == 0.6 and body["top_p"] == 0.95, body
    assert body["chat_template_kwargs"] == {"enable_thinking": True}, body
    assert body["reasoning_budget_tokens"] == 2048, body
    assert "reasoning_reserve_tokens" not in body  # meta, non va al server


@case
def test_explicit_profile_wins_over_thinking_bool():
    body, _ = run({"messages": [{"role": "user", "content": "x"}],
                   "thinking": True, "profile": "non_thinking"})
    assert body["chat_template_kwargs"] == {"enable_thinking": False}, body


@case
def test_per_field_override_keeps_rest_of_profile():
    body, _ = run({"messages": [{"role": "user", "content": "x"}],
                   "thinking": True, "temperature": 0.2})
    assert body["temperature"] == 0.2          # override
    assert body["top_p"] == 0.95               # dal profilo
    assert body["reasoning_budget_tokens"] == 2048


@case
def test_chat_template_kwargs_deep_merge():
    body, _ = run({"messages": [{"role": "user", "content": "x"}],
                   "thinking": True,
                   "chat_template_kwargs": {"extra": 1}})
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "extra": 1}, body


@case
def test_passthrough_unknown_forward_warn():
    body, _ = run({"messages": [{"role": "user", "content": "x"}],
                   "made_up_param": 123})
    assert body["made_up_param"] == 123, "forward_warn deve inoltrare comunque"


@case
def test_alias_thinking_budget_tokens():
    body, _ = run({"messages": [{"role": "user", "content": "x"}],
                   "thinking_budget_tokens": 512})
    assert body["reasoning_budget_tokens"] == 512, body
    assert "thinking_budget_tokens" not in body


@case
def test_validation_clamps_out_of_range():
    body, _ = run({"messages": [{"role": "user", "content": "x"}], "temperature": 9.0})
    assert body["temperature"] == 2.0, "temperature deve essere clampata a max"


@case
def test_validation_rejects_wrong_type():
    try:
        run({"messages": [{"role": "user", "content": "x"}], "top_k": "molti"})
    except ValueError:
        return
    raise AssertionError("atteso ValueError su top_k di tipo sbagliato")


@case
def test_greedy_guard_thinking():
    for bad in ({"temperature": 0}, {"top_k": 1}):
        try:
            run({"messages": [{"role": "user", "content": "x"}], "thinking": True, **bad})
        except ValueError:
            continue
        raise AssertionError(f"atteso ValueError su greedy in thinking: {bad}")


@case
def test_greedy_allowed_in_non_thinking():
    body, _ = run({"messages": [{"role": "user", "content": "x"}],
                   "profile": "non_thinking", "temperature": 0})
    assert body["temperature"] == 0  # ok fuori da thinking


@case
def test_prompt_shorthand_becomes_user_message():
    body, _ = run({"prompt": "riassumi"})
    assert body["messages"] == [{"role": "user", "content": "riassumi"}], body


@case
def test_reserved_keys_not_forwarded():
    body, _ = run({"messages": [{"role": "user", "content": "x"}],
                   "job_id": "abc", "timeout_seconds": 99})
    assert "job_id" not in body and "timeout_seconds" not in body


@case
def test_finish_reason_propagated():
    b, h = _capture({
        "choices": [{"finish_reason": "length",
                     "message": {"content": "ta", "reasoning_content": "pensiero"}}],
        "usage": {},
    })
    out = b.run({"messages": [{"role": "user", "content": "x"}]},
                aria_root=Path("/x"), local_ip="127.0.0.1")
    assert out["finish_reason"] == "length", out
    assert out["thinking"] == "pensiero" and out["text"] == "ta"


@case
def test_think_tag_fallback_parse():
    b, h = _capture({
        "choices": [{"finish_reason": "stop",
                     "message": {"content": "<think>ragiono</think>\nRisposta"}}],
        "usage": {},
    })
    out = b.run({"messages": [{"role": "user", "content": "x"}]},
                aria_root=Path("/x"), local_ip="127.0.0.1")
    assert out["thinking"] == "ragiono" and out["text"] == "Risposta", out


@case
def test_unknown_profile_raises():
    try:
        run({"messages": [{"role": "user", "content": "x"}], "profile": "turbo"})
    except ValueError:
        return
    raise AssertionError("atteso ValueError su profilo inesistente")


if __name__ == "__main__":
    ok = 0
    for fn in CASES:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            ok += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{ok}/{len(CASES)} passati")
    sys.exit(0 if ok == len(CASES) else 1)
