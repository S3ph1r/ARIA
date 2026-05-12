"""
ARIA — Lifelog LLM Backend Client

Calls llama-server.exe (OpenAI-compatible) on port 8089.
Used by the orchestrator to dispatch enrichment tasks from Lifelog2.
"""

import re
import logging
import requests
from pathlib import Path

logger = logging.getLogger("aria.backend.lifelog_llm")

LIFELOG_LLM_PORT = 8090


class LifelogLLMBackend:
    model_id   = "qwen3-14b-q4km"
    model_type = "llm"

    def is_loaded(self) -> bool:
        try:
            r = requests.get(f"http://127.0.0.1:{LIFELOG_LLM_PORT}/health", timeout=2)
            return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            return False

    def run(self, payload: dict, aria_root: Path, local_ip: str) -> dict:
        messages = payload.get("messages")
        if not messages:
            prompt = payload.get("prompt", payload.get("text", "")).strip()
            if not prompt:
                raise ValueError("Campo 'messages' o 'prompt' obbligatorio")
            messages = [{"role": "user", "content": prompt}]

        request_body = {
            "model":       self.model_id,
            "messages":    messages,
            "max_tokens":  payload.get("max_tokens", 4096),
            "temperature": payload.get("temperature", 0.6),
            "top_p":       payload.get("top_p", 0.95),
            "top_k":       payload.get("top_k", 20),
            "min_p":       payload.get("min_p", 0.0),
            "stream":      False,
        }

        # Disable thinking mode: inject /no_think system prompt for Qwen3
        if payload.get("thinking") is False:
            if not any(m.get("role") == "system" for m in request_body["messages"]):
                request_body["messages"] = [{"role": "system", "content": "/no_think"}] + request_body["messages"]

        timeout = payload.get("timeout_seconds", 300)
        url = f"http://{local_ip}:{LIFELOG_LLM_PORT}/v1/chat/completions"

        logger.info("LifelogLLM request to %s | max_tokens=%d", url, request_body["max_tokens"])

        try:
            resp = requests.post(url, json=request_body, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            msg = data["choices"][0]["message"]
            # llama-server returns thinking in reasoning_content (separate field),
            # content holds only the visible answer
            text = msg.get("content", "") or ""
            thinking = msg.get("reasoning_content", "") or ""

            # Fallback: parse <think> tags if server embeds them in content
            if not thinking and text:
                m = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
                if m:
                    thinking = m.group(1).strip()
                    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

            return {
                "text":     text,
                "thinking": thinking,
                "usage":    data.get("usage"),
            }

        except Exception as e:
            logger.error("LifelogLLM call failed: %s", e)
            raise RuntimeError(f"LifelogLLM call failed: {e}")
