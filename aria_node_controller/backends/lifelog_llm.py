"""
ARIA — Lifelog LLM Backend Client

Calls backends/lifelog_llm/server.py (OpenAI-compatible) on port 8089.
Used by the orchestrator to dispatch enrichment tasks from Lifelog2.
"""

import re
import logging
import requests
from pathlib import Path

logger = logging.getLogger("aria.backend.lifelog_llm")

LIFELOG_LLM_PORT = 8089


class LifelogLLMBackend:
    model_id   = "qwen3-14b-q4km"
    model_type = "llm"

    def is_loaded(self) -> bool:
        try:
            r = requests.get(f"http://127.0.0.1:{LIFELOG_LLM_PORT}/v1/health", timeout=2)
            return r.status_code == 200 and r.json().get("status") == "ready"
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

        # Disable thinking mode if caller requests it
        if payload.get("thinking") is False:
            request_body["temperature"] = 0.6
            request_body["top_k"] = 1

        timeout = payload.get("timeout_seconds", 300)
        url = f"http://{local_ip}:{LIFELOG_LLM_PORT}/v1/chat/completions"

        logger.info("LifelogLLM request to %s | max_tokens=%d", url, request_body["max_tokens"])

        try:
            resp = requests.post(url, json=request_body, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"]
            thinking, text = "", content

            m = re.search(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
            if m:
                thinking = m.group(1).strip()
                text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()

            return {
                "text":     text,
                "thinking": thinking,
                "usage":    data.get("usage"),
            }

        except Exception as e:
            logger.error("LifelogLLM call failed: %s", e)
            raise RuntimeError(f"LifelogLLM call failed: {e}")
