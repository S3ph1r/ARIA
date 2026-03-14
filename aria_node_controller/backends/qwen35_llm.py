"""
ARIA — Backend Qwen3.5-35B LLM (Headless Standalone)
======================================
Utilizza llama-server.exe (OpenAI API compatible) su porta 1234.
"""

import time
import logging
import requests
import json
import re
from pathlib import Path

logger = logging.getLogger("aria.backend.qwen35_llm")

# Default llama-server API URL (Porta 1234 standard per ARIA LLM)
LLAMA_SERVER_URL = "http://127.0.0.1:1234/v1"

class Qwen35LLMBackend:
    """
    Backend LLM per Qwen3.5-35B via llama-server.exe.
    Supporta l'estrazione del Thinking Mode dai tag <thought> o <think>.
    """

    model_id   = "qwen3.5-35b-moe-q3ks"
    model_type = "llm"

    def load(self, model_path: str, config: dict) -> None:
        """Il caricamento fisico è gestito dall'Orchestratore tramite il processo llama-server."""
        pass

    def is_loaded(self) -> bool:
        """Verifica se il server llama-server risponde."""
        try:
            response = requests.get(f"{LLAMA_SERVER_URL}/models", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def run(self, payload: dict, aria_root: Path, local_ip: str) -> dict:
        """
        Esegue l'inferenza LLM chiamando l'API OpenAI-compatible del server.
        """
        # Estrazione prompt dai vari formati possibili (DIAS vs Standard)
        messages = payload.get("messages")
        if not messages:
            prompt = payload.get("prompt", payload.get("text", "")).strip()
            if not prompt:
                raise ValueError("Campo 'messages' o 'prompt' obbligatorio")
            messages = [{"role": "user", "content": prompt}]

        request_body = {
            "model":       "qwen35-35b",
            "messages":    messages,
            "max_tokens":  payload.get("max_tokens", 4096),
            "temperature": payload.get("temperature", 0.7),
            "stream":      False
        }

        # Gestione Thinking Budget se il modello lo supporta (OpenAI style)
        if "thinking" in payload and payload["thinking"] is False:
             request_body["thinking_budget_tokens"] = 0

        timeout = payload.get("timeout_seconds", 600)
        server_ip = local_ip if local_ip else "127.0.0.1"
        url = f"http://{server_ip}:1234/v1/chat/completions"

        logger.info(f"Qwen3.5 request to {url} | tokens={request_body['max_tokens']}")

        try:
            response = requests.post(
                url,
                json=request_body,
                timeout=timeout
            )
            response.raise_for_status()
            full_result = response.json()
            
            content = full_result['choices'][0]['message']['content']
            
            thinking = ""
            text = content
            
            # Parsing "thinking" salvato separatamente se presente in tag
            thought_match = re.search(r'<thought>(.*?)</thought>', content, re.DOTALL | re.IGNORECASE)
            if not thought_match:
                thought_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)

            if thought_match:
                thinking = thought_match.group(1).strip()
                text = content.replace(thought_match.group(0), "").strip()
            
            return {
                "text":     text,
                "thinking": thinking,
                "usage":    full_result.get("usage")
            }

        except Exception as e:
            logger.error(f"Errore chiamata llama-server su {server_ip}: {e}")
            raise RuntimeError(f"Errore chiamata llama-server su {server_ip}: {e}")
