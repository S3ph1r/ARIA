import sys
import json
import os
from pathlib import Path

# Tentativo di setup SDK Google Moderno (google-genai)
try:
    from google import genai
    from google.genai import types
    MODERN_SDK = True
except ImportError:
    # Fallback per compatibilità (nonostante gli ambienti attuali siano aggiornati)
    try:
        import google.generativeai as genai
        MODERN_SDK = False
    except ImportError:
        print(json.dumps({"status": "error", "error": "Google SDK non installato", "error_code": "SDK_MISSING"}))
        sys.exit(0)

def main():
    if len(sys.argv) < 2:
        input_data = sys.stdin.read()
    else:
        input_data = sys.argv[1]

    try:
        task = json.loads(input_data)
        
        # 1. Recupero API Key
        api_key = task.get("config", {}).get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY mancante")

        # 2. Parametri Modello e Config
        model_name = task.get("model_id", "gemini-flash-lite-latest")
        temperature = task.get("config", {}).get("temperature", 0.7)
        max_tokens = task.get("config", {}).get("max_tokens", 4096)

        # 3. Preparazione Contenuti (unificata)
        contents = task.get("contents")
        if not contents:
            messages = task.get("messages", [])
            contents = []
            for m in messages:
                role = "user" if m["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})

        if not contents:
            text = task.get("text", "")
            if text:
                contents = [{"role": "user", "parts": [{"text": text}]}]
            else:
                raise ValueError("Nessun contenuto valido nel payload")

        # 4. Esecuzione Chiamata (Logica basata sull'SDK rilevato)
        if MODERN_SDK:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            response_text = response.text
            finish_reason = str(response.candidates[0].finish_reason) if response.candidates else "unknown"
        else:
            # Vecchia sintassi per compatibilità estrema
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                contents=contents,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )
            response_text = response.text
            finish_reason = str(response.candidates[0].finish_reason) if response.candidates else "unknown"

        # 5. Output Standardizzato
        result = {
            "status": "success",
            "output": {
                "text": response_text,
                "model_version": model_name,
                "finish_reason": finish_reason
            }
        }
        print(json.dumps(result))

    except Exception as e:
        result = {
            "status": "error",
            "error": str(e),
            "error_code": "GEMINI_WORKER_FAILED"
        }
        # Gestione Quota (compatibile con entrambi gli SDK)
        err_msg = str(e).lower()
        if "429" in err_msg or "exhausted" in err_msg:
            result["error_code"] = "QUOTA_EXHAUSTED"
            
        print(json.dumps(result))
        sys.exit(0)

if __name__ == "__main__":
    main()
