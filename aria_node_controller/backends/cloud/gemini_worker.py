import sys
import json
import os
from pathlib import Path

# Tentativo di setup SDK Google in ambiente isolato
try:
    import google.generativeai as genai
except ImportError:
    # Se fallisce qui, il worker non può funzionare
    print(json.dumps({"status": "error", "error": "google-generativeai non installato", "error_code": "SDK_MISSING"}))
    sys.exit(0)

def main():
    if len(sys.argv) < 2:
        # Leggi da stdin se non passato via arg
        input_data = sys.stdin.read()
    else:
        input_data = sys.argv[1]

    try:
        task = json.loads(input_data)
        
        # 1. Recupero API Key (Priorità: payload config -> Env)
        api_key = task.get("config", {}).get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY mancante")

        genai.configure(api_key=api_key)

        # 2. Configurazione Modello
        model_name = task.get("model_id", "gemini-flash-lite-latest")
        model = genai.GenerativeModel(model_name)

        # 3. Preparazione Payload (Standard Google: contents)
        # Se DIAS invia 'messages' (OpenAI), convertiamo in 'contents' (Google)
        contents = task.get("contents")
        if not contents:
            messages = task.get("messages", [])
            contents = []
            for m in messages:
                role = "user" if m["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})

        if not contents:
            # Fallback a text se presente
            text = task.get("text", "")
            if text:
                contents = [{"role": "user", "parts": [{"text": text}]}]
            else:
                raise ValueError("Nessun contenuto valido nel payload (contents/messages/text)")

        # 4. Generazione (Blocking)
        config = {
            "temperature": task.get("config", {}).get("temperature", 0.7),
            "max_output_tokens": task.get("config", {}).get("max_tokens", 4096),
        }

        response = model.generate_content(
            contents=contents,
            generation_config=config
        )

        # 5. Ritorno Risultato su stdout
        result = {
            "status": "success",
            "output": {
                "text": response.text,
                "model_version": model_name,
                "finish_reason": str(response.candidates[0].finish_reason) if response.candidates else "unknown"
            }
        }
        print(json.dumps(result))

    except Exception as e:
        result = {
            "status": "error",
            "error": str(e),
            "error_code": "GEMINI_WORKER_FAILED"
        }
        if "429" in str(e) or "exhausted" in str(e).lower():
            result["error_code"] = "QUOTA_EXHAUSTED"
            
        print(json.dumps(result))
        sys.exit(0)

if __name__ == "__main__":
    main()
