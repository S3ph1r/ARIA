import os
import sys
import json
import logging
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Optional: ensure we are in the right directory or have path setup
# to load internal ARIA utils if needed, but for now we keep it minimal.

def main():
    """
    Main entry point for the Gemini worker process.
    Expected usage: python gemini_worker.py <payload_json>
    Outputs JSON result to stdout.
    """
    try:
        # 1. Setup Logging (to stderr so it doesn't pollute stdout JSON)
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)
        logger = logging.getLogger("gemini_worker")

        # 2. Parse Input
        if len(sys.argv) < 2:
            raise ValueError("Missing payload argument")
        
        payload = json.loads(sys.argv[1])
        book_id = payload.get("book_id", "unknown")
        job_id = payload.get("job_id", "unknown")
        
        logger.info(f"Gemini Worker started for job {job_id} (book {book_id})")

        # 3. Load Environment
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not found in environment.")

        # 4. Import SDK (only inside the process to avoid main-process overhead)
        try:
            import google.genai as genai
        except ImportError:
            raise ImportError("Libreria 'google-genai' non installata nell'ambiente del worker.")

        # 5. Execute Gemini Call
        client = genai.Client(api_key=api_key)
        
        # Use model_id if provided, otherwise default to gemini-1.5-flash
        model_name = payload.get("model_id") or payload.get("model") or "gemini-1.5-flash"
        contents   = payload.get("contents")
        config     = payload.get("config", {})

        if not contents:
            raise ValueError("Payload must contain 'contents'")

        logger.info(f"Calling Gemini ({model_name})...")
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )

        # 6. Return Success result as JSON on stdout
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
        # Return Error result as JSON
        result = {
            "status": "error",
            "error": str(e),
            "error_code": "GEMINI_WORKER_FAILED"
        }
        # Check for 429 specifically
        if "429" in str(e) or "exhausted" in str(e).lower():
            result["error_code"] = "QUOTA_EXHAUSTED"
            
        print(json.dumps(result))
        sys.exit(0) # Exit with 0 to allow the manager to read the JSON result

if __name__ == "__main__":
    main()
