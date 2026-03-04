import requests
import base64
import os
import json
import time
from pathlib import Path

# Configurazione
ARIA_ROOT = r"C:\Users\Roberto\aria"
VOICE_PATH = os.path.join(ARIA_ROOT, "data", "voices", "narratore.wav")
SERVER_8081 = "http://localhost:8081"
SERVER_8080 = "http://localhost:8080"
OUT_PATH = r"D:\download\reconstruct-narrator.wav"

# Il testo esatto del reference (da usare sia come prompt che come target)
EXACT_TRANSCRIPT = (
    "Leggi la Bibbia, Brett? E allora ascolta questo passo che conosco a memoria, "
    "è perfetto per l'occasione: Ezechiele 25:17. Il cammino dell'uomo timorato è "
    "minacciato da ogni parte dalle iniquità degli esseri egoisti e dalla tirannia "
    "degli uomini malvagi. Benedetto sia colui che nel nome della carità e della "
    "buona volontà conduce i deboli attraverso la valle"
)

def run_diagnostic():
    print(f"--- DIAGNOSTIC: VOICE RECONSTRUCTION TEST ---")
    
    # 1. Encoding
    print(f"1. Encoding {VOICE_PATH} via 8081...")
    with open(VOICE_PATH, "rb") as f:
        audio_bytes = f.read()
    
    resp = requests.post(f"{SERVER_8081}/encode", files={"file": ("ref.wav", audio_bytes)})
    resp.raise_for_status()
    tokens_b64 = resp.json()["npy_base64"]
    print("   [OK] Tokens received.")

    # 2. Synthesis (Reconstruction)
    print(f"2. Requesting Reconstruction to 8080...")
    payload = {
        "text": EXACT_TRANSCRIPT,
        "references": [{
            "tokens": tokens_b64,
            "audio": tokens_b64,
            "text": EXACT_TRANSCRIPT
        }],
        "temperature": 0.7,
        "top_p": 0.7,
        "format": "wav"
    }
    
    start_t = time.time()
    resp = requests.post(f"{SERVER_8080}/v1/tts", json=payload)
    resp.raise_for_status()
    gen_audio = resp.content
    elapsed = time.time() - start_t
    print(f"   [OK] Synthesis done in {elapsed:.2f}s.")

    # 3. Save
    with open(OUT_PATH, "wb") as f:
        f.write(gen_audio)
    print(f"--- SUCCESS: Reconstruction saved to {OUT_PATH} ---")

if __name__ == "__main__":
    try:
        run_diagnostic()
    except Exception as e:
        print(f"!!! DIAGNOSTIC FAILED: {e}")
