from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import requests
import json
import os
from typing import Optional

app = FastAPI(title="ARIA Orpheus TTS", version="1.0.0")

# Configurazione
LLAMA_SERVER_URL = os.getenv("ORPHEUS_API_URL", "http://llama-cpp-server:5006/v1/completions")
SAMPLE_RATE = int(os.getenv("ORPHEUS_SAMPLE_RATE", "24000"))

class TTSSRequest(BaseModel):
    model: str = "orpheus"
    input: str
    voice: str = "pietro"  # pietro, giulia, carlo
    response_format: str = "wav"
    speed: float = 1.0

class HealthResponse(BaseModel):
    status: str
    llama_server: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Controlla se llama-server è raggiungibile
        response = requests.get(f"{LLAMA_SERVER_URL.replace('/v1/completions', '')}/health", timeout=5)
        llama_status = "ok" if response.status_code == 200 else "error"
    except:
        llama_status = "unreachable"
    
    return HealthResponse(status="ok", llama_server=llama_status)

@app.post("/v1/audio/speech")
async def text_to_speech(request: TTSSRequest):
    """Converti testo in audio con Orpheus"""
    
    # Prepara il testo con la voce
    voice_text = f"{request.voice}: {request.input}"
    
    # Prepara la richiesta per llama.cpp
    llama_request = {
        "prompt": voice_text,
        "max_tokens": 2048,
        "temperature": 0.6,
        "top_p": 0.9,
        "stop": ["</s>", "<|im_end|>"],
        "stream": False
    }
    
    try:
        # Chiama llama.cpp server
        response = requests.post(LLAMA_SERVER_URL, json=llama_request, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        generated_text = result.get("choices", [{}])[0].get("text", "")
        
        if not generated_text:
            raise HTTPException(status_code=500, detail="Nessun testo generato")
        
        # Per ora restituiamo un WAV dummy - in produzione convertiremo l'output
        # Questo è un placeholder finché non implementiamo la conversione audio completa
        dummy_wav = generate_dummy_wav()
        
        return Response(
            content=dummy_wav,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=generated_{request.voice}.wav"}
        )
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"llama-server non raggiungibile: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la generazione: {str(e)}")

def generate_dummy_wav():
    """Genera un file WAV dummy per test - da sostituire con vera conversione"""
    # Questo è un WAV header minimale con dati dummy
    wav_header = bytes.fromhex(
        "52494646"  # RIFF
        "24000000"  # File size - 36
        "57415645"  # WAVE
        "666d7420"  # fmt
        "10000000"  # Subchunk1Size (16)
        "0100"       # AudioFormat (PCM)
        "0100"       # NumChannels (1)
        "803e0000"   # SampleRate (16000)
        "007d0000"   # ByteRate
        "0200"       # BlockAlign
        "1000"       # BitsPerSample (16)
        "64617461"   # data
        "00000000"   # Subchunk2Size (0)
    )
    return wav_header

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ORPHEUS_PORT", "5005"))
    host = os.getenv("ORPHEUS_HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)