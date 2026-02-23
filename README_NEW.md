# ARIA Server - Orpheus TTS per RTX 5060 Ti

## 🎯 Obiettivo
Server TTS italiano basato su llama.cpp + Orpheus-FastAPI, ottimizzato per RTX 5060 Ti (Blackwell sm_120).

## 📁 Struttura Files

### File Principali (nuovi)
- `docker-compose-aria-gpu.yml` - Compose con 2 container
- `Dockerfile.llama-blackwell` - llama.cpp per RTX 5060 Ti
- `Dockerfile.gpu` - Orpheus-FastAPI wrapper
- `app.py` - API FastAPI per TTS
- `start-llama-server.sh` - Script avvio llama.cpp
- `.env` - Configurazione modelli
- `requirements.txt` - Dipendenze Python
- `aria-download.bat` - Download modello Windows

### File Legacy (backup in backup_legacy/)
- Tutto il codice ARIA precedente è stato spostato in `backup_legacy/`
- `redis_bridge_backup.py` - Componenti Redis per futura integrazione

## 🚀 Comandi per PC Gaming (Windows 11)

### 1. Installa dipendenze
```powershell
pip install huggingface-hub
```

### 2. Scarica modello italiano
```powershell
# Esegui dalla directory del progetto
.\aria-download.bat
```

### 3. Build e avvio
```powershell
docker-compose -f docker-compose-aria-gpu.yml build
docker-compose -f docker-compose-aria-gpu.yml up -d
```

### 4. Test
```powershell
# Health check
curl http://localhost:5005/health

# Test TTS
curl http://localhost:5005/v1/audio/speech -H "Content-Type: application/json" -d '{"model":"orpheus","input":"Ciao, sono una voce italiana","voice":"pietro","response_format":"wav"}' --output test.wav
```

## 📊 Architettura
```
Container 1: llama-cpp-server (porta 5006 interna)
  └── CUDA 12.8 + llama.cpp compilato per sm_120
  └── Modello: Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf

Container 2: orpheus-fastapi (porta 5005 esterna)
  └── FastAPI wrapper per API TTS
  └── Connette a llama-cpp-server:5006
```

## 🔧 Prossimi Step
1. Test completo con modello italiano
2. Integrazione con Redis Bridge (se necessario)
3. Implementazione conversione audio reale (attualmente dummy)