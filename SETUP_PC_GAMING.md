# ARIA Server - Docker Setup per RTX 5060 Ti

## 📋 Istruzioni per PC Gaming (Windows 11)

### 1. Preparazione Ambiente
```powershell
# Su PowerShell (Run as Administrator)
# Installa huggingface-cli se non presente
pip install huggingface-hub

# Crea directory modelli
mkdir C:\models\orpheus
```

### 2. Scarica Modello Italiano
```powershell
# Scarica il modello Orpheus italiano
huggingface-cli download lex-au/Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf --local-dir C:\models\orpheus --local-dir-use-symlinks False
```

### 3. Build e Avvio Container
```powershell
# Build immagini Docker
docker-compose -f docker-compose-aria-gpu.yml build

# Avvia container
docker-compose -f docker-compose-aria-gpu.yml up -d
```

### 4. Test
```powershell
# Test health check
curl http://localhost:5006/health

# Test API TTS
curl http://localhost:5005/v1/audio/speech -H "Content-Type: application/json" -d '{"model":"orpheus","input":"Ciao, sono una voce italiana","voice":"pietro","response_format":"wav"}' --output test.wav
```

### 5. Log e Monitoraggio
```powershell
# Log llama-server
docker logs -f aria-llama-server

# Log orpheus-api
docker logs -f aria-orpheus-api
```

## 🎯 Voci Italiane Disponibili
- `pietro` - Maschile, appassionato
- `giulia` - Femminile, espressiva
- `carlo` - Maschile, raffinato

## ⚠️ Requisiti Importanti
- CUDA 12.8 (automatico nel container)
- Architettura sm_120 per RTX 5060 Ti
- 4GB VRAM liberi per il modello