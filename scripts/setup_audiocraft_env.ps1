# ARIA Sound Library - Environment Setup Script
# Configurazione ottimizzata per RTX 5060 Ti (sm_120) e Audiocraft

$ENVPATHT = "C:\Users\Roberto\aria\envs\audiocraft-env"
$CONDAPATH = "C:\Users\Roberto\miniconda3\Scripts\conda.exe"

# 1. Creazione Ambiente Conda isolato
Write-Host ">>> Creating Conda environment in $ENVPATHT..." -ForegroundColor Cyan
& $CONDAPATH create --prefix $ENVPATHT python=3.10 -y

# 2. Installazione PyTorch 2.7+cu128 (Necessario per Blackwell sm_120)
Write-Host ">>> Installing PyTorch (CUDA 12.8 index)..." -ForegroundColor Cyan
& "$ENVPATHT\python.exe" -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Installazione Audio Engine (Audiocraft, Stable Audio, Pydub, FFmpeg)
Write-Host ">>> Installing Audio Generation tools..." -ForegroundColor Cyan
& "$ENVPATHT\python.exe" -m pip install --no-cache-dir audiocraft stable-audio-tools pydub ffmpeg-python transformers huggingface-hub

# 4. Installazione FFmpeg e Av via Conda (Più stabili su Windows)
Write-Host ">>> Installing FFmpeg and PyAV (static builds via conda)..." -ForegroundColor Cyan
& $CONDAPATH install --prefix $ENVPATHT -c conda-forge ffmpeg av -y

Write-Host ">>> Setup terminato correttamente." -ForegroundColor Green
Write-Host "Ambiente: $ENVPATHT" -ForegroundColor Yellow
