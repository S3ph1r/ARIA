@echo off
:: ============================================================
:: ARIA — Setup Ambiente Qwen3-TTS 1.7B
:: Esegui come Amministratore in Anaconda Prompt
:: Target: C:\Users\Roberto\aria\  (Windows 11, RTX 5060 Ti)
:: ============================================================

echo.
echo  ARIA — Setup Qwen3-TTS 1.7B Environment
echo  =========================================
echo.

:: Verifica che conda sia disponibile
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRORE] conda non trovato! Assicurati di eseguire in Anaconda Prompt.
    pause
    exit /b 1
)

:: 1. Crea ambiente conda Python 3.12
echo [1/5] Creazione ambiente conda 'qwen3-tts' con Python 3.12...
conda create -n qwen3-tts python=3.12 -y
if %errorlevel% neq 0 (
    echo [WARN] L'ambiente potrebbe già esistere. Continuando...
)

:: 2. Attiva ambiente
echo [2/5] Attivazione ambiente...
call conda activate qwen3-tts

:: 3. Installa PyTorch cu128 (critico per RTX 5060 Ti / sm_120)
echo [3/5] Installazione PyTorch cu128 (Blackwell-compatibile)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if %errorlevel% neq 0 (
    echo [ERRORE] Installazione PyTorch fallita.
    pause
    exit /b 1
)

:: 4. Installa dipendenze
echo [4/5] Installazione dipendenze Qwen3-TTS...
pip install "transformers>=4.52.0" "accelerate>=1.7.0" soundfile numpy fastapi uvicorn requests huggingface_hub
if %errorlevel% neq 0 (
    echo [ERRORE] Installazione dipendenze fallita.
    pause
    exit /b 1
)

:: 5. Flash Attention (opzionale — se non disponibile il modello funziona ugualmente)
echo [5/5] Tentativo installazione flash-attention 2 (opzionale)...
pip install flash-attn --no-build-isolation 2>nul
if %errorlevel% neq 0 (
    echo [INFO] flash-attention non installata (ok, +15%% latenza / +800MB VRAM).
)

:: 6. Verifica PyTorch
echo.
echo Verifica installazione PyTorch...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo  Setup completato!
echo  Prossimi passi:
echo    1. download_qwen3_model.bat  (scarica modello ~3.5GB)
echo    2. create_padded_ref.py      (prepara ref_padded.wav)
echo    3. test_qwen3_direct.py      (verifica qualita')
echo.
pause
