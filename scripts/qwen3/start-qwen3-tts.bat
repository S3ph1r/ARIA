@echo off
:: ============================================================
:: ARIA — Avvio Qwen3-TTS Server (porta 8083)
:: Da aggiungere in Task Scheduler con ritardo 90s
:: ============================================================

echo.
echo  ARIA — Qwen3-TTS Server
echo  Porta: 8083  -  Modello: Qwen3-TTS-12Hz-1.7B-Base
echo.

:: Variabili configurabili
set QWEN3_MODEL_PATH=C:\Users\Roberto\aria\data\models\qwen3-tts-1.7b
set ARIA_OUTPUT_DIR=C:\Users\Roberto\aria\data\outputs
set QWEN3_HOST=0.0.0.0
set QWEN3_PORT=8083

:: Cambia directory
cd /d C:\Users\Roberto\aria

:: Attiva ambiente conda e avvia server
call conda activate qwen3-tts
if %errorlevel% neq 0 (
    echo [ERRORE] Ambiente conda 'qwen3-tts' non trovato.
    echo Esegui prima: scripts\qwen3\setup_qwen3_env.bat
    pause
    exit /b 1
)

echo Avvio qwen3_server.py...
python scripts\qwen3\qwen3_server.py

:: Se il processo finisce (crash), attende prima di chiudersi
echo.
echo [WARN] Server terminato. Premi un tasto per chiudere.
pause
