@echo off
echo [ARIA] Download modello Orpheus italiano...

set MODEL_NAME=Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf
set MODEL_REPO=lex-au/Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf
set MODEL_DIR=C:\models\orpheus

if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

if exist "%MODEL_DIR%\%MODEL_NAME%" (
    echo [ARIA] Modello gia' presente in %MODEL_DIR%\%MODEL_NAME%
    echo [ARIA] Nulla da fare.
    pause
    exit /b 0
)

echo [ARIA] Scarico da HuggingFace: lex-au/%MODEL_NAME%
echo [ARIA] Destinazione: %MODEL_DIR%

huggingface-cli download ^
    "lex-au/Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf" ^
    "%MODEL_NAME%" ^
    --local-dir "%MODEL_DIR%" ^
    --local-dir-use-symlinks False

if %ERRORLEVEL% NEQ 0 (
    echo [ARIA] ❌ Download fallito. Verifica connessione e huggingface-cli.
    pause
    exit /b 1
)

echo [ARIA] ✅ Modello scaricato correttamente in %MODEL_DIR%
pause