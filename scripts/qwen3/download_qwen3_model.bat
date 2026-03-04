@echo off
:: ============================================================
:: ARIA — Download Qwen3-TTS-12Hz-1.7B-Base
:: Scarica ~3.5GB da HuggingFace in C:\models\qwen3-tts-1.7b
:: ============================================================

echo.
echo  ARIA — Download Qwen3-TTS-12Hz-1.7B-Base
echo  ==========================================
echo.

call conda activate qwen3-tts

echo Download in corso verso C:\Users\Roberto\aria\data\models\qwen3-tts-1.7b...

echo (Dimensione attesa: ~3.5GB — dipende dalla connessione)
echo.

:: Metodo principale: huggingface-cli
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base ^
  --local-dir C:\Users\Roberto\aria\data\models\qwen3-tts-1.7b ^
  --exclude "*.msgpack" "*.h5" "flax_model*" "tf_model*"


if %errorlevel% equ 0 (
    echo.
    echo  Download completato in C:\Users\Roberto\aria\data\models\qwen3-tts-1.7b
    dir C:\Users\Roberto\aria\data\models\qwen3-tts-1.7b
) else (
    echo.
    echo [WARN] huggingface-cli ha restituito un errore. Provo con Python...
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='C:/Users/Roberto/aria/data/models/qwen3-tts-1.7b', ignore_patterns=['*.msgpack','*.h5','flax_model*','tf_model*'])"

)

echo.
pause
