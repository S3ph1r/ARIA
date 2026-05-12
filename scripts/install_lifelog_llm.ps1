# ARIA — Lifelog LLM Backend Setup
# Crea env isolato envs/lifelog-llm con llama-cpp-python (prebuilt cu128 wheel, no compile)
# e scarica Qwen3-14B-Q4_K_M da Unsloth GGUF.
#
# Prerequisiti: Miniconda installato, NVIDIA Driver >= 570, CUDA 12.8+
# Non servono VS Build Tools o cmake — usiamo wheel precompilato cu128.
#
# Eseguire da: C:\users\roberto\aria\   (la ARIA_ROOT)

param(
    [string]$AriaRoot = (Get-Location).Path
)

$ENV_NAME  = "lifelog-llm"
$ENV_PATH  = "$AriaRoot\envs\$ENV_NAME"
$MODEL_DIR = "$AriaRoot\data\assets\models\Qwen3-14B-Q4_K_M"
$LOG_PATH  = "$AriaRoot\logs\install_lifelog_llm.log"

function Log($msg) {
    $ts = Get-Date -Format "HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line -ForegroundColor Cyan
    Add-Content $LOG_PATH $line
}

New-Item -ItemType Directory -Path "$AriaRoot\logs" -Force | Out-Null
"" | Out-File $LOG_PATH  # reset log

Log "[1/5] Creo env conda $ENV_NAME (Python 3.12)..."
if (Test-Path $ENV_PATH) {
    Log "  Env gia' esistente, salto creazione."
} else {
    conda create --prefix $ENV_PATH python=3.12 -y 2>&1 | Tee-Object -Append $LOG_PATH
    if ($LASTEXITCODE -ne 0) { Log "ERRORE: conda create fallito"; exit 1 }
}

Log "[2/5] Installo llama-cpp-python prebuilt (cu128, include sm_120)..."
& "$ENV_PATH\python.exe" -m pip install llama-cpp-python `
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu128 `
    --no-cache-dir 2>&1 | Tee-Object -Append $LOG_PATH

if ($LASTEXITCODE -ne 0) {
    Log "  Prebuilt fallito — provo build da sorgente con nvcc 13.2..."
    $env:CMAKE_ARGS = "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_GENERATOR_PLATFORM=x64"
    $env:FORCE_CMAKE = "1"
    & "$ENV_PATH\python.exe" -m pip install llama-cpp-python --no-cache-dir 2>&1 | Tee-Object -Append $LOG_PATH
    if ($LASTEXITCODE -ne 0) { Log "ERRORE: llama-cpp-python build fallito"; exit 1 }
}

Log "[3/5] Installo dipendenze server (fastapi, uvicorn, etc.)..."
& "$ENV_PATH\python.exe" -m pip install `
    fastapi "uvicorn[standard]" "pydantic>=2.0" requests huggingface_hub hf_transfer `
    2>&1 | Tee-Object -Append $LOG_PATH
if ($LASTEXITCODE -ne 0) { Log "ERRORE: pip install dipendenze fallito"; exit 1 }

Log "[4/5] Scarico Qwen3-14B-Q4_K_M (unsloth/Qwen3-14B-GGUF)..."
if (-not (Test-Path $MODEL_DIR)) { New-Item -ItemType Directory -Path $MODEL_DIR | Out-Null }
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
& "$ENV_PATH\Scripts\huggingface-cli.exe" download `
    unsloth/Qwen3-14B-GGUF `
    --include "Qwen3-14B-Q4_K_M.gguf" `
    --local-dir $MODEL_DIR 2>&1 | Tee-Object -Append $LOG_PATH
if ($LASTEXITCODE -ne 0) { Log "ERRORE: Download modello fallito"; exit 1 }

Log "[5/5] Verifica installazione..."
& "$ENV_PATH\python.exe" -c "from llama_cpp import Llama; print('llama_cpp OK')" 2>&1 | Tee-Object -Append $LOG_PATH
if ($LASTEXITCODE -ne 0) { Log "ERRORE: import llama_cpp fallito"; exit 1 }

$gguf = "$MODEL_DIR\Qwen3-14B-Q4_K_M.gguf"
if (Test-Path $gguf) {
    $size = [math]::Round((Get-Item $gguf).Length / 1GB, 2)
    Log "Modello OK: $gguf ($size GB)"
} else {
    Log "ERRORE: File GGUF non trovato: $gguf"; exit 1
}

Log ""
Log "=== Setup completato ==="
Log "Test manuale server:"
Log "  $ENV_PATH\python.exe backends\lifelog_llm\server.py --model-path $MODEL_DIR\Qwen3-14B-Q4_K_M.gguf --port 8089"
