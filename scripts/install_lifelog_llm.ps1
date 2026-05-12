# ARIA — Lifelog LLM Backend Setup
# Crea env isolato envs/lifelog-llm, compila llama-cpp-python per Blackwell sm_120,
# scarica Qwen3-14B-Q4_K_M da Unsloth GGUF.
#
# Prerequisiti: Miniconda installato, NVIDIA Driver >= 570, CUDA Toolkit 12.8+,
#               Visual Studio 2022 Build Tools (C++), cmake in PATH.
#
# Eseguire da: C:\users\roberto\aria\   (la ARIA_ROOT)

param(
    [string]$AriaRoot = (Get-Location).Path
)

$ENV_NAME  = "lifelog-llm"
$ENV_PATH  = "$AriaRoot\envs\$ENV_NAME"
$MODEL_DIR = "$AriaRoot\data\assets\models\Qwen3-14B-Q4_K_M"

Write-Host "[1/5] Creo env conda $ENV_NAME..." -ForegroundColor Cyan
conda create --prefix $ENV_PATH python=3.12 -y
if ($LASTEXITCODE -ne 0) { Write-Error "Conda create fallito"; exit 1 }

Write-Host "[2/5] Compilo llama-cpp-python con CUDA sm_120a (Blackwell)..." -ForegroundColor Cyan
# CMAKE_CUDA_ARCHITECTURES=120a = RTX 5060 Ti / 5070 / 5080 / 5090
$env:CMAKE_ARGS = "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_GENERATOR_PLATFORM=x64"
$env:FORCE_CMAKE = "1"
& "$ENV_PATH\python.exe" -m pip install llama-cpp-python --no-cache-dir
if ($LASTEXITCODE -ne 0) { Write-Error "llama-cpp-python build fallito"; exit 1 }

Write-Host "[3/5] Installo dipendenze server..." -ForegroundColor Cyan
& "$ENV_PATH\python.exe" -m pip install fastapi "uvicorn[standard]" "pydantic>=2.0" requests huggingface_hub hf_transfer
if ($LASTEXITCODE -ne 0) { Write-Error "Pip install dipendenze fallito"; exit 1 }

Write-Host "[4/5] Scarico Qwen3-14B-Q4_K_M (unsloth/Qwen3-14B-GGUF)..." -ForegroundColor Cyan
if (-not (Test-Path $MODEL_DIR)) { New-Item -ItemType Directory -Path $MODEL_DIR | Out-Null }
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
& "$ENV_PATH\Scripts\huggingface-cli.exe" download `
    unsloth/Qwen3-14B-GGUF `
    --include "Qwen3-14B-Q4_K_M.gguf" `
    --local-dir $MODEL_DIR
if ($LASTEXITCODE -ne 0) { Write-Error "Download modello fallito"; exit 1 }

Write-Host "[5/5] Verifica installazione..." -ForegroundColor Cyan
& "$ENV_PATH\python.exe" -c "from llama_cpp import Llama; print('llama_cpp OK')"
if ($LASTEXITCODE -ne 0) { Write-Error "Verifica llama_cpp fallita"; exit 1 }

$gguf = "$MODEL_DIR\Qwen3-14B-Q4_K_M.gguf"
if (Test-Path $gguf) {
    $size = [math]::Round((Get-Item $gguf).Length / 1GB, 2)
    Write-Host "Modello OK: $gguf ($size GB)" -ForegroundColor Green
} else {
    Write-Error "File GGUF non trovato: $gguf"
    exit 1
}

Write-Host ""
Write-Host "Setup completato." -ForegroundColor Green
Write-Host "Avvio manuale di test:" -ForegroundColor Yellow
Write-Host "  $ENV_PATH\python.exe backends\lifelog_llm\server.py --model-path $MODEL_DIR\Qwen3-14B-Q4_K_M.gguf --port 8089"
