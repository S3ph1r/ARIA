# ARIA — Lifelog LLM Backend Setup
# Scarica llama-server.exe (prebuilt CUDA 13.1, include sm_120) e il modello Qwen3-14B-Q4_K_M.
#
# Prerequisiti: Miniconda installato, NVIDIA Driver >= 570
# Non servono VS Build Tools, cmake, o CUDA toolkit — usiamo il binario precompilato.
#
# Eseguire da: C:\users\roberto\aria\   (la ARIA_ROOT)

param(
    [string]$AriaRoot = (Get-Location).Path
)

$ENV_NAME    = "lifelog-llm"
$ENV_PATH    = "$AriaRoot\envs\$ENV_NAME"
$MODEL_DIR   = "$AriaRoot\data\assets\models\Qwen3-14B-Q4_K_M"
$LLAMA_DIR   = "$AriaRoot\tools\llama"
$LOG_PATH    = "$AriaRoot\logs\install_lifelog_llm.log"

# llama.cpp b9119, CUDA 13.1 build (includes sm_120 for RTX 5060 Ti)
$LLAMA_BUILD = "b9119"
$LLAMA_ZIP   = "llama-$LLAMA_BUILD-bin-win-cuda-13.1-x64.zip"
$LLAMA_URL   = "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD/$LLAMA_ZIP"

function Log($msg) {
    $ts = Get-Date -Format "HH:mm:ss"
    $line = "[$ts] $msg"
    Write-Host $line -ForegroundColor Cyan
    Add-Content $LOG_PATH $line
}

New-Item -ItemType Directory -Path "$AriaRoot\logs" -Force | Out-Null
"" | Out-File $LOG_PATH  # reset log

Log "[1/5] Creo env conda $ENV_NAME (Python 3.12, solo huggingface_hub)..."
if (Test-Path $ENV_PATH) {
    Log "  Env gia' esistente, salto creazione."
} else {
    conda create --prefix $ENV_PATH python=3.12 -y 2>&1 | Tee-Object -Append $LOG_PATH
    if ($LASTEXITCODE -ne 0) { Log "ERRORE: conda create fallito"; exit 1 }
}

Log "[2/5] Installo huggingface_hub + hf_transfer..."
& "$ENV_PATH\python.exe" -m pip install huggingface_hub hf_transfer --no-cache-dir 2>&1 | Tee-Object -Append $LOG_PATH
if ($LASTEXITCODE -ne 0) { Log "ERRORE: pip install fallito"; exit 1 }

Log "[3/5] Scarico llama-server.exe ($LLAMA_BUILD, CUDA 13.1, sm_120)..."
New-Item -ItemType Directory -Path $LLAMA_DIR -Force | Out-Null
$zipPath = "$LLAMA_DIR\$LLAMA_ZIP"
if (-not (Test-Path "$LLAMA_DIR\llama-server.exe")) {
    Log "  Download da: $LLAMA_URL"
    Invoke-WebRequest -Uri $LLAMA_URL -OutFile $zipPath -UseBasicParsing
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $zipPath)) { Log "ERRORE: Download llama.cpp fallito"; exit 1 }

    Log "  Estrazione zip..."
    Expand-Archive -Path $zipPath -DestinationPath $LLAMA_DIR -Force
    Remove-Item $zipPath -Force

    # Binaries land in a build/ subdir — flatten
    $buildDir = Get-ChildItem -Path $LLAMA_DIR -Directory | Select-Object -First 1
    if ($buildDir) {
        Move-Item "$($buildDir.FullName)\*" $LLAMA_DIR -Force
        Remove-Item $buildDir.FullName -Recurse -Force
    }
} else {
    Log "  llama-server.exe gia' presente, salto download."
}

if (-not (Test-Path "$LLAMA_DIR\llama-server.exe")) {
    Log "ERRORE: llama-server.exe non trovato dopo estrazione"; exit 1
}
Log "  llama-server.exe OK"

Log "[4/5] Scarico Qwen3-14B-Q4_K_M (unsloth/Qwen3-14B-GGUF)..."
if (-not (Test-Path $MODEL_DIR)) { New-Item -ItemType Directory -Path $MODEL_DIR | Out-Null }
$gguf = "$MODEL_DIR\Qwen3-14B-Q4_K_M.gguf"
if (-not (Test-Path $gguf)) {
    $env:HF_HUB_ENABLE_HF_TRANSFER = "1"
    & "$ENV_PATH\Scripts\huggingface-cli.exe" download `
        unsloth/Qwen3-14B-GGUF `
        --include "Qwen3-14B-Q4_K_M.gguf" `
        --local-dir $MODEL_DIR 2>&1 | Tee-Object -Append $LOG_PATH
    if ($LASTEXITCODE -ne 0) { Log "ERRORE: Download modello fallito"; exit 1 }
} else {
    Log "  Modello gia' presente, salto download."
}

Log "[5/5] Verifica finale..."
if (Test-Path $gguf) {
    $size = [math]::Round((Get-Item $gguf).Length / 1GB, 2)
    Log "  Modello OK: $gguf ($size GB)"
} else {
    Log "ERRORE: File GGUF non trovato: $gguf"; exit 1
}
if (Test-Path "$LLAMA_DIR\llama-server.exe") {
    Log "  Binary OK: $LLAMA_DIR\llama-server.exe"
} else {
    Log "ERRORE: llama-server.exe non trovato"; exit 1
}

Log ""
Log "=== Setup completato ==="
Log "Test manuale server:"
Log "  $LLAMA_DIR\llama-server.exe -m $gguf --port 8089 --host 0.0.0.0 --n-gpu-layers -1 --ctx-size 16384"
Log ""
Log "Health check (dopo avvio):"
Log "  curl http://localhost:8089/health"
