# PowerShell script to automate ARIA Qwen3.5-35B LLM Backend installation
# Requirements: Miniconda/Anaconda installed, NVIDIA Drivers, CUDA 13+, VS 2022 Build Tools

$ARIA_ROOT = Get-Location
$ENV_NAME = "nh-qwen35-llm"
$ENV_PATH = "$ARIA_ROOT\envs\$ENV_NAME"

Write-Host "🚀 Starting ARIA LLM Backend Installation..." -ForegroundColor Cyan

# 1. Create Conda Environment
Write-Host "📦 Creating Conda environment..." -ForegroundColor Green
conda create --prefix $ENV_PATH python=3.12 -y

# 2. Build llama-cpp-python with GPU support
Write-Host "🛠️ Compiling llama-cpp-python for Blackwell (sm_120)..." -ForegroundColor Green
$env:CMAKE_ARGS = "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_GENERATOR_PLATFORM=x64"
& "$ENV_PATH\python.exe" -m pip install llama-cpp-python --no-cache-dir --verbose

# 3. Install other requirements
Write-Host "📚 Installing additional dependencies..." -ForegroundColor Green
& "$ENV_PATH\python.exe" -m pip install huggingface_hub fastapi uvicorn pydantic redis

# 4. Download Model
Write-Host "⬇️ Downloading Qwen3.5-35B-A3B-Instruct GGUF Q3_K_S..." -ForegroundColor Green
$MODEL_DIR = "$ARIA_ROOT\data\models\Qwen3.5-35B-A3B-GGUF"
if (-not (Test-Path $MODEL_DIR)) { New-Item -ItemType Directory -Path $MODEL_DIR }
& "$ENV_PATH\Scripts\huggingface-cli.exe" download bartowski/Qwen_Qwen3.5-35B-A3B-GGUF --include "*Q3_K_S.gguf*" --local-dir $MODEL_DIR

Write-Host "✅ Installation Complete!" -ForegroundColor Cyan
Write-Host "To start the backend, use: python aria_node_controller/llm_server.py"
