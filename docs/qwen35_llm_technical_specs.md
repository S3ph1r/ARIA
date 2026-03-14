# Technical Specification: ARIA Qwen 3.5 MoE VLM Backend

## Overview
This document outlines the specialized setup for running the **Qwen 3.5 35B MoE (A3B)** model with full multimodal (VLM) support on NVIDIA Blackwell hardware (RTX 5060 Ti). This configuration bypasses several regressions found in the `llama-cpp-python` Master branch.

## Environment Details
- **Conda Env**: `nh-qwen35-llm`
- **Python**: 3.12
- **CUDA Toolkit**: 13.2
- **Architecture**: `sm_120` (NVIDIA Blackwell)
- **Compiler Requirements**: MSVC v143 (Visual Studio 2022) with `Native Tools x64 Command Prompt`.
- **Mandatory Flags**: `/Zc:preprocessor` (Required for CUDA 13.2 header compatibility).

## Build Process (Surgical Edition)
Due to very recent regressions in the `llama.cpp` Master branch (March 2026), the standard `pip install` fails. We use a "Surgical-Fix" approach:

1. **Clone Master**: `git clone --recursive https://github.com/abetlen/llama-cpp-python.git`
2. **C++ Patch (mtmd)**: In `vendor/llama.cpp/tools/mtmd/CMakeLists.txt`, we force `VERSION "1.0.0"` because the `${LLAMA_INSTALL_VERSION}` variable is empty in manual builds, causing a CMake syntax error.
3. **Build Command**:
   ```cmd
   set "FORCE_CMAKE=1"
   set "CMAKE_GENERATOR=Ninja"
   set "CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CXX_FLAGS=/Zc:preprocessor -DCMAKE_CUDA_FLAGS=-Xcompiler=/Zc:preprocessor"
   python -m pip install . --force-reinstall --upgrade --no-cache-dir
   ```

## Current Known Issues & Troubleshooting (API-Shield)
The Python wrapper (`llama_cpp-python`) is currently out of sync with the latest `llama.cpp` API in Master. 

### Symptom: `AttributeError: function 'llama_get_kv_self' not found`
**Fix**: Surgically comment out the `@ctypes_function` binding for the deleted function in the installed `site-packages/llama_cpp/llama_cpp.py`.

### Symptom: `AttributeError: function 'llama_set_adapter_lora' not found`
**Fix**: Same as above. A full "API-Shield" scan is required to identify and comment out all functions that have been removed from the C++ core but remain in the Python wrapper.

## Model Configuration
- **Model**: `Qwen3.5-35B-A3B-Instruct-Q3_K_S.gguf`
- **Context Size**: 32k
- **Optimization**: 8-bit KV Cache enabled via `flash_attn` (where supported) or specialized llama.cpp flags to fit within 16GB VRAM.

## Orchestration
- **Server**: `llm_server.py` (FastAPI, Port 8085)
- **Thinking Mode**: The backend is configured to extract reasoning tags (`<thought>`) for the DIAS Scene Director.

---
*Last Updated: 2026-03-16 (Antigravity)*
