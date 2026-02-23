# Dockerfile.llama-blackwell
# Build llama.cpp per RTX 5060 Ti (Blackwell, sm_120, CUDA 12.8)
# Fix: linker error su Blackwell — usa multi-stage + flag corretti

# ─── STAGE 1: BUILD ─────────────────────────────────────────────────────────
# Usa cudnn-devel che include le librerie stub necessarie al linker
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Dipendenze build — includi libcurl che è richiesta da llama-server
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    ninja-build \
    libcurl4-openssl-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Clona llama.cpp
RUN git clone --depth=1 https://github.com/ggerganov/llama.cpp /llama.cpp

WORKDIR /llama.cpp

# Compila per Blackwell sm_120
# FIX 1: -j4 invece di -j$(nproc) — evita OOM del linker con troppi core
# FIX 2: -DGGML_CUDA_FORCE_MMQ=OFF — disabilita istruzioni non supportate su Blackwell
# FIX 3: -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" — fix linker Blackwell
# FIX 4: -DCMAKE_LIBRARY_PATH include le CUDA stubs
# FIX 5: -DLLAMA_CURL=ON deve essere dichiarato esplicitamente
RUN cmake -B build \
      -DGGML_CUDA=ON \
      -DLLAMA_CURL=ON \
      -DCMAKE_CUDA_ARCHITECTURES="120" \
      -DGGML_CUDA_F16=ON \
      -DGGML_CUDA_FORCE_MMQ=OFF \
      -DGGML_NATIVE=OFF \
      -DGGML_CCACHE=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_LIBRARY_PATH="/usr/local/cuda/lib64/stubs" \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
      -GNinja \
    && cmake --build build \
             --config Release \
             --target llama-server \
             -j4

# ─── STAGE 2: RUNTIME ────────────────────────────────────────────────────────
# Immagine finale leggera — solo runtime, non devel
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Solo le librerie runtime necessarie per eseguire llama-server
RUN apt-get update && apt-get install -y \
    libcurl4 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copia solo il binario compilato dallo stage builder
COPY --from=builder /llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

# Directory modelli (montata come volume da Windows C:\models\orpheus)
RUN mkdir -p /models

# Script di avvio
COPY start-llama-server.sh /start-llama-server.sh
RUN chmod +x /start-llama-server.sh

EXPOSE 5006

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD wget --quiet --tries=1 --spider http://localhost:5006/health || exit 1

CMD ["/start-llama-server.sh"]
