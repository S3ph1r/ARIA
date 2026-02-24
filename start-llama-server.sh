#!/bin/bash
# start-llama-server.sh
set -e

MODEL_PATH="/models/${ORPHEUS_MODEL_NAME:-Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf}"
MAX_TOKENS="${ORPHEUS_MAX_TOKENS:-8192}"

echo "========================================="
echo " ARIA — llama.cpp server per Orpheus TTS"
echo "========================================="

if [ ! -f "$MODEL_PATH" ]; then
    echo ""
    echo "❌ ERRORE: Modello non trovato in $MODEL_PATH"
    echo ""
    echo "   Esegui prima su Windows:"
    echo "   aria-download.bat"
    echo ""
    echo "   Verifica anche che C:\\models\\orpheus sia"
    echo "   condiviso in Docker Desktop → Settings →"
    echo "   Resources → File Sharing"
    echo ""
    exit 1
fi

FILE_SIZE=$(stat -c%s "$MODEL_PATH" 2>/dev/null || echo "0")
echo "✅ Modello: $MODEL_PATH"
echo "   Dimensione: $(( FILE_SIZE / 1024 / 1024 )) MB"
echo "   Max tokens: $MAX_TOKENS"
echo ""
echo "🚀 Avvio llama-server..."
echo "   (Il caricamento richiede 30-90s — attendere)"
echo ""

exec llama-server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 5006 \
    --ctx-size "$MAX_TOKENS" \
    --n-predict "$MAX_TOKENS" \
    --rope-scaling linear \
    --n-gpu-layers 99 \
    --flash-attn \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
