#!/bin/bash
set -e

MODEL_PATH="/models/${ORPHEUS_MODEL_NAME}"

# Verifica che il modello esista
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ ERRORE: Modello non trovato in $MODEL_PATH"
    echo "Scarica il modello prima di avviare il container."
    echo "Comando: aria-download.bat"
    exit 1
fi

echo "✅ Modello trovato: $MODEL_PATH"
echo "🚀 Avvio llama-server per Orpheus..."

exec /llama.cpp/build/bin/llama-server \
    -m "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 5006 \
    --ctx-size "${ORPHEUS_MAX_TOKENS:-8192}" \
    --n-predict "${ORPHEUS_MAX_TOKENS:-8192}" \
    --rope-scaling linear \
    -ngl 99 \
    --flash-attn \
    -c "${ORPHEUS_MAX_TOKENS:-8192}" \
    --cache-type-k q8_0 \
    --cache-type-v q8_0