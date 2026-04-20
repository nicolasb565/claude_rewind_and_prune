#!/usr/bin/env bash
# Download an upstream-compatible Gemma 4 E4B-it GGUF from HuggingFace
# and run a one-shot inference against it with the local llama.cpp build
# (which has HIP enabled for gfx1201).
#
# Usage:
#   benchmarks/smoke_gemma4_llama.sh                    # default Q4_K_M
#   GEMMA_QUANT=Q5_K_M benchmarks/smoke_gemma4_llama.sh # pick a different quant
#   GEMMA_QUANT=BF16   benchmarks/smoke_gemma4_llama.sh # full precision (15 GB)
set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-$HOME/source/llama.cpp/build/bin/llama-cli}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
QUANT="${GEMMA_QUANT:-Q4_K_M}"
REPO="unsloth/gemma-4-E4B-it-GGUF"
FILENAME="gemma-4-E4B-it-${QUANT}.gguf"
LOCAL="${MODELS_DIR}/${FILENAME}"
NGL="${NGL:-99}"   # offload all layers to GPU

mkdir -p "$MODELS_DIR"

if [ ! -f "$LOCAL" ]; then
  echo "=== Downloading $FILENAME from $REPO ==="
  # Prefer huggingface-cli if available; fall back to curl against the raw URL.
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$REPO" "$FILENAME" --local-dir "$MODELS_DIR" --local-dir-use-symlinks False
  elif command -v hf >/dev/null 2>&1; then
    hf download "$REPO" "$FILENAME" --local-dir "$MODELS_DIR"
  else
    URL="https://huggingface.co/${REPO}/resolve/main/${FILENAME}"
    echo "  using curl → $URL"
    curl -L --fail -o "$LOCAL" "$URL"
  fi
fi

[ -f "$LOCAL" ] || { echo "ERROR: $LOCAL not present after download"; exit 1; }

echo
echo "=== Model info ==="
ls -lh "$LOCAL"
echo

echo "=== Running inference (--n-gpu-layers $NGL) ==="
"$LLAMA_BIN" \
  -m "$LOCAL" \
  -ngl "$NGL" \
  -p "Reply with exactly: ACK" \
  -n 8 \
  --no-warmup 2>&1 | tail -40
