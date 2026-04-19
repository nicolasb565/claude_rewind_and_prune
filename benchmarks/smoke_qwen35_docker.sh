#!/usr/bin/env bash
# Smoke test: Qwen 3.5 inference inside rocm/pytorch to avoid host kernel gaps.
# Default model: Qwen/Qwen3.5-4B (bf16, ~8.7 GB, fits 16 GB with room to spare).
#
# Why Docker: host torch-rocm has had kernel issues with Qwen 3.5 hybrid
# architecture. rocm/pytorch:latest ships AMD-validated torch+rocm.
#
# Usage:
#   benchmarks/smoke_qwen35_docker.sh
#   SMOKE_MODEL=Qwen/Qwen3.5-9B benchmarks/smoke_qwen35_docker.sh
#   SMOKE_PROMPT="write a haiku" benchmarks/smoke_qwen35_docker.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen3.5-4B}"

VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)

IMAGE="rocm/pytorch:latest"

# Qwen 3.5 hybrid Mamba/Transformer needs flash-linear-attention for
# Triton-AMD SSM kernels during training; inference with transformers
# alone tends to work via the SDPA/native path but fla is cheap insurance.
# causal-conv1d is for fla's fast path on SSM layers; tolerate build failure
# since the torch fallback already works (~34 t/s observed).
PIP_INSTALL='pip install --quiet --no-input "transformers>=5.3,<6" "accelerate>=1.10" "flash-linear-attention" && (pip install --quiet --no-input "causal-conv1d" || echo "[warn] causal-conv1d install failed; fla will use torch fallback")'

SMOKE_SCRIPT="${SMOKE_SCRIPT:-benchmarks/smoke_qwen35_inference.py}"

echo "=== Qwen 3.5 ROCm inference smoke ==="
echo "  image:     $IMAGE"
echo "  model:     $SMOKE_MODEL"
echo "  repo:      $REPO_DIR"
echo "  hf_cache:  $HF_CACHE"
echo "  video gid: $VIDEO_GID  render gid: $RENDER_GID"
echo

docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add "$VIDEO_GID" --group-add "$RENDER_GID" \
    --security-opt seccomp=unconfined \
    --shm-size=8g \
    -v "$REPO_DIR":/workspace \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -w /workspace \
    -e HF_HOME=/root/.cache/huggingface \
    -e SMOKE_MODEL="$SMOKE_MODEL" \
    ${SMOKE_PROMPT:+-e "SMOKE_PROMPT=$SMOKE_PROMPT"} \
    ${SMOKE_MAX_NEW:+-e "SMOKE_MAX_NEW=$SMOKE_MAX_NEW"} \
    -e TRANSFORMERS_VERBOSITY=warning \
    -e PYTHONUNBUFFERED=1 \
    "$IMAGE" \
    bash -c "$PIP_INSTALL && python -u $SMOKE_SCRIPT"
