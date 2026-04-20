#!/usr/bin/env bash
# Run benchmarks/test_adapter.py inside rocm/pytorch:latest.
# Compares base vs base+adapter generations on one training chunk.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)
IMAGE="rocm/pytorch:latest"

if [ -z "${HF_TOKEN:-}" ] && [ -f "$REPO_DIR/.env" ]; then
  set -a; source "$REPO_DIR/.env"; set +a
fi
HF_TOKEN_ARG=()
[ -n "${HF_TOKEN:-}" ] && HF_TOKEN_ARG=(-e "HF_TOKEN=$HF_TOKEN")

PIP_INSTALL='pip install --quiet --no-input "transformers>=5,<6" "peft>=0.17" "accelerate>=1.10"'
PY_ARGS="benchmarks/test_adapter.py"
for arg in "$@"; do PY_ARGS="$PY_ARGS \"$arg\""; done

docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add "$VIDEO_GID" --group-add "$RENDER_GID" \
    --security-opt seccomp=unconfined --shm-size=8g \
    -v "$REPO_DIR":/workspace \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -w /workspace \
    -e HF_HOME=/root/.cache/huggingface \
    -e TRANSFORMERS_VERBOSITY=warning -e PYTHONUNBUFFERED=1 \
    -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0 \
    "${HF_TOKEN_ARG[@]}" \
    "$IMAGE" \
    bash -c "$PIP_INSTALL && python -u $PY_ARGS"
