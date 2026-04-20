#!/usr/bin/env bash
# Run the Gemma 4 hygiene LoRA training inside rocm/pytorch:latest.
#
# Same container stack as the Gemma 4 smoke (torch 2.10.0+rocm7.2.2).
# Mounts the repo + HF cache + output dir. Forwards HF_TOKEN from .env.
#
# Usage:
#   benchmarks/train_gemma4_hygiene_docker.sh [args]
#   benchmarks/train_gemma4_hygiene_docker.sh --smoke     # plumbing check
#
# Output LoRA adapter lands at proxy/experiments/gemma4_hygiene_v1/final/.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)

IMAGE="rocm/pytorch:latest"

# HF auth — source from .env if not already in env.
if [ -z "${HF_TOKEN:-}" ] && [ -f "$REPO_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_DIR/.env"
  set +a
fi
HF_TOKEN_ARG=()
if [ -n "${HF_TOKEN:-}" ]; then
  HF_TOKEN_ARG=(-e "HF_TOKEN=$HF_TOKEN")
fi

PIP_INSTALL='pip install --quiet --no-input "transformers>=5,<6" "peft>=0.17" "accelerate>=1.10" "trl>=0.20" "datasets>=4.0" "liger-kernel>=0.5.0" flash-linear-attention'

# Assemble python args verbatim
PY_ARGS="benchmarks/train_gemma4_hygiene.py"
for arg in "$@"; do
    PY_ARGS="$PY_ARGS \"$arg\""
done

echo "=== Gemma 4 hygiene LoRA training ==="
echo "  image:     $IMAGE"
echo "  repo:      $REPO_DIR"
echo "  hf_cache:  $HF_CACHE"
echo "  video gid: $VIDEO_GID  render gid: $RENDER_GID"
echo "  args:      $PY_ARGS"
echo

mkdir -p "$REPO_DIR/proxy/experiments/gemma4_hygiene_v1"

docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add "$VIDEO_GID" --group-add "$RENDER_GID" \
    --security-opt seccomp=unconfined \
    --shm-size=8g \
    -v "$REPO_DIR":/workspace \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -w /workspace \
    -e HF_HOME=/root/.cache/huggingface \
    -e TRANSFORMERS_VERBOSITY=warning \
    -e PYTHONUNBUFFERED=1 \
    -e HIP_VISIBLE_DEVICES=0 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e ROCR_VISIBLE_DEVICES=0 \
    -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    "${HF_TOKEN_ARG[@]}" \
    "$IMAGE" \
    bash -c "$PIP_INSTALL && python -u $PY_ARGS"
