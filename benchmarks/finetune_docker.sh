#!/usr/bin/env bash
# Run the fine-tuning script inside the rocm/pytorch:latest container.
#
# Why Docker: host torch-rocm wheels (both 2.9.1+rocm6.3 and 2.11.0+rocm7.2)
# have gfx1201 kernel gaps that cause Phi-4-mini forward-pass segfaults during
# LoRA training on RX 9070 XT. The rocm/pytorch container ships a validated
# AMD-maintained torch+ROCm stack (torch 2.10.0+rocm7.2.2) that works.
#
# Usage:
#   benchmarks/finetune_docker.sh [--smoke]          # smoke test (500 examples)
#   benchmarks/finetune_docker.sh                    # full training
#   benchmarks/finetune_docker.sh --model Qwen/Qwen2.5-7B-Instruct  # different model
#
# Outputs land in proxy/experiments/phi4_mini_stuck_lora/ on the host.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

# AMDGPU device IDs for --group-add (numeric, container doesn't have the names)
VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)

IMAGE="rocm/pytorch:latest"

mkdir -p "$REPO_DIR/proxy/experiments/phi4_mini_stuck_lora"
mkdir -p "$REPO_DIR/data/generated"

# Build pip install command for additional packages. The base container has
# torch + basic ML stack but we need the transformers/peft/trl/datasets we
# use in the training script. Installing into the container's site-packages
# at runtime is simpler than building a custom image.
PIP_INSTALL='pip install --quiet --no-input "transformers>=5,<6" "peft>=0.17" "accelerate>=1.10" "datasets>=4.0"'

# Build the python command — same flags pass through.
PY_ARGS="benchmarks/finetune_train.py"
for arg in "$@"; do
    PY_ARGS="$PY_ARGS \"$arg\""
done

echo "=== Launching $IMAGE ==="
echo "  repo:     $REPO_DIR"
echo "  hf_cache: $HF_CACHE"
echo "  video gid: $VIDEO_GID"
echo "  render gid: $RENDER_GID"
echo "  args:     $PY_ARGS"
echo

docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add "$VIDEO_GID" --group-add "$RENDER_GID" \
    --security-opt seccomp=unconfined \
    --shm-size=8g \
    -v "$REPO_DIR":/workspace \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -v /tmp:/tmp \
    -w /workspace \
    -e HF_HOME=/root/.cache/huggingface \
    -e TRANSFORMERS_VERBOSITY=warning \
    -e PYTHONUNBUFFERED=1 \
    "$IMAGE" \
    bash -c "$PIP_INSTALL && python -u $PY_ARGS"
