#!/usr/bin/env bash
# Run harness/run_shadow.py inside rocm/pytorch:latest.
#
# Default: one session of bug_01_offbyone with Qwen/Qwen3.5-4B.
# Overrides via env vars:
#   FIXTURE=...       fixture name under harness/fixtures/
#   SMOKE_MODEL=...   model id
#   MAX_STEPS=...

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
FIXTURE="${FIXTURE:-bug_01_offbyone}"
SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen3.5-4B}"
MAX_STEPS="${MAX_STEPS:-30}"
ACT_ON_SHADOW="${ACT_ON_SHADOW:-0}"

VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)

IMAGE="rocm/pytorch:latest"

PIP_INSTALL='pip install --quiet --no-input "transformers>=5.3,<6" "accelerate>=1.10" "flash-linear-attention"'
# Install Node.js for JS fixtures (cheap apt install, no-op if already present)
NODE_INSTALL='command -v node >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq nodejs >/dev/null 2>&1)'

ACT_FLAG=""
if [ "$ACT_ON_SHADOW" = "1" ] || [ "$ACT_ON_SHADOW" = "true" ]; then
    ACT_FLAG="--act-on-shadow"
fi

echo "=== Qwen 3.5 shadow-log run ==="
echo "  image:     $IMAGE"
echo "  model:     $SMOKE_MODEL"
echo "  fixture:   $FIXTURE"
echo "  max_steps: $MAX_STEPS"
echo "  act_rewind:$ACT_ON_SHADOW"
echo "  repo:      $REPO_DIR"

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
    "$IMAGE" \
    bash -c "$NODE_INSTALL && $PIP_INSTALL && python -u harness/run_shadow.py --fixture \"$FIXTURE\" --model \"$SMOKE_MODEL\" --max-steps \"$MAX_STEPS\" $ACT_FLAG"
