#!/usr/bin/env bash
# Smoke test: run benchmarks/smoke_gemma4.py inside rocm/pytorch:latest
# to verify Gemma 4 loads + takes one LoRA step on the RX 9070 XT.
#
# Why Docker: host torch-rocm wheels have gfx1201 kernel gaps. The
# rocm/pytorch container ships AMD-validated torch 2.10.0+rocm7.2.2.
#
# Usage:
#   benchmarks/smoke_gemma4_docker.sh                          # default: gemma-4-4b-it
#   SMOKE_MODEL=google/gemma-4-2b-it benchmarks/smoke_gemma4_docker.sh
#   SMOKE_MODEL=google/gemma-4-e4b-it benchmarks/smoke_gemma4_docker.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
SMOKE_MODEL="${SMOKE_MODEL:-google/gemma-4-E2B-it}"

# Auto-source HF_TOKEN from repo-root .env if not already in env.
if [ -z "${HF_TOKEN:-}" ] && [ -f "$REPO_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_DIR/.env"
  set +a
fi

VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)

IMAGE="rocm/pytorch:latest"

PIP_INSTALL='pip install --quiet --no-input "transformers>=5,<6" "peft>=0.17" "accelerate>=1.10"'

echo "=== Gemma 4 ROCm smoke test ==="
echo "  image:     $IMAGE"
echo "  model:     $SMOKE_MODEL"
echo "  repo:      $REPO_DIR"
echo "  hf_cache:  $HF_CACHE"
echo "  video gid: $VIDEO_GID  render gid: $RENDER_GID"

# Gemma models on HF are gated. Detect token availability up front and
# fail fast with a helpful message if the user isn't authenticated.
HF_TOKEN_ARG=()
if [ -n "${HF_TOKEN:-}" ]; then
  HF_TOKEN_ARG=(-e "HF_TOKEN=$HF_TOKEN")
  echo "  hf auth:   HF_TOKEN env var"
elif [ -f "$HOME/.cache/huggingface/token" ]; then
  TOKEN_FILE_VAL=$(head -c 200 "$HOME/.cache/huggingface/token")
  HF_TOKEN_ARG=(-e "HF_TOKEN=$TOKEN_FILE_VAL")
  echo "  hf auth:   ~/.cache/huggingface/token"
else
  echo "  hf auth:   NONE DETECTED — Gemma models are gated."
  echo "             Accept license at https://huggingface.co/$SMOKE_MODEL"
  echo "             then run: huggingface-cli login (or export HF_TOKEN=...)"
  echo
  echo "  Proceeding anyway — the download will 401 if auth is missing."
fi
echo

# SMOKE_MODE selects which script runs:
#   training   — forward+backward via smoke_gemma4.py (default)
#   inference  — generate() via smoke_gemma4_inference.py
SMOKE_MODE="${SMOKE_MODE:-training}"
case "$SMOKE_MODE" in
  training)  SMOKE_SCRIPT="benchmarks/smoke_gemma4.py" ;;
  inference) SMOKE_SCRIPT="benchmarks/smoke_gemma4_inference.py" ;;
  *) echo "ERROR: SMOKE_MODE must be 'training' or 'inference' (got '$SMOKE_MODE')" >&2; exit 1 ;;
esac
echo "  mode:      $SMOKE_MODE  ($SMOKE_SCRIPT)"
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
    "${HF_TOKEN_ARG[@]}" \
    "$IMAGE" \
    bash -c "$PIP_INSTALL && python -u $SMOKE_SCRIPT"
