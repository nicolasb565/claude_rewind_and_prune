#!/bin/bash
# Alternate off/on benchmark runs for A/B comparison.
# Usage: bash benchmarks/run_ab.sh [N] [--manifest path] [extra args passed to run.sh]
#   N defaults to 5 (runs 2*N total)
set -euo pipefail

N="${1:-5}"
# Validate N looks like a positive integer before shifting so a mistyped
# `bash run_ab.sh --manifest foo.json` fails loudly instead of feeding
# `--manifest` to `seq`.
if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -lt 1 ]]; then
  echo "first arg N must be a positive integer (got: $N)" >&2
  echo "usage: bash run_ab.sh [N] [--manifest path] [extra run.sh args]" >&2
  exit 2
fi
shift || true
EXTRA_ARGS=("$@")  # forwarded verbatim to run.sh (--manifest, --tasks, --concurrency, etc.)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$SCRIPT_DIR/results/ab_manifest.log"

echo "=== A/B benchmark: $N pairs (${N} off + ${N} on)  extra=${EXTRA_ARGS[*]:-none} ===" | tee "$LOG"
echo "started: $(date)" | tee -a "$LOG"

for i in $(seq 1 "$N"); do
  echo "" | tee -a "$LOG"
  echo "--- pair $i/$N: OFF ---" | tee -a "$LOG"
  bash "$SCRIPT_DIR/run.sh" --proxy off --runs 1 "${EXTRA_ARGS[@]}" 2>&1 | tail -15 | tee -a "$LOG"

  echo "" | tee -a "$LOG"
  echo "--- pair $i/$N: ON ---" | tee -a "$LOG"
  bash "$SCRIPT_DIR/run.sh" --proxy on --runs 1 "${EXTRA_ARGS[@]}" 2>&1 | tail -15 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "finished: $(date)" | tee -a "$LOG"
echo "=== done ===" | tee -a "$LOG"
