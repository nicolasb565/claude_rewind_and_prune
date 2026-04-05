#!/usr/bin/env bash
# Run a single benchmark trial.
# Usage: ./benchmark/run_trial.sh <task> <condition> <trial>
#   task:      rate_limiter | async_refactor
#   condition: stock | compact | full
#   trial:     1, 2, 3, ...

set -euo pipefail

TASK="${1:?Usage: run_trial.sh <task> <condition> <trial>}"
CONDITION="${2:?Usage: run_trial.sh <task> <condition> <trial>}"
TRIAL="${3:?Usage: run_trial.sh <task> <condition> <trial>}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TASKS_DIR="/tmp/rewind-tasks"
RESULTS_DIR="$REPO_ROOT/results"
RUNS_DIR="/tmp/rewind-runs"

mkdir -p "$RESULTS_DIR" "$RUNS_DIR"

WORK_DIR="$RUNS_DIR/${TASK}_${CONDITION}_t${TRIAL}"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cp -r "$TASKS_DIR/$TASK/"* "$WORK_DIR/"

# Read task config
TASK_DESC=$(grep -A 20 'description:' "$TASKS_DIR/$TASK/task.yaml" | sed '1d;/^[a-z]/,$d' | sed 's/^  //')
TEST_CMD=$(grep 'test_command:' "$TASKS_DIR/$TASK/task.yaml" | sed 's/test_command: *//' | tr -d '"')

# Pick binary and mode
case $CONDITION in
  stock)
    CLAUDE_BIN="$(which claude)"
    export CLAUDE_REWIND_MODE="off"
    ;;
  compact)
    CLAUDE_BIN="$REPO_ROOT/bin/claude"
    export CLAUDE_REWIND_MODE="compact_only"
    ;;
  full)
    CLAUDE_BIN="$REPO_ROOT/bin/claude"
    export CLAUDE_REWIND_MODE="full"
    ;;
  *)
    echo "Unknown condition: $CONDITION" >&2
    exit 1
    ;;
esac

echo "=== $TASK | $CONDITION | trial $TRIAL ==="
echo "Working dir: $WORK_DIR"
echo "Claude bin: $CLAUDE_BIN"
echo "Mode: ${CLAUDE_REWIND_MODE:-default}"
echo ""

PROMPT="You are working in $WORK_DIR. $TASK_DESC Run '$TEST_CMD' to verify your fix. All tests must pass (exit code 0)."

START_TIME=$(date +%s)

# Run claude in print mode with allowed tools
"$CLAUDE_BIN" -p "$PROMPT" \
  --allowedTools "Edit,Bash,Read,Grep,Glob,Write" \
  2>"$WORK_DIR/stderr.log" \
  >"$WORK_DIR/claude_output.txt" || true

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Run tests to verify
echo ""
echo "--- Running tests ---"
cd "$WORK_DIR"
TEST_OUTPUT=$(eval "$TEST_CMD" 2>&1) || true
TEST_EXIT=$?
echo "$TEST_OUTPUT"

SUCCESS="false"
[ "$TEST_EXIT" -eq 0 ] && SUCCESS="true"

# Count output lines from claude
CLAUDE_LINES=$(wc -l < "$WORK_DIR/claude_output.txt" 2>/dev/null | tr -d ' ' || echo 0)

# Check for rewind events in telemetry
LOGFILE=~/.claude-rewind-logs/events-$(date +%Y-%m-%d).jsonl
REWIND_COUNT=0
COMPACT_COUNT=0
if [ -f "$LOGFILE" ]; then
  REWIND_COUNT=$(grep -c '"type":"rewind"' "$LOGFILE" 2>/dev/null || echo 0)
  COMPACT_COUNT=$(grep -c '"type":"compact"' "$LOGFILE" 2>/dev/null || echo 0)
fi

echo ""
echo "=== RESULT: $TASK | $CONDITION | trial $TRIAL ==="
echo "  success:      $SUCCESS"
echo "  duration:     ${DURATION}s"
echo "  output_lines: $CLAUDE_LINES"
echo "  rewinds:      $REWIND_COUNT"
echo "  compactions:  $COMPACT_COUNT"

# Log result
cat >> "$RESULTS_DIR/benchmark_log.jsonl" << JSONEOF
{"task":"$TASK","condition":"$CONDITION","trial":$TRIAL,"success":$SUCCESS,"duration":$DURATION,"output_lines":$CLAUDE_LINES,"rewinds":$REWIND_COUNT,"compactions":$COMPACT_COUNT,"timestamp":"$(date -Iseconds)"}
JSONEOF

echo ""
echo "Result logged to $RESULTS_DIR/benchmark_log.jsonl"
