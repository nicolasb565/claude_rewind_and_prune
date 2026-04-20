#!/bin/bash
# Fan out benchmark tasks into parallel benchmark-runner containers.
#
# Usage:
#   bash benchmarks/run.sh [options]
#
# Options:
#   --runs N               repeat each task N times (default 1)
#   --proxy on|off|bare    on = proxy injects clear_tool_uses_20250919
#                                (INJECT_CLEAR_TOOL_USES=1, COMPACT_ENABLED=0);
#                          off = proxy observes only (both disabled) — the
#                                cache_stats baseline;
#                          bare = no proxy at all.
#                          Default: off.
#   --auth subscription|env
#                          credential source (default subscription).
#                          subscription: mounts ~/.claude/.credentials.json
#                          env: loads benchmarks/.env (ANTHROPIC_API_KEY=...)
#   --tasks a,b,c          filter to a subset (default: all)
#   --concurrency N        max parallel containers (default 6)
#   --mode run|surrogate   run = real claude -p; surrogate = plumbing-only
#                          (default run)
#   --bookmarks on|off     expose the bookmark MCP server to the agent
#                          (default off). Activates mark/recall/list tools.
#   --rewind on|off        enable summarize_and_forget: MCP tool + proxy
#                          elision on outgoing requests (default off).
#                          Implies --bookmarks on (same MCP server hosts it).
#   --rewind-hint on|off   append a prompt hint telling the agent to use
#                          summarize_and_forget on dead-ends (default off).
#                          Answers "does it help when told to?" vs
#                          "does it find the tool itself?".
#   --run-id NAME          label results dir (default auto-increment run_NNN)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$SCRIPT_DIR/manifest.json"
FIXTURES_DIR="$SCRIPT_DIR/fixtures"
RESULTS_DIR="$SCRIPT_DIR/results"
IMAGE="benchmark-runner:latest"
PROXY_SCRIPT="$REPO_DIR/proxy/proxy.mjs"

RUNS=1
PROXY="off"
AUTH="subscription"
TASK_FILTER=""
CONCURRENCY=6
MODE="run"
RUN_ID=""
BOOKMARKS="off"
REWIND="off"
REWIND_HINT="off"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs)         RUNS="$2"; shift 2 ;;
    --proxy)        PROXY="$2"; shift 2 ;;
    --auth)         AUTH="$2"; shift 2 ;;
    --tasks)        TASK_FILTER="$2"; shift 2 ;;
    --concurrency)  CONCURRENCY="$2"; shift 2 ;;
    --mode)         MODE="$2"; shift 2 ;;
    --bookmarks)    BOOKMARKS="$2"; shift 2 ;;
    --rewind)       REWIND="$2"; shift 2 ;;
    --rewind-hint)  REWIND_HINT="$2"; shift 2 ;;
    --run-id)       RUN_ID="$2"; shift 2 ;;
    --manifest)     MANIFEST="$2"; shift 2 ;;
    -h|--help)      sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# --rewind requires the proxy to be able to mutate outgoing requests AND
# the bookmarks MCP server to expose the tool.
if [ "$REWIND" = "on" ]; then
  BOOKMARKS="on"
fi

# Resolve manifest to absolute path so docker mount works regardless of cwd
[[ -f "$MANIFEST" ]] || { echo "--manifest not found: $MANIFEST"; exit 1; }
MANIFEST="$(cd "$(dirname "$MANIFEST")" && pwd)/$(basename "$MANIFEST")"

[[ "$PROXY" == "on" || "$PROXY" == "off" || "$PROXY" == "bare" ]] \
  || { echo "--proxy must be on|off|bare"; exit 1; }
[[ "$BOOKMARKS" == "on" || "$BOOKMARKS" == "off" ]] \
  || { echo "--bookmarks must be on|off"; exit 1; }
[[ "$REWIND" == "on" || "$REWIND" == "off" ]] \
  || { echo "--rewind must be on|off"; exit 1; }
[[ "$REWIND_HINT" == "on" || "$REWIND_HINT" == "off" ]] \
  || { echo "--rewind-hint must be on|off"; exit 1; }
if [ "$REWIND" = "on" ] && [ "$PROXY" = "bare" ]; then
  echo "--rewind=on is incompatible with --proxy=bare (rewind requires the proxy)"; exit 1
fi
[[ "$AUTH" == "subscription" || "$AUTH" == "env" ]] || { echo "--auth must be subscription|env"; exit 1; }
[[ "$MODE" == "run" || "$MODE" == "surrogate" ]] || { echo "--mode must be run|surrogate"; exit 1; }

command -v docker >/dev/null || { echo "docker not found"; exit 1; }
command -v jq >/dev/null     || { echo "jq not found";     exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "image $IMAGE not found — run: docker build -t $IMAGE -f $SCRIPT_DIR/Dockerfile $REPO_DIR"; exit 1; }

# ── Run ID ─────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"
if [ -z "$RUN_ID" ]; then
  N=1
  while [ -d "$RESULTS_DIR/run_$(printf '%03d' "$N")" ]; do
    N=$((N + 1))
  done
  RUN_ID="run_$(printf '%03d' "$N")"
fi
RUN_DIR="$RESULTS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"
cp "$MANIFEST" "$RUN_DIR/manifest_snapshot.json"
LOG="$RUN_DIR/run.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "run_id=$RUN_ID proxy=$PROXY bookmarks=$BOOKMARKS rewind=$REWIND rewind_hint=$REWIND_HINT auth=$AUTH mode=$MODE runs=$RUNS concurrency=$CONCURRENCY"

# ── Auth mount ─────────────────────────────────────────────────────────────
AUTH_MOUNTS=()
# Container runs as host UID with HOME=/tmp, so auth files must be mounted
# under /tmp (matching $HOME inside the container) to be found by `claude`.
case "$AUTH" in
  subscription)
    CRED="$HOME/.claude/.credentials.json"
    [ -f "$CRED" ] || { log "ERROR: $CRED not found — log in with 'claude' first, or use --auth env"; exit 1; }
    AUTH_MOUNTS+=(-v "$CRED:/tmp/.claude/.credentials.json:ro")
    [ -f "$HOME/.claude.json" ] && AUTH_MOUNTS+=(-v "$HOME/.claude.json:/tmp/.claude.json:ro")
    ;;
  env)
    ENV_FILE="$SCRIPT_DIR/.env"
    [ -f "$ENV_FILE" ] || { log "ERROR: $ENV_FILE not found"; exit 1; }
    AUTH_MOUNTS+=(--env-file "$ENV_FILE")
    ;;
esac

# ── Proxy start ────────────────────────────────────────────────────────────
PROXY_PID=""
PROXY_LOG_DIR="$RUN_DIR/proxy_logs"
start_proxy() {
  # Args: <compact_enabled> <inject_clear_tool_uses> <rewind_enabled>
  # Whichever primitives are turned on for this run.
  local compact="${1:-0}"
  local inject="${2:-0}"
  local rewind="${3:-0}"
  [ -f "$PROXY_SCRIPT" ] || { log "ERROR: $PROXY_SCRIPT not found"; exit 1; }
  # Kill anything already on :8080
  if command -v lsof >/dev/null && lsof -ti :8080 >/dev/null 2>&1; then
    log "killing process already on :8080"
    lsof -ti :8080 | xargs -r kill 2>/dev/null || true
    sleep 1
  fi
  mkdir -p "$PROXY_LOG_DIR"
  log "starting proxy (COMPACT_ENABLED=$compact INJECT_CLEAR_TOOL_USES=$inject REWIND_ENABLED=$rewind LOG_DIR=$PROXY_LOG_DIR)"
  # The proxy always emits `cache_stats` parsed from upstream responses
  # and `request_summary` per outgoing request — that is the A/B
  # baseline regardless of which hygiene primitives are active.
  ( cd "$REPO_DIR" && LOG_DIR="$PROXY_LOG_DIR" \
      COMPACT_ENABLED="$compact" INJECT_CLEAR_TOOL_USES="$inject" \
      REWIND_ENABLED="$rewind" \
      node "$PROXY_SCRIPT" >"$RUN_DIR/proxy.log" 2>&1 ) &
  PROXY_PID=$!
  sleep 2
  kill -0 "$PROXY_PID" 2>/dev/null || { log "ERROR: proxy failed to start"; cat "$RUN_DIR/proxy.log"; exit 1; }
  log "proxy pid=$PROXY_PID"
}
stop_proxy() {
  if [ -n "$PROXY_PID" ] && kill -0 "$PROXY_PID" 2>/dev/null; then
    log "stopping proxy (pid=$PROXY_PID)"
    kill "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
  fi
}
trap stop_proxy EXIT

case "$PROXY" in
  # on  = proxy observes + injects clear_tool_uses (native API clearing).
  # off = proxy observes only (cache_stats baseline).
  # bare = no proxy at all.
  # Rewind is orthogonal and is enabled independently via --rewind.
  on)   start_proxy 0 1 $([ "$REWIND" = "on" ] && echo 1 || echo 0) ;;
  off)  start_proxy 0 0 $([ "$REWIND" = "on" ] && echo 1 || echo 0) ;;
  bare) ;;
esac

# ── Task list ──────────────────────────────────────────────────────────────
if [ -n "$TASK_FILTER" ]; then
  IFS=',' read -ra TASKS <<<"$TASK_FILTER"
else
  mapfile -t TASKS < <(jq -r '.tasks[].id' "$MANIFEST")
fi
log "tasks: ${TASKS[*]}"

# ── Launch ─────────────────────────────────────────────────────────────────
# Semaphore-style concurrency using a fifo of tokens.
SEM=$(mktemp -u); mkfifo "$SEM"; exec 3<>"$SEM"; rm "$SEM"
for ((i=0; i<CONCURRENCY; i++)); do echo "tok" >&3; done

launch_one() {
  local TASK_ID="$1" RUN_IDX="$2"
  local SUFFIX="_$RUN_IDX"
  local TASK_OUT="$RUN_DIR/$TASK_ID"
  local FIX_DIR="$FIXTURES_DIR/$TASK_ID"
  mkdir -p "$TASK_OUT"

  if [ ! -d "$FIX_DIR" ]; then
    log "$TASK_ID: fixture missing — run setup.sh first"
    echo '{"error": "fixture missing"}' > "$TASK_OUT/summary${SUFFIX}.json"
    return 1
  fi

  local CONTAINER="bench_${RUN_ID}_${TASK_ID}${SUFFIX}"
  local PROXY_ENV=()
  if [ "$PROXY" = "on" ] || [ "$PROXY" = "off" ]; then
    PROXY_ENV+=(-e "ANTHROPIC_BASE_URL=http://localhost:8080")
  fi
  if [ "$BOOKMARKS" = "on" ]; then
    PROXY_ENV+=(-e "ENABLE_BOOKMARKS=1")
  fi
  if [ "$REWIND" = "on" ]; then
    PROXY_ENV+=(-e "ENABLE_REWIND=1")
  fi
  if [ "$REWIND_HINT" = "on" ]; then
    PROXY_ENV+=(-e "REWIND_HINT=1")
  fi

  log "$TASK_ID${SUFFIX}: launching"
  # --memory caps each container so a runaway build can't OOM the host.
  # Override via BENCH_MEM_LIMIT env; unset means no cap.
  local mem_args=()
  if [ -n "${BENCH_MEM_LIMIT:-}" ]; then
    mem_args=(--memory "$BENCH_MEM_LIMIT" --memory-swap "$BENCH_MEM_LIMIT")
  fi
  docker run --rm \
    --name "$CONTAINER" \
    --network host \
    --user "$(id -u):$(id -g)" \
    "${mem_args[@]}" \
    -e HOME=/tmp \
    -e "RUN_SUFFIX=$SUFFIX" \
    "${PROXY_ENV[@]}" \
    "${AUTH_MOUNTS[@]}" \
    -v "$MANIFEST:/manifest.json:ro" \
    -v "$SCRIPT_DIR/tasks:/tasks:ro" \
    -v "$FIX_DIR:/work:ro" \
    -v "$TASK_OUT:/output" \
    "$IMAGE" "$MODE" "$TASK_ID" \
    >>"$TASK_OUT/docker${SUFFIX}.log" 2>&1
  local rc=$?
  log "$TASK_ID${SUFFIX}: container exit=$rc"
  return $rc
}

PIDS=()
for RUN_IDX in $(seq 1 "$RUNS"); do
  for TASK_ID in "${TASKS[@]}"; do
    read -u 3 _token
    (
      launch_one "$TASK_ID" "$RUN_IDX"
      echo "tok" >&3
    ) &
    PIDS+=($!)
  done
done

for pid in "${PIDS[@]}"; do
  wait "$pid" || true
done

log "all tasks finished"

# ── Summary ────────────────────────────────────────────────────────────────
log "--- per-task summary ---"
for TASK_ID in "${TASKS[@]}"; do
  for f in "$RUN_DIR/$TASK_ID"/summary_*.json; do
    [ -f "$f" ] || continue
    DUR=$(jq -r '.duration_seconds // "?"' "$f" 2>/dev/null || echo "?")
    EX=$(jq -r '.exit_code // "?"' "$f" 2>/dev/null || echo "?")
    log "  $TASK_ID $(basename "$f"): dur=${DUR}s exit=$EX"
  done
done

log "results: $RUN_DIR"
