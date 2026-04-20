#!/bin/bash
# Benchmark container entrypoint.
#
# Modes:
#   run <task_id>      : copy fixture -> /scratch, run claude -p, emit summary
#   compile <task_id>  : build the fixture in-place (mounted RW at /work)
#   surrogate <task_id>: same plumbing as "run" but substitute claude with a
#                        trivial ls+make check (for sanity testing without
#                        burning API credits)
#   shell              : drop into bash for debugging
#
# Mounts expected:
#   /manifest.json       (ro)   — the benchmark manifest
#   /tasks               (ro)   — benchmarks/tasks source dir
#   /work                (ro|rw) — per-task fixture (ro for run, rw for compile)
#   /output              (rw)   — per-task results dir
#
# Env:
#   RUN_SUFFIX           — suffix appended to summary/log filenames (e.g. "_1")
#   ANTHROPIC_BASE_URL   — proxy URL if --proxy on
#   (credentials mounted or passed via env by run.sh)

set -uo pipefail

MODE="${1:-run}"
TASK_ID="${2:-}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

fail() { echo "entrypoint: $*" >&2; exit 1; }

case "$MODE" in
  shell)
    exec bash
    ;;

  compile)
    [ -n "$TASK_ID" ] || fail "compile: task_id required"
    [ -f /manifest.json ] || fail "compile: /manifest.json not mounted"
    [ -d /work ] || fail "compile: /work not mounted"

    BUILD_CMD=$(jq -r --arg id "$TASK_ID" \
      '.tasks[] | select(.id==$id) | .fixture.build_cmd // empty' \
      /manifest.json)

    cd /work
    git config --global --add safe.directory /work 2>/dev/null || true

    if [ -d .git ] || [ -f .git ]; then
      echo "compile[$TASK_ID]: resetting repo state"
      git reset --hard HEAD 2>&1 || true
      git clean -fdx 2>&1 || true
    fi

    if [ -z "$BUILD_CMD" ] || [ "$BUILD_CMD" = "null" ]; then
      echo "compile[$TASK_ID]: no build_cmd — skipping"
      exit 0
    fi

    echo "compile[$TASK_ID]: running: $BUILD_CMD"
    bash -c "$BUILD_CMD"
    EXIT=$?
    echo "compile[$TASK_ID]: exit=$EXIT"
    exit $EXIT
    ;;

  run|surrogate)
    [ -n "$TASK_ID" ] || fail "$MODE: task_id required"
    [ -f /manifest.json ] || fail "$MODE: /manifest.json not mounted"
    [ -d /work ] || fail "$MODE: /work not mounted"
    [ -d /output ] || fail "$MODE: /output not mounted"
    [ -f "/tasks/$TASK_ID/task.md" ] || fail "$MODE: /tasks/$TASK_ID/task.md missing"

    DEFAULT_MODEL=$(jq -r '.default_model' /manifest.json)
    DEFAULT_MAX_TURNS=$(jq -r '.default_max_turns' /manifest.json)
    MODEL=$(jq -r --arg id "$TASK_ID" --arg d "$DEFAULT_MODEL" \
      '.tasks[] | select(.id==$id) | .model // $d' /manifest.json)
    MAX_TURNS=$(jq -r --arg id "$TASK_ID" --arg d "$DEFAULT_MAX_TURNS" \
      '.tasks[] | select(.id==$id) | .max_turns // $d' /manifest.json)

    # Materialize a writable copy of the fixture
    mkdir -p /scratch
    echo "$MODE[$TASK_ID]: cp -a /work/. /scratch/"
    cp -a /work/. /scratch/
    cd /scratch

    SUM="/output/summary${RUN_SUFFIX}.json"
    TRANSCRIPT="/output/transcript${RUN_SUFFIX}.jsonl"
    STDERR="/output/stderr${RUN_SUFFIX}.log"
    SURROGATE_LOG="/output/surrogate${RUN_SUFFIX}.log"
    VERIFY_JSON="/output/verify${RUN_SUFFIX}.json"
    VERIFY_LOG="/output/verify${RUN_SUFFIX}.log"
    PATCH="/output/patch${RUN_SUFFIX}.diff"
    PATCH_STAT="/output/patch${RUN_SUFFIX}.stat"

    # Baseline git commit so "the agent's changes" can be diffed cleanly.
    # Build artifacts from setup.sh are captured in this baseline and won't
    # pollute the patch. Skips silently for non-git fixtures.
    #
    # safe.directory='*' is required because the container UID (1000) has no
    # /etc/passwd entry, so git can't resolve the directory owner and refuses
    # with "dubious ownership" unless we explicitly whitelist.
    HAS_GIT=0
    GIT_SAFE=(-c safe.directory=*)
    if [ -e /scratch/.git ]; then
      HAS_GIT=1
      git "${GIT_SAFE[@]}" -C /scratch add -A 2>/dev/null || true
      git "${GIT_SAFE[@]}" -C /scratch \
        -c user.name=bench -c user.email=bench@local \
        commit -m "bench baseline" --quiet --allow-empty 2>/dev/null || true
    fi

    PROMPT=$(cat "/tasks/$TASK_ID/task.md")
    START=$(date +%s)

    # ── Optional: bookmark MCP server (agent-memory probe) ────────────
    # When ENABLE_BOOKMARKS=1, drop a .mcp.json in cwd referencing the
    # server baked into the image and extend the allowedTools whitelist.
    # Without both, Claude Code won't discover or invoke the tools.
    ALLOWED_TOOLS="Bash,Edit,Write,Read,Grep,Glob"
    if [ "${ENABLE_BOOKMARKS:-0}" = "1" ]; then
      # Per-run-idx dir so concurrent runs on the same task_id don't
      # interleave their bookmark logs in the same file.
      BM_DIR="/output/bookmark_logs${RUN_SUFFIX}"
      mkdir -p "$BM_DIR"
      cat >/scratch/.mcp.json <<MCP_JSON
{
  "mcpServers": {
    "bookmarks": {
      "command": "node",
      "args": ["/opt/mcp/bookmark_server.mjs"],
      "env": { "BOOKMARK_LOG_DIR": "$BM_DIR" }
    }
  }
}
MCP_JSON
      ALLOWED_TOOLS="$ALLOWED_TOOLS,mcp__bookmarks__bookmark_mark,mcp__bookmarks__bookmark_recall,mcp__bookmarks__bookmark_list"
      echo "run[$TASK_ID]: bookmark MCP enabled (log: $BM_DIR)"
    fi
    if [ "${ENABLE_REWIND:-0}" = "1" ]; then
      ALLOWED_TOOLS="$ALLOWED_TOOLS,mcp__bookmarks__checkpoint_progress"
      echo "run[$TASK_ID]: checkpoint_progress enabled"
    fi

    # Optional prompt hint — tests whether the agent uses the rewind
    # tool when told to vs when it has to discover it on its own.
    if [ "${REWIND_HINT:-0}" = "1" ]; then
      PROMPT="$PROMPT

[You have \`mcp__bookmarks__checkpoint_progress\` available. Call it ONLY
when you have concrete evidence of progress — either:
(a) milestone_achieved: a specific, observable event like 'bug reproduces
at file:line', 'test that was failing now passes', 'build now succeeds'.
(b) approach_eliminated: a specific hypothesis was ruled out by a specific
observation, e.g. 'applied patch X, bug still reproduces with same output'.

Do NOT checkpoint when you're still exploring and unsure. A checkpoint
that says 'I don't see a bug' or 'code looks correct' is worse than no
checkpoint — it entrenches a wrong conclusion in the compressed history.
If you cannot cite a specific tool output, test result, or build log as
evidence, keep exploring.]"
    fi

    if [ "$MODE" = "run" ]; then
      # --output-format stream-json streams one JSON event per line
      # (system init, user, assistant with tool_use/tool_result blocks,
      # and a final "type":"result" summary). That's our full transcript;
      # we parse the last line for usage/cost fields below.
      # stream-json requires --verbose.
      claude --dangerously-skip-permissions -p "$PROMPT" \
        --model "$MODEL" \
        --max-turns "$MAX_TURNS" \
        --allowedTools "$ALLOWED_TOOLS" \
        --output-format stream-json \
        --verbose \
        >"$TRANSCRIPT" 2>"$STDERR"
      EXIT=$?
    else
      # Surrogate: prove plumbing without an API call
      {
        echo "surrogate for $TASK_ID"
        echo "--- pwd ---"; pwd
        echo "--- ls /scratch (top) ---"; ls -la /scratch | head -20
        echo "--- claude --version ---"; claude --version || true
        echo "--- prompt (first 200 chars) ---"; head -c 200 "/tasks/$TASK_ID/task.md"; echo
      } >"$SURROGATE_LOG" 2>"$STDERR"
      : > "$TRANSCRIPT"  # empty transcript for surrogate
      EXIT=0
    fi

    END=$(date +%s)
    DURATION=$((END - START))

    # Parse Claude's result summary from the last line of the stream-json
    # transcript (type:"result"). Defaults to null if not found (so the
    # summary schema stays stable across modes and broken runs).
    INPUT_TOKENS=null
    OUTPUT_TOKENS=null
    CACHE_CREATION_TOKENS=null
    CACHE_READ_TOKENS=null
    TOTAL_COST_USD=null
    NUM_TURNS=null
    CLAUDE_DURATION_MS=null
    CLAUDE_API_MS=null
    STOP_REASON=null
    IS_ERROR=null
    if [ "$MODE" = "run" ] && [ -s "$TRANSCRIPT" ]; then
      RESULT_LINE=$(grep -F '"type":"result"' "$TRANSCRIPT" | tail -n 1)
      if [ -n "$RESULT_LINE" ] && echo "$RESULT_LINE" | jq -e . >/dev/null 2>&1; then
        INPUT_TOKENS=$(echo           "$RESULT_LINE" | jq '.usage.input_tokens                // null')
        OUTPUT_TOKENS=$(echo          "$RESULT_LINE" | jq '.usage.output_tokens               // null')
        CACHE_CREATION_TOKENS=$(echo  "$RESULT_LINE" | jq '.usage.cache_creation_input_tokens // null')
        CACHE_READ_TOKENS=$(echo      "$RESULT_LINE" | jq '.usage.cache_read_input_tokens     // null')
        TOTAL_COST_USD=$(echo         "$RESULT_LINE" | jq '.total_cost_usd                    // null')
        NUM_TURNS=$(echo              "$RESULT_LINE" | jq '.num_turns                         // null')
        CLAUDE_DURATION_MS=$(echo     "$RESULT_LINE" | jq '.duration_ms                       // null')
        CLAUDE_API_MS=$(echo          "$RESULT_LINE" | jq '.duration_api_ms                   // null')
        STOP_REASON=$(echo            "$RESULT_LINE" | jq '.stop_reason                       // null')
        IS_ERROR=$(echo               "$RESULT_LINE" | jq '.is_error                          // null')
      fi
    fi

    cat >"$SUM" <<EOF
{
  "task_id": "$TASK_ID",
  "mode": "$MODE",
  "model": "$MODEL",
  "max_turns": $MAX_TURNS,
  "duration_seconds": $DURATION,
  "exit_code": $EXIT,
  "run_suffix": "$RUN_SUFFIX",
  "input_tokens": $INPUT_TOKENS,
  "output_tokens": $OUTPUT_TOKENS,
  "cache_creation_input_tokens": $CACHE_CREATION_TOKENS,
  "cache_read_input_tokens": $CACHE_READ_TOKENS,
  "total_cost_usd": $TOTAL_COST_USD,
  "num_turns": $NUM_TURNS,
  "claude_duration_ms": $CLAUDE_DURATION_MS,
  "claude_duration_api_ms": $CLAUDE_API_MS,
  "stop_reason": $STOP_REASON,
  "is_error": $IS_ERROR
}
EOF

    # Capture the agent's changes as a patch diff + summary stat.
    # The baseline commit above pinned post-setup state, so everything here
    # is strictly what the agent modified during the run.
    if [ "$HAS_GIT" -eq 1 ]; then
      git "${GIT_SAFE[@]}" -C /scratch add -A 2>/dev/null || true
      git "${GIT_SAFE[@]}" -C /scratch diff --cached        > "$PATCH"      2>/dev/null || true
      git "${GIT_SAFE[@]}" -C /scratch diff --cached --stat > "$PATCH_STAT" 2>/dev/null || true
    fi

    # Transcript is already captured directly via --output-format stream-json
    # above; no post-hoc filesystem search needed.

    if [ -x "/tasks/$TASK_ID/verify.sh" ]; then
      echo "$MODE[$TASK_ID]: running verify.sh"
      bash "/tasks/$TASK_ID/verify.sh" /scratch >"$VERIFY_LOG" 2>&1
      VEXIT=$?
      echo "{\"verify_exit\": $VEXIT}" >"$VERIFY_JSON"
    fi

    echo "$MODE[$TASK_ID]: done exit=$EXIT dur=${DURATION}s"
    exit 0
    ;;

  *)
    fail "unknown mode: $MODE (expected: run|surrogate|compile|shell)"
    ;;
esac
