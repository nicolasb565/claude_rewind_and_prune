# Context Rewind & Auto-Compact — Changes

This fork adds two context management mechanisms to Claude Code:

1. **Auto-compact** (invisible, always on) — automatically truncates tool outputs after the model has processed them
2. **Rewind** (model-initiated tool) — lets the model abandon a failed approach and replace it with a compact summary

## Architecture

Since Claude Code is distributed as a compiled/minified JS binary, modifications are applied as **text patches** to the original `cli.js`. The patcher (`src/patch.mjs`) locates unique anchor strings in the minified source and inserts the new code.

### Files

| File | Purpose |
|---|---|
| `vendor/cli-original.js` | Unmodified Claude Code CLI (v2.1.92) |
| `src/patch.mjs` | Patcher script — applies 4 patches to generate the runnable CLI |
| `src/bootstrap.mjs` | Loader — sets up config, telemetry, then imports the patched CLI |
| `bin/claude` | Shell wrapper — auto-patches if needed, sets env, runs bootstrap |
| `bin/claude-rewind.js` | Generated — the patched CLI (gitignored) |

### Patches Applied

1. **Auto-compact truncation** — inserted after microcompact phase in the main query loop. Truncates `tool_result` content that is older than N assistant turns.

2. **Rewind tool registration** — adds a `Rewind` tool to the core tool list (alongside Read, Write, Edit, etc.). Only active when `CLAUDE_REWIND_MODE=full`.

3. **Rewind handler in main loop** — after tool execution, checks for a pending Rewind request and truncates the message array accordingly.

4. **System prompt addition** — appends Rewind usage instructions to the base system prompt when in full mode.

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CLAUDE_REWIND_MODE` | `full` | `off` / `compact_only` / `full` |
| `REWIND_STALE_TURNS` | `2` | Turns before tool output becomes eligible for truncation |
| `REWIND_KEEP_FIRST` | `20` | Lines to keep from start of truncated output |
| `REWIND_KEEP_LAST` | `5` | Lines to keep from end of truncated output |
| `REWIND_MIN_LINES` | `30` | Minimum output length to trigger truncation |

### Three Conditions (for benchmarking)

| Condition | Command |
|---|---|
| Stock | `claude` (system install) |
| Compact only | `CLAUDE_REWIND_MODE=compact_only ./bin/claude` |
| Compact + Rewind | `CLAUDE_REWIND_MODE=full ./bin/claude` |

## How Auto-Compact Works

After each turn, before the next API call, the orchestrator scans all messages:

1. For each `user` message containing `tool_result` blocks:
2. Count how many assistant turns have occurred since that result
3. If older than `staleAfterTurns` (default: 2):
4. If the content is longer than `minLinesForCompaction` (default: 30 lines):
5. Replace with first 20 lines + `[... N lines truncated ...]` + last 5 lines
6. Prefix with `[COMPACTED — original was N lines]`

Already-compacted and already-cleared results are skipped. Short outputs are left untouched.

## How Rewind Works

1. Model calls `Rewind(turns_back=N, summary="...")` 
2. Tool handler validates: summary >= 100 chars, turns_back >= 1, session limit not exceeded (max 5)
3. Stores the request in `globalThis.__REWIND_PENDING__`
4. Returns confirmation message as tool result
5. After tool execution completes, the main loop handler:
   - Counts assistant messages to find the Nth-from-last
   - Truncates the message array at that point
   - Injects a user message with `[REWIND — N messages pruned]` + summary
   - Clears the pending request
6. Next iteration proceeds with the truncated history

### Guardrails

- Summary must be >= 100 characters
- turns_back must be >= 1 and <= total turns minus 1
- Max 5 rewinds per session
- Does NOT undo file changes (summary should mention what was modified)

## Telemetry

Events are logged to `~/.claude-rewind-logs/events-YYYY-MM-DD.jsonl`:

- `session_start` — config, mode, version
- `session_end` — duration
- `compact` — per tool result: original lines, compacted lines, tokens saved estimate
- `rewind` — turns back, summary length, rewind number
- `rewind_applied` — messages pruned, messages remaining

## Building

```bash
# Generate the patched CLI
node src/patch.mjs

# Verify patches can be applied (dry run)
node src/patch.mjs --check

# Run the fork
./bin/claude "your prompt here"

# Or with explicit mode
CLAUDE_REWIND_MODE=compact_only ./bin/claude "your prompt here"
```

## Updating

When Claude Code releases a new version:

1. Download the new `cli.js` to `vendor/cli-original.js`
2. Run `node src/patch.mjs --check` to verify anchors still exist
3. If anchors moved, update the search strings in `src/patch.mjs`
4. Run `node src/patch.mjs` to regenerate
