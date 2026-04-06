# Context Management for AI Coding Agents

Research into reducing context window waste and detecting circular reasoning in AI coding agents, tested on Claude Code with a real GCC compiler bug.

## The Problem

AI coding agents accumulate all tool output in their context window forever. After 30 minutes of debugging, half the context is stale test output, old tree dumps, and failed approach artifacts. The model has no mechanism to:
1. Discard tool outputs it has already processed
2. Recognize when it's going in circles
3. Backtrack from a failed approach

## Two Approaches Explored

### 1. Patching Claude Code (`claude-code-rewind/`)

Direct patches to Claude Code's minified JS binary. Adds auto-compact (truncate old Bash outputs) and a Rewind tool. **Abandoned** — fragile, breaks on updates, and the model doesn't use novel tools it wasn't trained on.

See [REWIND_CHANGES.md](claude-code-rewind/REWIND_CHANGES.md) for the full patching approach documentation.

### 2. HTTP Proxy (`stuck-detector-proxy/`) ← Current approach

A local proxy between Claude Code and the Anthropic API. Intercepts requests to compact tool outputs and detect stuck reasoning. **No patches, no plugins, works with vanilla Claude Code.**

```
Claude Code (unmodified)
    │
    │  ANTHROPIC_BASE_URL=http://localhost:8080
    │
    ▼
Proxy (localhost:8080)
    ├── Compact old Bash tool results in message array
    ├── Detect circular thinking patterns (heuristic)
    ├── Inject corrective nudge when stuck detected
    └── Forward to api.anthropic.com with auth headers
```

#### Usage

```bash
cd stuck-detector-proxy
node proxy.mjs &

# Run vanilla Claude Code through the proxy
ANTHROPIC_BASE_URL=http://localhost:8080 claude "your prompt"

# A/B testing is trivial — without proxy:
claude "your prompt"
```

#### Configuration

| Variable | Default | Description |
|---|---|---|
| `PROXY_PORT` | `8080` | Listen port |
| `PROXY_UPSTREAM` | `https://api.anthropic.com` | Upstream API |
| `COMPACT_ENABLED` | `1` | Auto-compact Bash outputs |
| `COMPACT_STALE_TURNS` | `2` | Turns before compaction |
| `COMPACT_KEEP_FIRST` | `30` | Lines kept from start |
| `COMPACT_KEEP_LAST` | `10` | Lines kept from end |
| `COMPACT_MIN_LINES` | `50` | Minimum lines to trigger |
| `STUCK_ENABLED` | `1` | Stuck detection |
| `STUCK_COOLDOWN` | `5` | Turns between nudges |

## Benchmark: GCC Compiler Bug (PR 123310)

Tested on [GCC PR 123310](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=123310) — a wrong-code bug in the value numbering pass (`tree-ssa-sccvn.cc`). The fix is a 1-character change: `-1U` → `-1` in an offset comparison. The unsigned `-1U` (0x00000000FFFFFFFF) doesn't match the signed sentinel `-1` (0xFFFFFFFFFFFFFFFF), causing incorrect aggregate copy translation at `-O2`.

- **Reproducer**: A struct copy in a loop produces wrong value (0 instead of 5)
- **Root cause**: `known_ne(lhs_ops[j].off, -1U)` should be `known_ne(lhs_ops[j].off, -1)`
- **Affected versions**: GCC 12-15 (regression from [r12-2657](https://gcc.gnu.org/git/?p=gcc.git;a=commit;h=f9fcf754825a1e))
- **Fix commit**: [r16-6577](https://gcc.gnu.org/git/?p=gcc.git;a=commit;h=6225251b9005) by Richard Biener

Each trial involves reading thousands of lines of compiler source, running tree dumps, adding debug prints, rebuilding GCC, and iterating. 50-200 tool calls per run.

### Results

#### Patching approach (v1-v5)

| Version | Approach | Result |
|---|---|---|
| v1: truncate all | All tool outputs after 2 turns | **Caused 8 re-reads**, 86% slower |
| v2: Bash-only | Only truncate Bash, never Read/Edit | 0 re-reads, modest savings |
| v4: ephemeral default-true | Bash ephemeral by default | Clean savings, model never uses opt-out |
| v5: +CLAUDE.md | Added explicit instructions | Model ignores — behavior unchanged |

#### GCC bug trials (patching approach)

| Trial | Stock | Patched | Winner |
|---|---|---|---|
| T2 (parallel) | 1537s, 17.5M tokens | 1169s, 14.3M tokens | **Patched by 24%** |
| T3 (parallel) | 867s, 5.7M tokens | 1731s, 21.5M tokens | Stock |

#### Proxy approach

| Run | Duration | Compactions | Stuck nudges | Correct fix? |
|---|---|---|---|---|
| Proxy (fixed) | 1636s | 7 | 6 (turns 72, 86, 117) | ✓ |

The stuck detector fired 3 times:
- **Turn 72**: Model stuck in `varpool.cc` (irrelevant file) — nudge injected
- **Turn 86**: Model stuck in `tree-ssa-alias.cc` (wrong subsystem) — nudge injected
- **Turn 117**: Model iterating in `tree-ssa-sccvn.cc` (right file) — nudge injected

Model eventually found the correct `-1U` → `-1` fix.

## Key Findings

1. **Only compact Bash outputs.** Truncating Read/Edit/Write outputs causes the model to re-read files, costing more than it saves.

2. **Models don't use novel tools without training.** `ephemeral` parameter, `Rewind` tool, CLAUDE.md instructions — the model ignores all of them. Agent-mode behavior is trained, not prompted.

3. **Stuck heuristic works.** Substring repetition + keyword counting correctly detects circular reasoning. Fired at turns 30/75 (patching trials) and turns 72/86/117 (proxy trial).

4. **Proxy > patches > plugins.** Proxy gives full message control, survives updates, works with vanilla Claude Code, enables trivial A/B testing.

5. **Variance dominates.** Same task, same model: 219s to 1731s range across trials. Non-deterministic token sampling determines the reasoning path.

6. **Smaller models would benefit more.** Opus solves the GCC bug every time (just at varying speed). A 7-35B model with limited context would benefit dramatically from auto-compact and stuck detection.

## Related Work

- [MemGPT](https://arxiv.org/abs/2310.08560) — Virtual memory paging for LLMs
- [LATS](https://arxiv.org/abs/2310.04406) — Tree search with backtracking for agents
- [Reflexion](https://arxiv.org/abs/2303.11366) — Self-reflection for LLM agents
- [Meta-Harness](https://arxiv.org/abs/2603.28052) — End-to-end harness optimization (raw traces beat summaries)
- [context-mode](https://github.com/mksglu/context-mode) — MCP-based context savings plugin for Claude Code

## Next Steps

See [stuck-detector-prompt.md](stuck-detector-prompt.md) for the full plan:
1. Train a stuck classifier on real reasoning traces (replace heuristic)
2. LoRA fine-tune an open source model (Qwen 3.5 Coder) on context management behaviors
3. Benchmark on SWE-bench with the proxy

## License

MIT for all code in this repo. Claude Code is under Anthropic's license — the proxy does not modify or redistribute it.
