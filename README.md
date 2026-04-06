# Context Management for AI Coding Agents

Research into reducing context window waste and detecting circular reasoning in AI coding agents, tested on Claude Code with a real GCC compiler bug.

## The Problem

AI coding agents accumulate all tool output in their context window forever. After 30 minutes of debugging, half the context is stale test output, old tree dumps, and failed approach artifacts. The model has no mechanism to:
1. Discard tool outputs it has already processed
2. Recognize when it's going in circles
3. Backtrack from a failed approach

## HTTP Proxy (`stuck-detector-proxy/`)

A local proxy between Claude Code and the Anthropic API. Intercepts requests to compact tool outputs and detect stuck reasoning. **No patches, no plugins, works with vanilla Claude Code.**

```
Claude Code (unmodified)
    │
    │  ANTHROPIC_BASE_URL=http://localhost:8080
    │
    ▼
Proxy (localhost:8080)
    ├── Compact old Bash tool results in message array
    ├── Detect circular thinking (trained LogReg classifier, pure JS)
    ├── Inject corrective nudge when stuck detected
    ├── Retry with exponential backoff on 429/529
    ├── Concurrency limiter (semaphore, default 5 in-flight)
    └── Forward to api.anthropic.com with auth headers
```

### Usage

```bash
cd stuck-detector-proxy
node proxy.mjs &

# Run vanilla Claude Code through the proxy
ANTHROPIC_BASE_URL=http://localhost:8080 claude "your prompt"

# A/B testing is trivial — without proxy:
claude "your prompt"

# Monitor concurrency under load:
curl http://localhost:8080/stats
```

### Configuration

| Variable | Default | Description |
|---|---|---|
| `PROXY_PORT` | `8080` | Listen port |
| `PROXY_UPSTREAM` | `https://api.anthropic.com` | Upstream API |
| `PROXY_MAX_CONCURRENT` | `5` | Max in-flight upstream requests |
| `PROXY_MAX_RETRIES` | `8` | Max retries on 429/529 |
| `PROXY_BASE_DELAY_MS` | `1000` | Initial backoff delay |
| `PROXY_MAX_DELAY_MS` | `60000` | Max backoff delay |
| `COMPACT_ENABLED` | `1` | Auto-compact Bash outputs |
| `COMPACT_STALE_TURNS` | `2` | Turns before compaction |
| `COMPACT_KEEP_FIRST` | `30` | Lines kept from start |
| `COMPACT_KEEP_LAST` | `10` | Lines kept from end |
| `COMPACT_MIN_LINES` | `50` | Minimum lines to trigger |
| `STUCK_ENABLED` | `1` | Stuck detection |
| `STUCK_THRESHOLD` | `0.80` | Classifier confidence threshold |
| `STUCK_COOLDOWN` | `5` | Turns between nudges |

### Stuck Classifier

Logistic regression classifier (15 features, `class_weight=balanced`) that detects circular reasoning in agent thinking blocks. Inference runs in pure JS — no Python dependency at runtime.

Two categories of features:

**Text features** (9) — extracted from the thinking block text:
`self_sim`, `max_substr_repeat`, `circle_kw`, `false_starts`, `avg_sent_len`, `sent_len_std`, `vocab_diversity`, `code_ratio`, `question_marks`

**Tool-call behavioral features** (6) — extracted from the message history:
`bash_cmd_repeat` (+2.84), `tool_diversity` (-3.03), `circle_kw` (+0.99), `code_ratio` (-1.53), `false_starts` (+0.70), `file_read_repeat` (-0.86)

Tool features dominate: `bash_cmd_repeat` (agent re-running the same command) and low `tool_diversity` (same tool over and over) are the strongest stuck signals.

Trained on 151 windows with full tool-call data from 17 tasks (80 stuck, 71 productive):

| Threshold | Precision | Recall | FP | FN |
|---|---|---|---|---|
| 0.50 | 79% | 100% | 25 | 0 |
| 0.85 | 85% | 96% | 16 | 4 |
| 0.95 | 91% | 90% | 9 | 10 |

### Training Data

The classifier is trained on labeled 1000-character windows of agent thinking blocks, collected by running agents on real bugs and feature implementations across diverse codebases. Each window is labeled stuck/productive based on manual review of agent transcripts.

**Training tasks — bugs:**

| Task | Codebase | Bug |
|---|---|---|
| [GCC PR 123310](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=123310) | GCC tree-ssa-sccvn.cc | Wrong aggregate copy: `-1U` vs `-1` offset comparison |
| [GCC PR 123864](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=123864) | GCC match.pd | `__builtin_mul_overflow_p` wrong with unsigned, missing `!TYPE_UNSIGNED` guard |
| [LLVM #125374](https://github.com/llvm/llvm-project/issues/125374) | LLVM VPlanTransforms.cpp | Loop vectorizer replaces reduction with wrong SCEV live-in |
| [SQLite forum/86ddb1e](https://sqlite.org/forum/info/86ddb1effebcfa5c) | SQLite optimizer | EXISTS-to-JOIN with UNION returns empty (3.51.0 regression) |
| [curl CVE-2025-0665](https://curl.se/docs/CVE-2025-0665.html) | curl connect.c | eventfd double close on threaded resolver teardown |
| [Django #36109](https://code.djangoproject.com/ticket/36109) | Django ORM query.py | Chained FilteredRelation causes RecursionError |
| [Express v5.2.0](https://github.com/expressjs/express/releases/tag/v5.2.1) | Express.js | Query parser returns null-prototype object, breaks `hasOwnProperty` |
| [Linux xHCI](https://bbs.archlinux.org/viewtopic.php?id=303879) | Linux xhci-ring.c | Link TRB cycle bit not cleared on suspend/resume |
| [Linux btrfs](https://lists.debian.org/debian-kernel/2025/08/msg00125.html) | Linux tree-log.c | Log replay fails for 0-link inodes with extents |
| [LAPACK #1138](https://github.com/Reference-LAPACK/lapack/issues/1138) | LAPACK dlasd7.f | SVD convergence failure from ordering bug in divide-and-conquer ([fix](https://github.com/Reference-LAPACK/lapack/pull/1140)) |
| [Boost.Beast #3028](https://github.com/boostorg/beast/issues/3028) | Boost Beast websocket | permessage-deflate corruption with small read buffers ([fix](https://github.com/boostorg/beast/pull/3029)) |

**Training tasks — features:**

| Task | Codebase | Feature |
|---|---|---|
| [LAPACK #1155](https://github.com/Reference-LAPACK/lapack/pull/1155) | LAPACK BLAS | Skew-symmetric matrix subroutines (DSKEWSYMV, DSKEWSYR2, DSKEWSYMM, DSKEWSYR2K) |
| [Boost.Geometry #1409](https://github.com/boostorg/geometry/pull/1409) | Boost Geometry | `is_valid` algorithm for 3D polyhedral surfaces |

**Training tasks — synthetic algorithms:**

| Task | Bug |
|---|---|
| Red-black tree | Wrong color assignment in deletion fixup mirror case |
| A* pathfinding | Inadmissible heuristic (average weight instead of minimum) |
| Raft consensus | Fixed election timeout causes split vote deadlock |

**Stuck patterns observed:**
- **Hypothesis cycling** — repeating the same theory with minor variations (GCC match.pd, 11 blocks)
- **Grep broadening spiral** — 12+ regex variations searching for code that doesn't exist (LLVM, 36 blocks)
- **Can't-reproduce loop** — tests pass but problem statement says there's a bug (RBTree, 7 blocks)
- **Wrong-version search** — searching for code in a refactored codebase (curl, 23 blocks)

### Training Pipeline

```
1. Run agents on tasks (run_all_tasks.sh)
   └── Proxy with STUCK_ENABLED=0 so agents behave naturally

2. Extract thinking blocks from transcripts
   └── *_thinking.json per task

3. Review transcripts for stuck episodes
   └── Manual or agent-assisted labeling

4. Window into 1000-char chunks, label stuck/productive
   └── labeled_windows.json

5. Train: python3 train.py --balanced --tool-features
   └── stuck_classifier.pkl

6. Export: python3 export_model.py
   └── model_weights.json (loaded by classify.mjs at runtime)
```

## Benchmark: GCC Compiler Bug (PR 123310)

Tested on [GCC PR 123310](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=123310) — a wrong-code bug in the value numbering pass (`tree-ssa-sccvn.cc`). The fix is a 1-character change: `-1U` → `-1` in an offset comparison.

| Run | Duration | Compactions | Stuck nudges | Correct fix? |
|---|---|---|---|---|
| Proxy | 1636s | 7 | 3 (turns 72, 86, 117) | Yes |

## Key Findings

1. **Only compact Bash outputs.** Truncating Read/Edit/Write outputs causes the model to re-read files, costing more than it saves.

2. **Models don't use novel tools without training.** `ephemeral` parameter, `Rewind` tool, CLAUDE.md instructions — the model ignores all of them. Agent-mode behavior is trained, not prompted.

3. **Proxy > patches > plugins.** Proxy gives full message control, survives updates, works with vanilla Claude Code, enables trivial A/B testing.

4. **Variance dominates.** Same task, same model: 219s to 1731s range across trials. Non-deterministic token sampling determines the reasoning path.

5. **Tool-call features beat text features.** An agent re-running the same grep or re-reading the same file is a much stronger stuck signal than keyword counting or vocabulary analysis in the thinking text.

## Related Work

- [MemGPT](https://arxiv.org/abs/2310.08560) — Virtual memory paging for LLMs
- [LATS](https://arxiv.org/abs/2310.04406) — Tree search with backtracking for agents
- [Reflexion](https://arxiv.org/abs/2303.11366) — Self-reflection for LLM agents
- [Meta-Harness](https://arxiv.org/abs/2603.28052) — End-to-end harness optimization (raw traces beat summaries)
- [context-mode](https://github.com/mksglu/context-mode) — MCP-based context savings plugin for Claude Code

## Next Steps

1. Collect more training data from diverse codebases (LAPACK, Boost, React)
2. Evaluate classifier precision/recall on held-out tasks
3. LoRA fine-tune an open source model (Qwen 3.5 Coder) on context management behaviors
4. Benchmark on SWE-bench with the proxy

## License

MIT for all code in this repo. Claude Code is under Anthropic's license — the proxy does not modify or redistribute it.
