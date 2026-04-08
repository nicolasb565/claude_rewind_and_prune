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
| 0.70 | 81% | 99% | 22 | 1 |
| 0.75 | 83% | 98% | 19 | 2 |
| **0.80** | **85%** | **98%** | **17** | **2** |
| 0.85 | 85% | 96% | 16 | 4 |

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

## Benchmark Results (5 runs per condition)

13 tasks run 5 times each with classifier OFF and ON. Median comparison:

| Task | OFF (median) | ON (median) | Δ time | Δ tokens |
|---|---|---|---|---|
| 24_rbtree_bug | 671s | 45s | **-93%** | **-88%** |
| 02_gcc_bug | 697s | 371s | **-47%** | **-52%** |
| 06_django_bug | 688s | 366s | **-47%** | **-32%** |
| 12_linux_fs_bug | 270s | 218s | **-19%** | -9% |
| 08_express_bug | 78s | 67s | **-14%** | -16% |
| 32_beast_bug | 293s | 262s | -11% | -15% |
| 07_react_bug | 180s | 167s | -7% | -16% |
| 04_sqlite_bug | 70s | 66s | -6% | +11% |
| 33_geometry_feature | 543s | 517s | -5% | +2% |
| 01_gcc_bug | 114s | 110s | -4% | -9% |
| 03_llvm_bug | 1375s | 1628s | +18% | +34% |
| 10_linux_usb_bug | 56s | 69s | +23% | +23% |
| 30_lapack_bug | 162s | 245s | +51% | +185% |
| **TOTAL** | **5197s** | **4131s** | **-21%** | **-12%** |

The classifier consistently helps on tasks where agents get stuck (RBTree, GCC#2, Django — large improvements across all 5 runs). Tasks with clean solves (SQLite, USB, GCC#1) show no significant change. High-variance regressions (LLVM, LAPACK) are driven by non-deterministic sampling rather than classifier interference — the same tasks show 10x variance between runs with or without the classifier.

### Held-out evaluation (unseen tasks)

6 tasks the classifier was never trained on (3 hard, 3 easy), from GCC, LLVM, Django, and LAPACK:

| Task | Type | OFF (s) | ON (s) | Δ time | OFF (tok) | ON (tok) | Δ tokens |
|---|---|---|---|---|---|---|---|
| 40_django_jsonfield | hard | 68 | 54 | -21% | 2,801 | 2,590 | -8% |
| 41_django_keytexttransform | easy | 324 | 266 | -18% | 10,455 | 14,987 | +43% |
| 42_lapack_uninit | easy | 116 | 84 | -28% | 2,261 | 1,812 | -20% |
| 43_gcc_tbaa | hard | 137 | 390 | +185% | 5,406 | 16,212 | +200% |
| 44_llvm_arith | hard | 156 | 143 | -8% | 3,239 | 5,813 | +79% |
| 45_llvm_delete | easy | 164 | 86 | -48% | 7,781 | 2,746 | -65% |
| **TOTAL** | | **965** | **1,023** | **+6%** | **31,943** | **44,160** | **+38%** |

**The classifier does not generalize well to unseen tasks.** On held-out data, it increased token usage by 38%. The root cause: agents solving these tasks productively but slowly (many tool calls, deep file exploration) trigger false positives. The training data lacks examples of "productive but takes many turns" — it mostly contains quick clean solves as productive samples, so the model learned "many tool calls = stuck."

This is a training data diversity problem, not an architecture problem. The tool-call features are the right signal; we just need more diverse productive samples that include long, multi-step successful debugging sessions.

## Key Findings

1. **Stuck detection saves 21% time and 12% tokens on known tasks.** On tasks where agents genuinely get stuck, savings reach 47-93%. On clean-solve tasks, the classifier correctly stays silent.

2. **Does not generalize to unseen tasks yet.** The classifier fires false positives on productive agents that happen to use many tool calls, because the training data lacks slow-but-productive examples. More diverse training data would fix this.

3. **Tool-call features are the right signal, but need better training data.** `bash_cmd_repeat` and `tool_diversity` are the strongest features. They correctly identify behavioral loops but can't yet distinguish "exploring thoroughly" from "going in circles" without more examples of each.

4. **Only compact Bash outputs.** Truncating Read/Edit/Write outputs causes the model to re-read files, costing more than it saves.

5. **Variance dominates.** Same task, same model: 93s to 3210s range across trials (LLVM). Non-deterministic token sampling determines the reasoning path. 5+ runs per condition needed for statistical comparison.

6. **Models don't use novel tools without training.** `ephemeral` parameter, `Rewind` tool, CLAUDE.md instructions — the model ignores all of them. Agent-mode behavior is trained, not prompted.

7. **Proxy > patches > plugins.** Proxy gives full message control, survives updates, works with vanilla Claude Code, enables trivial A/B testing.

## Related Work

### Stuck/loop detection in agents

- [SpecRA](https://openreview.net/forum?id=xVO4BqmzVD) (Oct 2025) — Uses FFT autocorrelation on token sequences to detect periodicity in agent output. Signal-processing approach at the token level, no behavioral features.
- [Agentic Metacognition](https://arxiv.org/abs/2509.19783) (Xu, Sep 2025) — External metacognitive layer monitors a primary agent for repetitive actions and loop traps. On predicted failure, hands off to a human. Closest architectural match to our proxy approach.
- [strongdm/attractor](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md) — Open-source spec that tracks tool-call signatures over a sliding window, detects cycles, and injects a steering message. Same concept as our system but no ML classifier.
- [Ralph](https://github.com/frankbria/ralph-claude-code) — Autonomous dev loop wrapper for Claude Code with exit detection and circuit breaker patterns.

### Self-reflection and metacognition

- [Reflexion](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023) — Verbal self-reflection after task failure. Post-hoc, not real-time.
- [Multi-Agent Reflexion](https://arxiv.org/html/2512.20845) (Dec 2025) — Diverse reasoning personas + judge to avoid repeating misconceptions.
- [LLMs Have Metacognitive Monitoring](https://arxiv.org/abs/2505.13763) (Ji-An et al., May 2025) — Shows LLMs can monitor their own activations, but only in a low-dimensional subspace, and may learn to obfuscate internals to evade oversight.
- [Experiential Reflective Learning](https://arxiv.org/html/2603.24639) (Mar 2026) — Builds reusable heuristics from past failure trajectories, injected as context for new tasks.

### Context management

- [MemGPT](https://arxiv.org/abs/2310.08560) — Virtual memory paging for LLMs
- [LATS](https://arxiv.org/abs/2310.04406) — Tree search with backtracking for agents
- [Meta-Harness](https://arxiv.org/abs/2603.28052) — End-to-end harness optimization (raw traces beat summaries)
- [context-mode](https://github.com/mksglu/context-mode) — MCP-based context savings plugin for Claude Code

### What's different about our approach

None of the above combine proxy-based interception, thinking-block text features, tool-call behavioral features, and a trained classifier with corrective nudge injection. The metacognition research (Ji-An et al.) suggests intrinsic self-monitoring is limited and gameable, supporting the case for an external monitor.

Longer-term, this kind of monitoring belongs inside the model or API — similar to how speculative decoding uses a small draft model alongside the main model. A lightweight "reasoning monitor" model could run in parallel during inference, detecting stuck patterns at the token level and redirecting attention before a full stuck episode forms, without consuming the main model's capacity. The proxy approach demonstrated here is a proof of concept — the real implementation should be at the inference layer where the monitor has access to internal model state, not just the message history.

## The Case for Built-in Stuck Detection

AI coding agents routinely waste 30-50% of their token budget going in circles on hard tasks. This is a known problem that every user of Claude Code, Cursor, Copilot Workspace, and similar tools experiences. Our proxy demonstrates that:

1. **Stuck behavior is detectable** — a simple logistic regression on 15 features achieves 85% precision at 98% recall on known tasks
2. **Corrective nudges work** — injecting a "you may be going in circles" message saves 21% time and 12% tokens on average, with 47-93% savings on the hardest tasks
3. **The right signals are behavioral, not textual** — tool-call repetition patterns are far more reliable than analyzing the reasoning text
4. **An external monitor can't fully distinguish exploration from stuck** — the classifier needs examples of both to learn the boundary, and a proxy only sees messages, not internal model state

This last point is why it should be built into the model or API. An inference-layer monitor could access attention patterns, internal confidence signals, and the full generation state — not just the finished messages. It could intervene at the token level before a stuck episode fully forms, rather than waiting for a complete turn to analyze.

## CNN Stuck Detector v2 (`stuck-detector-v2/`)

A language-agnostic CNN trained on 12,477 trajectories from 6 public SWE-bench datasets, replacing the 151-window LogReg classifier. Uses cycle-detection features (CRC32 hashed commands/files, Jaccard output similarity) that generalize across programming languages, agent scaffolds, and model families.

### Architecture

3,133-parameter CNN with two parallel Conv1d branches (kernel 3 + kernel 5), tool embedding, and window-level aggregate features. Input: 10-step windows of 19 features each + 6 window features. Output: stuck probability.

### Training Pipeline

```
6 datasets (MEnvData, Nebius OH, Nemotron, SWE-Smith, SWE-Gym, Nebius ADP)
    │ 330K+ trajectories available
    ▼
Curate 12,477 balanced pool (67% strong, 19% medium, 14% weak models)
    │
    ▼
Abstract to language-agnostic features (CRC32 hashing, Jaccard similarity)
    │
    ▼
Label via Sonnet with precomputed counts (496 STUCK, 11,788 PRODUCTIVE)
    │
    ▼
Window into 118K 10-step chunks (stride 5)
    │
    ▼
Train 3.1K param CNN (30.3x data ratio, class-balanced loss)
```

### Results

| Metric | Value |
|---|---|
| Precision (at recall >= 70%) | 77.3% |
| Recall | 77.9% |
| F1 | 0.776 |
| Threshold | 0.94 |
| Weights size | 68 KB JSON |

### Feature Importance (ablation)

| Feature | Impact when removed |
|---|---|
| `output_length` | -5.0% (critical) |
| `steps_since_same_cmd` | -4.9% (critical) |
| `thinking_length` | -4.0% (critical) |
| `tool_count_in_window` | -3.5% (critical) |
| `step_index_norm` | -2.9% (helpful) |
| `tool_embed` | -2.8% (helpful) |
| `false_start`, `strategy_change` | ~0% (neutral) |

### Datasets Used

| Dataset | License | Trajectories | Model | Role |
|---|---|---|---|---|
| [ernie-research/MEnvData-SWE-Trajectory](https://huggingface.co/datasets/ernie-research/MEnvData-SWE-Trajectory) | Apache-2.0 | 3,918 | Claude Sonnet | Strong, multi-language |
| [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | CC-BY-4.0 | 3,000 (sampled) | Qwen3-Coder-480B | Strong |
| [nvidia/Nemotron-SWE-v1](https://huggingface.co/datasets/nvidia/Nemotron-SWE-v1) | CC-BY-4.0 | 1,500 (sampled) | Qwen3-Coder-480B | Strong |
| [neulab/agent-data-collection](https://huggingface.co/datasets/neulab/agent-data-collection) (swe-smith) | MIT | 1,500 (sampled) | Qwen 2.5 Coder 32B | Medium |
| neulab/agent-data-collection (swe-gym) | MIT | 491 | OpenHands | Medium |
| neulab/agent-data-collection (nebius) | CC-BY-4.0 | 1,717 (sampled) | Llama 70B, Qwen 72B | Weak |

### Labeling Pipeline

Trajectories were labeled by Sonnet agents applying deterministic rules on precomputed feature counts (tight-loop steps, diverse steps, error steps). Rules iterated through 5 rounds of prompt tuning with Opus sanity checks. Updated rules include error-based STUCK detection (error_steps >= 7 AND diverse < 3) and `>=` threshold fix recommended by Opus review.

## Next Steps

1. Try Stage 2 CNN (25K params) if precision needs improvement
2. JS forward pass implementation and proxy integration
3. LoRA fine-tune an open source model (Qwen 3.5 Coder) on context management behaviors
4. Benchmark CNN vs LogReg on the 13-task suite
5. Explore lightweight monitor model architecture (speculative-decoding-style parallel inference)

## License

MIT for all code in this repo. Claude Code is under Anthropic's license — the proxy does not modify or redistribute it.
