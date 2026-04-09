# Context Management for AI Coding Agents

Research into detecting when Claude Code goes in circles — a 2,621-parameter CNN trained on 85K windows of real Claude Code sessions, running entirely in JavaScript inside a local proxy. When the agent gets stuck, the proxy injects a corrective nudge. Language-agnostic, no Python runtime, no patches to Claude Code.

## The Problem

When AI coding agents get stuck on a hard task, they can burn a significant portion of their token budget going in circles:
- Re-running the same failing command with minor variations
- Cycling through the same files without making progress
- Generating "summary" text that rationalizes not having solved the problem

On our 13-task benchmark, the worst stuck cases burned 10× more time than a normal solve (e.g. rbtree went from 671s stuck to 45s with a nudge). Stuck episodes are not the common case — most sessions are productive — but when they happen they dominate the cost, and the agent has no built-in mechanism to recognize circular reasoning or backtrack. A lightweight external monitor can.

## HTTP Proxy (`stuck-detector-v2/proxy/`)

A local proxy between Claude Code and the Anthropic API. Intercepts requests, scores the recent tool-call history with a CNN, and injects a corrective nudge when stuck is detected. **No patches, no plugins, works with vanilla Claude Code.**

```
Claude Code (unmodified)
    │
    │  ANTHROPIC_BASE_URL=http://localhost:8080
    │
    ▼
Proxy (localhost:8080)
    ├── Parse tool calls from message history
    ├── Abstract into 10-step sliding windows
    ├── Score with CNN (pure JS, 57 KB weights)
    ├── Inject corrective nudge when stuck detected
    ├── Retry with exponential backoff on 429/529
    └── Forward to api.anthropic.com
```

### Usage

```bash
cd stuck-detector-v2/proxy
node proxy_cnn.mjs &

# Run vanilla Claude Code through the proxy
ANTHROPIC_BASE_URL=http://localhost:8080 claude "your prompt"

# A/B testing is trivial — just unset the env var
```

### Configuration

| Variable | Default | Description |
|---|---|---|
| `PROXY_PORT` | `8080` | Listen port |
| `PROXY_UPSTREAM` | `https://api.anthropic.com` | Upstream API |
| `STUCK_ENABLED` | `1` | Enable stuck detection |
| `STUCK_COOLDOWN` | `5` | Turns between nudges |
| `COMPACT_ENABLED` | `0` | Auto-compact Bash outputs (optional) |

## CNN Stuck Detector

A 2,621-parameter CNN trained on 85,416 labeled windows from real Claude Code sessions. Uses cycle-detection features (CRC32-hashed `base_command:target_file` keys, Jaccard output similarity) that generalize across programming languages, agent scaffolds, and model families.

### Architecture

- **Input:** 10-step sliding windows of tool calls (stride 5)
- **Features per step:** 11 continuous (cycle detection, repetition, errors, output similarity) + 7-way tool embedding
- **Window-level features:** 6 aggregates (unique tool/file/cmd ratios, error rate, output diversity)
- **Model:** 2 parallel Conv1d branches (kernels 3+5, 16 filters each), max pool, MLP head
- **Output:** Sigmoid stuck probability
- **Size:** 57 KB JSON weights, runs in pure JS (no Python, no GPU)

### Results

**Test set** (16,936 windows from held-out trajectories):

| Metric | Value |
|---|---|
| Precision | 87.1% |
| Recall | 86.2% |
| F1 | 0.866 |
| Threshold | 0.96 |
| Weights | 57 KB |

**Benchmark on the LogReg-era task suite** (29 sessions, 6 stuck):

| Metric | Direct (CNN ≥ 0.96) |
|---|---|
| Precision | 71.4% |
| Recall | 83.3% |
| F1 | 0.769 |
| False positives | 2 (08_express, 02_gcc — both test-iteration patterns) |
| False negatives | 1 (03_llvm off_2, max score = 0.000) |

**Held-out tasks** (never seen in training): **all clean**. The earlier model's 44_llvm_arith false positive is fixed.

### Training Pipeline

```
Claude Code sessions (nlile 16.8K + DataClaw 136)
    │
    │ Parse with cmd_semantic_key: 'cd build && make -j8 | tail' → 'make'
    ▼
Abstract to features (CRC32 of semantic key, Jaccard output similarity)
    │
    ▼
Deterministic labeling (STUCK / PRODUCTIVE / UNCLEAR rules)
    │
    │ 82,128 high-confidence STUCK+PRODUCTIVE
    │ 4,074 UNCLEAR → Sonnet agent review with raw text
    │    → 3,277 resolved (STUCK/PRODUCTIVE)
    │    → 665 dropped (still UNCLEAR)
    ▼
85,416 labeled windows → train CNN (class-balanced loss, pos_weight 31:1)
```

### Key Innovations

1. **`cmd_semantic_key`** — Extracts `base_command:target_file` from bash commands. `cd build && ./gcc/xgcc -O2 test.c | tail` → `xgcc:test.c`. Makes command-repetition features work across projects without per-project retraining.

2. **Three-tier labeling** — Deterministic rules for clear cases, Sonnet agent review for UNCLEAR (with raw command/output text, not just numeric features), drop if still ambiguous.

3. **Trimmed feature set** — Dropped 4 near-dead features (`false_start`, `strategy_change`, `circular_lang`, `self_similarity` — thinking-block regex features only populated in 2.5% of data). Going from 15 → 11 features improved F1 from 0.840 → 0.866 and dropped FPs by 29%.

4. **Confirmation rules tested** — 2-of-3, 2-consecutive, streak-based, EMA smoothing. None improved on direct thresholding for this dataset (stuck patterns are short and bursty; multi-window rules mostly hurt recall). The current proxy uses direct CNN output at threshold 0.96.

### Datasets

| Dataset | License | Sessions | Role |
|---|---|---|---|
| [nlile/misc-merged](https://huggingface.co/datasets/nlile/misc-merged) | Apache-2.0 | 16,841 | Primary source, no thinking blocks |
| [DataClaw](https://huggingface.co/datasets/DataClaw) (woctordho) | Apache-2.0 | 136 | Has thinking blocks |

### Feature Set (11 continuous + tool embed + 6 window-level)

| Feature | Signal | Notes |
|---|---|---|
| `steps_since_same_cmd` | **Core** | With cmd_semantic_key, detects command repetition |
| `cmd_count_in_window` | **Core** | Repetition count within window |
| `output_similarity` | **Core** | Jaccard on output lines; same result = stuck |
| `is_error` | Moderate | Errors in loops = stuck, errors alone = debugging |
| `output_length` | Moderate | Log of output line count |
| `step_index_norm` | Moderate | Position in trajectory |
| `tool_count_in_window` | Moderate | Tool repetition frequency |
| `steps_since_same_tool` | Moderate | Tool type repetition |
| `steps_since_same_file` | Moderate | File access repetition |
| `file_count_in_window` | Moderate | File repetition count |
| `thinking_length` | Sparse but useful | Only populated in DataClaw |

Dropped features (near-dead): `false_start`, `strategy_change`, `circular_lang`, `self_similarity`.

**JS forward pass verified:** Pure JS inference matches Python with max diff 3.8e-8 across 100 test vectors. No Node dependencies beyond `node:zlib` for CRC32.

## Key Findings

1. **Stuck is detectable with a tiny model.** 2,621 parameters, trained on ~2,600 real stuck examples, is enough to reach 87% precision / 86% recall on held-out trajectories.

2. **The right signals are behavioral, not textual.** Command repetition and output similarity dominate. Thinking-block regex features (`false_start`, `circular_lang`) are either redundant or sparse.

3. **Training data distribution matters more than architecture.** Switching from SWE-bench (where "low since_cmd" means stuck) to Claude Code (where it often means efficient tool reuse) flipped the sign of multiple features. `cmd_semantic_key` and native Claude Code training data fixed this.

4. **Confirmation rules can't save a model from itself.** 2-of-N and streak-based confirmation either catch too few true stucks (because real patterns are short and bursty) or don't suppress the productive FPs (because test/build iteration is structurally similar to stuck loops).

5. **Generalization to unseen tasks is the hardest problem.** The previous LogReg-based classifier caused +38% token regression on held-out tasks; the old CNN caused the same on gcc_tbaa; this model is clean on all 6 held-out tasks. Getting there required labeling 4,074 ambiguous cases with Sonnet and training on 85K windows.

6. **Remaining weakness: productive edit→build→test cycles.** The one persistent false positive (08_express) is the agent iterating on test failures — structurally similar to a stuck loop at the feature level. Fixing it requires a feature that tracks output **change** between repeated commands, not just similarity.

## Related Work

### Stuck/loop detection in agents

- [SpecRA](https://openreview.net/forum?id=xVO4BqmzVD) (Oct 2025) — FFT autocorrelation on token sequences to detect periodicity. Signal-processing at the token level, no behavioral features.
- [Agentic Metacognition](https://arxiv.org/abs/2509.19783) (Xu, Sep 2025) — External metacognitive layer monitors a primary agent for repetitive actions. Closest architectural match.
- [strongdm/attractor](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md) — Open-source spec tracking tool-call signatures in a sliding window. Same concept as our system but no ML classifier.

### Context management

- [MemGPT](https://arxiv.org/abs/2310.08560) — Virtual memory paging for LLMs
- [LATS](https://arxiv.org/abs/2310.04406) — Tree search with backtracking for agents
- [context-mode](https://github.com/mksglu/context-mode) — MCP-based context savings plugin

### What's different about our approach

This work combines: proxy-based interception, tool-call behavioral features, a trained CNN, and corrective nudge injection — all running in pure JavaScript inside the proxy with no Python runtime. The cleanest comparison point is `strongdm/attractor` which uses similar behavioral signals but no ML.

Longer-term, this kind of monitoring belongs inside the model or API — similar to how speculative decoding uses a small draft model alongside the main model. A lightweight "reasoning monitor" model could run in parallel during inference, detecting stuck patterns at the token level before a full stuck episode forms.

## Next Steps

1. Address the build/test iteration false positive via an "output change between repeated commands" feature
2. Run a 5-run benchmark for statistical significance
3. Add timestamp-based heuristics in the proxy (fast retries boost stuck score, slow gaps dampen)
4. Explore a lightweight speculative-decoding-style parallel monitor inside inference

## License

MIT for all code in this repo. Claude Code is under Anthropic's license — the proxy does not modify or redistribute it.
