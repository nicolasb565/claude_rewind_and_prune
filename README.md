# Context Management for AI Coding Agents

Research into detecting when Claude Code goes in circles — a 5,569-parameter per-step MLP trained on real Claude Code sessions, running entirely in JavaScript inside a local proxy. When the agent gets stuck, the proxy injects a corrective nudge. Language-agnostic, no Python runtime, no patches to Claude Code.

## The Problem

When AI coding agents get stuck on a hard task, they can burn a significant portion of their token budget going in circles:
- Re-running the same failing command with minor variations
- Cycling through the same files without making progress
- Generating "summary" text that rationalizes not having solved the problem

On our 13-task benchmark, the worst stuck cases burned 10× more time than a normal solve (e.g. rbtree went from 671s stuck to 45s with a nudge). Stuck episodes are not the common case — most sessions are productive — but when they happen they dominate the cost, and the agent has no built-in mechanism to recognize circular reasoning or backtrack. A lightweight external monitor can.

## HTTP Proxy (`proxy/`)

A local proxy between Claude Code and the Anthropic API. Intercepts requests, scores the recent tool-call history with a per-step MLP, and injects a corrective nudge when stuck is detected. **No patches, no plugins, works with vanilla Claude Code.**

```
Claude Code (unmodified)
    │
    │  ANTHROPIC_BASE_URL=http://localhost:8080
    │
    ▼
Proxy (localhost:8080)
    ├── Parse tool calls from message history
    ├── Extract per-step features (8 continuous)
    ├── Score with per-step MLP + ring buffer (pure JS, ~60 KB weights)
    ├── Inject escalating nudge when stuck detected
    ├── Retry with exponential backoff on 429/529
    └── Forward to api.anthropic.com
```

### Usage

```bash
node proxy/proxy.mjs &

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
| `COMPACT_ENABLED` | `0` | Auto-compact Bash outputs (optional) |
| `PROXY_MAX_RETRIES` | `8` | Max retries on 429/529 |
| `PROXY_MAX_CONCURRENT` | `5` | Max in-flight upstream requests |
| `LOG_DIR` | `~/.stuck-detector/logs/` | JSONL event log directory |

Threshold and cooldowns are read from `proxy/stuck_config.json` (written by `train.py`).

### Escalating Nudge

When the MLP fires, it injects a corrective message into the conversation. If the agent stays stuck across multiple cooldown windows, the nudge escalates:

| Level | Trigger | Behavior |
|---|---|---|
| 0 (soft) | First detection | Asks the agent to reflect — "are you going in circles?" |
| 1 (medium) | Still stuck after cooldown | Demands a 3-step explicit diagnosis before the next tool call |
| 2 (hard) | Still stuck after two cooldowns | STOP directive — no tool calls until root cause is stated |

`nudgeLevel` resets to -1 when the MLP score drops below `threshold × 0.94`, even during a cooldown window — the next detection will be silently absorbed before firing at level 0 again.

## Per-Step MLP Stuck Detector (v5)

A 5,569-parameter per-step MLP trained on ~306K labeled steps from real Claude Code sessions. Uses an N=5 ring buffer of historical features and previous scores — the model sees the current step alongside the 5 preceding steps, giving it direct access to the repetition signal without windowing heuristics.

### Architecture

```
Input: [features_T(8), features_T-1(8), ..., features_T-5(8), score_T-1, ..., score_T-5]
       = 8 × 6 + 5 = 53 floats

Linear(53, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 1) → Sigmoid
```

- **Features per step (8):** `tool_idx`, `cmd_hash`, `file_hash`, `output_similarity`, `has_prior_output`, `output_length`, `is_error`, `step_index_norm`
- **Ring buffer depth:** N=5 (covers 85.4% of observed stuck loops)
- **Score dims:** Not normalized — left in [0,1] to avoid train/inference mismatch (trimodal {0, 0.5, 1} training labels vs continuous sigmoid at inference)
- **Size:** ~60 KB JSON weights, runs in pure JS (no Python, no GPU)

### Results

**Test set** (held-out sessions from training data):

| Metric | Value |
|---|---|
| Precision | 97.2% |
| Recall | 95.1% |
| F1 | 0.961 |
| Threshold | 0.5 |
| Parameters | 5,569 |
| Weights | ~60 KB |

**Score distribution on the test set** (n=13,679 STUCK, n=18,116 PRODUCTIVE):

| Percentile | STUCK | PRODUCTIVE |
|---|---|---|
| p50 | 1.000 | 0.014 |
| p75 | 1.000 | 0.056 |
| p90 | 1.000 | 0.163 |
| p95 | 1.000 | 0.297 |
| p99 | 1.000 | 0.661 |

The median STUCK step scores 1.000; 95% of productive steps score below 0.297. The threshold of 0.5 sits well inside the gap between the two distributions.

**Improvement over v4 (windowed CNN):**

| Model | Architecture | Params | F1 | Recall |
|---|---|---|---|---|
| v4 CNN | Conv1d, 10-step windows | 2,605 | 0.908 | 0.859 |
| v5 MLP (this) | Per-step + ring buffer | 5,569 | 0.961 | 0.951 |

## Data Pipeline

### Labeling Pipeline

```
Claude Code sessions (.jsonl from HuggingFace or local)
    │
    │  python generate.py [datasets/source_dir/]
    ▼
Parse sessions → extract 8 per-step features (schema_version=3)
    │
    ▼
Submit to Anthropic Batch API (claude-sonnet-4-6)
    │  Parse failures escalate to claude-opus-4-6
    ▼
Per-step labels: PRODUCTIVE / STUCK / UNSURE
    │
    ▼
Merge features + labels into JSONL training rows
    │
    ▼
data/generated/<source>_v3.jsonl
```

**Privacy guarantee:** Only numeric features (tool indices, CRC32 hashes, similarity scores) appear in the final JSONL. Raw commands, file paths, and output text are used only during feature extraction and never written to the training files.

### Training Pipeline

```bash
# Generate features and labels for all sources in training_manifest.json
python generate.py

# Re-extract features without re-labeling (useful after schema changes)
python generate.py --skip-labeling

# Train the v5 MLP
python train.py

# Outputs:
#   proxy/stuck_checkpoint.pt   — PyTorch checkpoint
#   proxy/stuck_weights.json    — JSON weights for JS inference
#   proxy/stuck_config.json     — Config + metrics
```

`generate.py` reads source directories from `training_manifest.json` by default. Pass one or more positional arguments to override (e.g. `python generate.py datasets/new_source/`).

### `training_manifest.json`

```json
{
  "schema_version": 3,
  "datasets": [
    {"source_dir": "datasets/nlile/",          "path": "data/generated/nlile_v3.jsonl",          "weight": 1.0},
    {"source_dir": "datasets/dataclaw_claude/", "path": "data/generated/dataclaw_claude_v3.jsonl", "weight": 1.0},
    {"source_dir": "datasets/masterclass/",    "path": "data/generated/masterclass_v3.jsonl",    "weight": 1.0},
    {"source_dir": "datasets/claudeset/",      "path": "data/generated/claudeset_v3.jsonl",      "weight": 1.0}
  ]
}
```

`weight` controls physical oversampling (integer duplication).

### Datasets

| Dataset | Source | Sessions | Steps | Role |
|---|---|---|---|---|
| [nlile/misc-merged](https://huggingface.co/datasets/nlile/misc-merged) | HuggingFace | 4,973 | 295,493 | Primary — broad distribution |
| [DataClaw](https://huggingface.co/datasets/DataClaw) (woctordho) | HuggingFace | 26 | 1,963 | Thinking blocks |
| masterclass | HuggingFace | 58 | 5,531 | Mixed sessions |
| claudeset | HuggingFace | 39 | 2,961 | Mixed sessions |

### Feature Set (8 per-step features)

| Feature | Signal |
|---|---|
| `tool_idx` | Tool type (bash/edit/view/search/create/submit/other) |
| `cmd_hash` | CRC32 of semantic command key (`base_cmd:target_file`), normalized to [0,1) |
| `file_hash` | CRC32 of file path, normalized to [0,1) |
| `output_similarity` | Jaccard similarity of current output vs. previous run of same command |
| `has_prior_output` | Whether this exact command has been run before in the session |
| `output_length` | `log1p(output_line_count)` |
| `is_error` | Error indicators in output |
| `step_index_norm` | Position in session, normalized to [0,1] |

The ring buffer provides the repetition signal directly: if `cmd_hash` at T equals `cmd_hash` at T-2, the MLP sees identical values in the current features and in history slot T-2. No windowing or aggregation needed.

## Key Findings

1. **Per-step with history beats sliding windows.** Moving from a 10-step windowed CNN (2,605 params, F1=0.908) to a per-step MLP with N=5 ring buffer (5,569 params, F1=0.961) improved recall by 9.2 points (0.859→0.951). The ring buffer exposes the repetition signal directly rather than aggregating it away.

2. **Stuck loop analysis.** 65% of observed stuck runs are single-step (the agent repeats one command). N=5 history covers 85.4% of multi-step stuck loops, making it the optimal depth without adding noisy padding for rare deep loops.

3. **Score dims must not be normalized.** Training labels are trimodal {0, 0.5, 1}; inference scores are continuous sigmoid outputs. Normalizing the score dims with training-set statistics causes a train/inference mismatch — kept in raw [0,1] space.

4. **CRC32 identity is learnable.** Hash values aren't meaningful as magnitudes, but when the same command appears in consecutive history slots, the MLP sees near-zero differences between those positions — a learnable equality signal without explicit equality features.

5. **Every STUCK label is LLM-verified.** Sonnet labels parse failures as UNSURE; those escalate to Opus. PRODUCTIVE and STUCK labels are never assigned by heuristic alone.

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

This work combines: proxy-based interception, tool-call behavioral features, a trained per-step MLP with ring buffer history, and corrective nudge injection — all running in pure JavaScript inside the proxy with no Python runtime. The cleanest comparison point is `strongdm/attractor` which uses similar behavioral signals but no ML.

## Next Steps

1. Run a held-out benchmark to validate v5 generalization on out-of-distribution code (expected: similar improvement over v4 benchmark F1=0.714)
2. Feature ablation study on the 8 features — `step_index_norm` (train/inference mismatch) and `has_prior_output` are the first candidates for removal
3. Collect more sessions with extended thinking blocks — DataClaw (26 sessions) is the only current source

## License

MIT for all code in this repo. Claude Code is under Anthropic's license — the proxy does not modify or redistribute it.
