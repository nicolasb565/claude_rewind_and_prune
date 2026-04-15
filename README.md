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

A 4,865-parameter per-step MLP trained on ~306K labeled steps from real Claude Code sessions. Uses an N=5 ring buffer of historical features — the model sees the current step alongside the 5 preceding steps' features, giving it direct access to the repetition signal without windowing heuristics.

### Architecture

```
Input: [features_T(7), features_T-1(7), ..., features_T-5(7)]
       = 7 × 6 = 42 floats

Linear(42, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 1) → Sigmoid
```

- **Features per step (7):** `tool_idx`, `cmd_hash`, `file_hash`, `output_similarity`, `has_prior_output`, `output_length`, `is_error`
- **Ring buffer depth:** N=5 (covers 85.4% of observed stuck loops)
- **Reproduce:** `python src/training/train.py --no-score-history --exclude-feature step_index_norm` (seed=42 default)
- **Size:** ~100 KB JSON weights, runs in pure JS (no Python, no GPU)

### Results (multi-seed)

5-seed mean ± std on the held-out test set, threshold=0.5:

| Metric | Value |
|---|---|
| F1 | 0.9548 ± 0.0054 |
| Precision | 0.9650 ± 0.0065 |
| Recall | 0.9447 ± 0.0064 |
| Parameters | 4,865 |
| Input dim | 42 |

**Score distribution on the test set** (seed=42 production model, n=13,679 STUCK, n=18,116 PRODUCTIVE):

| Percentile | STUCK | PRODUCTIVE |
|---|---|---|
| p50 | 0.999 | ~0.01 |
| p95 | 1.000 | ~0.27 |
| p99 | 1.000 | ~0.66 |

The median STUCK step scores ~1.0; 95% of productive steps score well below the threshold. The threshold of 0.5 sits well inside the gap between the two distributions.

**Model evolution:**

| Model | Architecture | Params | Input dim | F1 (mean) |
|---|---|---|---|---|
| v4 CNN | Conv1d, 10-step windows | 2,605 | — | 0.908 |
| v5 MLP (53-dim) | Per-step + ring + score feedback | 5,569 | 53 | 0.961 (teacher-forced eval, inflated) |
| v5 MLP (48-dim) | Per-step + ring buffer | 5,249 | 48 | 0.9559 ± 0.0046 |
| **v5 MLP (current)** | **48-dim minus step_index_norm** | **4,865** | **42** | **0.9548 ± 0.0054** |

### Why we dropped two feature groups

**Score feedback (5 dims)** — an earlier variant fed the model's own previous 5 sigmoid scores back as input. Training filled those slots with ground-truth labels (teacher forcing); inference would fill them with continuous sigmoid output that the model never saw during training. Multi-seed ablation showed F1 = 0.9559 ± 0.0046 with score feedback vs 0.9548 ± 0.0054 without — statistically equivalent. We dropped it to eliminate the train/inference distribution mismatch and shrink the model.

**`step_index_norm`** — at training time this was `step / (total_steps - 1)`; at inference the proxy doesn't know `total_steps` and used `min(step / 100, 1.0)`. A known approximation. Multi-seed ablation showed F1 = 0.9548 ± 0.0054 without it (vs 0.9559 ± 0.0046 with it) — within noise. Dropped to eliminate the second train/inference mismatch and shrink the model further.

The remaining 6 features (`file_hash`, `has_prior_output`, `output_length`, `is_error`, `tool_idx`, `cmd_hash`, `output_similarity`) were all individually tested with 5-seed ablation; every single-feature drop landed within the cross-seed noise range, so none was strictly removable, but none had the train/inference issues that `step_index_norm` and score feedback had. See `src/training/run_ablation.py` and `src/training/compare_ablation.py` to reproduce the matrix.

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

# Train the v5 MLP (current production model — 42-dim, no score feedback, no step_index_norm)
python src/training/train.py --no-score-history --exclude-feature step_index_norm

# Or run the full multi-seed ablation matrix (used to pick the production feature set)
python src/training/run_ablation.py --seeds 5
python src/training/compare_ablation.py

# Outputs (relative to repo root):
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

### Feature Set (7 per-step features)

| Feature | Signal |
|---|---|
| `tool_idx` | Tool type (bash/edit/view/search/create/submit/other) |
| `cmd_hash` | CRC32 of semantic command key (`base_cmd:target_file`), normalized to [0,1) |
| `file_hash` | CRC32 of file path, normalized to [0,1) |
| `output_similarity` | Jaccard similarity of current output vs. previous run of same command |
| `has_prior_output` | Whether this exact command has been run before in the session |
| `output_length` | `log1p(output_line_count)` |
| `is_error` | Error indicators in output |

`step_index_norm` (position in session) was dropped after multi-seed ablation — see Key Findings #3.

The ring buffer provides the repetition signal directly: if `cmd_hash` at T equals `cmd_hash` at T-2, the MLP sees identical values in the current features and in history slot T-2. No windowing or aggregation needed.

## Key Findings

1. **Per-step with history beats sliding windows.** Moving from a 10-step windowed CNN (2,605 params, F1=0.908) to a per-step MLP with N=5 ring buffer (4,865 params, F1=0.9548 ± 0.0054 multi-seed) keeps the recall gains while shrinking the input dimension and eliminating two train/inference mismatches present in earlier variants. The ring buffer exposes the repetition signal directly rather than aggregating it away.

2. **Stuck loop analysis.** 65% of observed stuck runs are single-step (the agent repeats one command). N=5 history covers 85.4% of multi-step stuck loops, making it the optimal depth without adding noisy padding for rare deep loops.

3. **Two train/inference mismatches removed via ablation.** The earliest v5 variant had 53-dim input: 8 features × 6 slots + 5 score feedback dims. Both groups were dropped via multi-seed ablation:
   - **Score feedback (5 dims)**: training filled those with ground-truth labels (teacher forcing); inference would have filled them with continuous sigmoid output the model never saw during training. F1 unchanged within noise.
   - **`step_index_norm` (1 feature × 6 slots)**: training used `step / (total_steps - 1)`; inference can't compute that (no `total_steps`) and approximated as `min(step / 100, 1.0)`. F1 unchanged within noise.

   Final production model: 7 features × 6 slots = 42-dim input, 4,865 params, F1 = 0.9548 ± 0.0054. The remaining 6 features all contribute marginally; no single-feature drop is statistically distinguishable from baseline, but none has a known mismatch problem either, so they all stayed.

4. **CRC32 identity is learnable.** Hash values aren't meaningful as magnitudes, but when the same command appears in consecutive history slots, the MLP sees near-zero differences between those positions — a learnable equality signal without explicit equality features.

5. **Every STUCK label is LLM-verified.** Sonnet labels parse failures as UNSURE; those escalate to Opus. PRODUCTIVE and STUCK labels are never assigned by heuristic alone.

6. **Multi-seed reporting matters.** The first feature-ablation pass used a single seed (42) and showed every removal hurting F1 by ~0.003-0.004 — looked like every feature was contributing. A 5-seed re-run revealed the cross-seed std (~0.005) was *larger* than every single-feature delta — the single-seed signal was noise. This is exactly the seed-hacking failure mode the multi-seed pattern catches.

7. **OOD ceiling is far below in-distribution F1.** On a 10-task OOD benchmark (gcc, llvm, sqlite, django, react, express, lapack, beast, geometry) with Sonnet-labeled per-step ground truth, every classifier we tested plateaus well below its in-distribution F1. The bottleneck is training-data distribution, not model capacity: 97% of training labels come from the `nlile` corpus, and OOD sessions use tools/languages the model never saw. A 9-parameter LR on content features reaches step F1 ≈ 0.33; a 400M-parameter fine-tuned Ettin encoder *loses* to the LR at F1 ≈ 0.26; zero/few-shot Qwen/Phi/Llama 3.8–14B cap around AUC 0.50; a LoRA-fine-tuned phi4-mini collapses to always-P. Capacity doesn't help — it overfits faster. A causal Sonnet labeling experiment (past-only mode on `03_llvm_loop_vec`) recovers only 17/39 stuck labels, establishing an information-theoretic causal ceiling around F1 0.50 for this benchmark, since 22 of the stuck labels depend on future information no causal classifier can see.

8. **Post-filter sweep: median beats linear averaging for step-level precision.** Given the classifier's raw signal is too noisy for hard intervention, trailing-window filters trade recall against precision. Running `benchmarks/lr_filter_sweep.py` across mean, K-of-N, and median aggregators at fine threshold grain produces three clean operating modes on the LR baseline (9 params, 8 content features). `mean-of-2 @ 0.34` preserves the baseline's 7-of-7 session recall with per-step F1 0.369. `median-of-4 @ 0.645` catches 5-of-7 sessions at step P=0.47. `median-of-9 @ 0.605` reaches step P=0.781 and episode F1=0.400 — the only configuration across the entire sweep where per-step TPs meaningfully outnumber FPs (25 vs 7). Intuition: the LR score distribution has asymmetric spike noise — productive steps produce isolated high spikes and stuck episodes produce isolated dips. Linear mean amplifies both; median ignores both by construction.

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

1. Build the OOD benchmark harness (Docker-per-task, 12 tasks, dual-mode auth) to validate v5 generalization on real-world code
2. A/B the nudge strategy on the benchmark: current (silent absorb at 1, soft/medium/hard with cooldowns 1,4,8,8) vs. long-loop focused (raise the silent buffer to ~5 turns, skip soft level)
3. Collect more sessions with extended thinking blocks — DataClaw (26 sessions) is the only current source

## License

MIT for all code in this repo. Claude Code is under Anthropic's license — the proxy does not modify or redistribute it.
