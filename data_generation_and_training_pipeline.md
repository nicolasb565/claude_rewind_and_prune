# Data Generation and Training Pipeline

## Quickstart (reproducing the model from scratch)

```bash
# Step 1 — generate labeled dataset (fetch → label → extract → merge)
# Cost calibration first: runs 5 sessions and prints token/cost estimate
python generate.py --max-sessions 5 --dry-run-estimate

# Then run for real (idempotent — safe to re-run, resumes from where it left off)
# Requires ANTHROPIC_API_KEY for the Batch API labeling step
ANTHROPIC_API_KEY=... python generate.py

# Step 2 — train and evaluate
python train.py --manifest training_manifest.json
```

`generate.py` submits labeling requests to the Anthropic Batch API (async, up to 24h),
then extracts features and merges into training files once labels are retrieved.
Re-running after an interruption or credit-limit hit resumes automatically.

`train.py` reads the manifest, trains the model, and runs benchmark evaluation.
Re-run this as many times as needed when tuning architecture or hyperparameters —
it never touches the labeled dataset.

---

## Overview

The pipeline is fully decoupled into independent stages. Labels, features, and
training are separate concerns that can evolve independently and be rerun in
isolation. All stages are idempotent — safe to interrupt and resume.

```
generate.py                         ← entry point for dataset generation
    │
    ├── [fetch.json]  ────────────── download / locate raw sessions
    ├── [filter.json] ────────────── select sessions
    │
    ├── pipeline/batch_label.py ──── per-step labels via Anthropic Batch API
    ├── pipeline/extract_features.py  per-step features (pure computation)
    ├── pipeline/merge_session.py ─── zip labels + features → training rows
    │
    └── data/generated/<source>_v<N>.jsonl
                │
                └── training_manifest.json
                            │
train.py                    └──────── entry point for training + eval
    │
    ├── training/train.py
    └── training/eval_benchmark.py
```

---

## Data Sources

All sources must be Claude Code sessions exclusively. Non-Claude sessions (Gemini,
GPT, open-source models) use different tool call patterns and thinking block
structures — mixing them adds noise and degrades generalization. The existing
`dataclaw` source (woctordho/dataclaw, 453 sessions) is 85% non-Claude and is
**replaced** by the two sources below.

| Source | Type | Sessions | Primary languages | Notes |
|---|---|---|---|---|
| `nlile` | parquet (internal) | ~5,000 | Rust 96% | hyperswitch codebase |
| `dataclaw_claude` | huggingface | 69 | C 40%, C++ 19%, Python 18% | Claude-only filter of woctordho/dataclaw |
| `masterclass` | huggingface | 141 | Python 56%, HTML 10% | gutenbergpbc/john-masterclass-cc, ML research |
| `claudeset` | huggingface | 101 | PHP 41%, Java 13%, JS 11%, TS 2.5% | lelouch0110/claudeset-community; filter out 13 synthetic sessions |
| `work_embedded_c` | labeled_gz | ~2,590 | C (embedded) | proprietary, pre-labeled, no raw sessions |

**Language coverage summary:** Rust (nlile), C/C++ (dataclaw_claude + work_embedded_c),
Python (dataclaw_claude + masterclass), PHP/Java/JS/TS (claudeset). Missing: Go —
need new sources when available.

**fetch.json for `dataclaw_claude`:**
```json
{
  "type": "huggingface",
  "repo": "woctordho/dataclaw",
  "split": "train",
  "parser": "dataclaw",
  "model_filter": ["anthropic/claude-opus-4-6", "claude-opus-4-6",
                   "anthropic/claude-sonnet-4-6", "claude-opus-4-5-20251101",
                   "claude-haiku-4-5-20251001", "openrouter/anthropic/claude-opus-4.6"],
  "description": "Claude-only sessions from woctordho/dataclaw (69/453 sessions)"
}
```

**fetch.json for `masterclass`:**
```json
{
  "type": "huggingface",
  "repo": "gutenbergpbc/john-masterclass-cc",
  "split": "train",
  "parser": "dataclaw",
  "description": "Claude Code ML research sessions — Python-heavy (gutenbergpbc/john-masterclass-cc)"
}
```

**fetch.json for `claudeset`:**
```json
{
  "type": "huggingface",
  "repo": "lelouch0110/claudeset-community",
  "split": "train",
  "parser": "claudeset",
  "model_filter": ["claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929",
                   "claude-sonnet-4-6"],
  "description": "Claude Code sessions — PHP/Java/JS coverage (lelouch0110/claudeset-community)"
}
```

The `model_filter` field in fetch.json is an optional allowlist of model strings.
The orchestrator skips sessions whose `model` field is not in the list (filters out
`<synthetic>` and any non-Claude entries). Omit the field to accept all sessions.

The `claudeset` source uses a different schema than `dataclaw` — turns are structured
as `{type: exchange|compact, user, assistant: {thinking, text, tool_calls}}` with
`tool_calls: [{tool, input, output}]`. Requires its own parser (`parse_claudeset.py`).

---

## Directory Layout

```
datasets/
  <source>/
    fetch.json          # how to obtain the raw sessions (URL, format, parser)
    filter.json         # which sessions to include (see Session Selection)

data/
  raw/<source>/         # raw session files (downloaded or symlinked)
  labels/<source>/      # per-session label files  (<session_id>_labels.json)
  features/<source>/    # per-session feature files (<session_id>_features.json)
  generated/            # merged training files (<source>_v<N>.jsonl)

training_manifest.json  # explicit list of generated files used for training
```

---

## Data Formats

### Per-session label file — `data/labels/<source>/<session_id>_labels.json`

```json
{
  "session_id": "nlile_26b99063-...",
  "source": "nlile",
  "n_steps": 53,
  "labeler": "claude-sonnet-4-6",
  "labeled_at": "2026-04-12T00:00:00Z",
  "labels": ["PRODUCTIVE", "PRODUCTIVE", "STUCK", ...]
}
```

Labels are one of `"PRODUCTIVE"`, `"STUCK"`, `"UNSURE"`. Array length must
equal `n_steps` — validated before writing.

### Per-session feature file — `data/features/<source>/<session_id>_features.json`

```json
{
  "session_id": "nlile_26b99063-...",
  "source": "nlile",
  "schema_version": 1,
  "n_steps": 53,
  "extracted_at": "2026-04-12T00:00:00Z",
  "steps": [
    {
      "tool_idx": 2,
      "steps_since_same_tool": 0.0,
      "output_similarity": 0.0,
      "is_error": 0.0,
      ...
    },
    ...
  ]
}
```

`schema_version` is bumped whenever the feature schema changes. The merge
step rejects mismatched versions and the orchestrator can trigger re-extraction
for stale files.

### Merged training file — `data/generated/<source>_v<N>.jsonl`

One JSON object per step, one line per step:

```json
{"session_id": "nlile_26b99063-...", "step": 0, "tool_idx": 2, "output_similarity": 0.0, ..., "label": 0.0}
{"session_id": "nlile_26b99063-...", "step": 1, "tool_idx": 1, "output_similarity": 0.3, ..., "label": 0.0}
{"session_id": "nlile_26b99063-...", "step": 2, "tool_idx": 2, "output_similarity": 0.9, ..., "label": 1.0}
```

Label encoding: `PRODUCTIVE=0.0`, `UNSURE=0.5`, `STUCK=1.0`.

---

## Modules

### `src/label_session.py`

Labels one session using the Anthropic SDK (Message Batches API).

**Input:** path to a human-readable session transcript (JSON array of steps
with tool name, cmd, file, output snippet)

**Output:** `data/labels/<source>/<session_id>_labels.json`

**Idempotent:** if the label file already exists, skip. Pass `--force` to
re-label.

**Prompt contract:**
- System prompt explains the PRODUCTIVE / STUCK / UNSURE definitions — keep
  short (~500 tokens); it is sent with every request
- User message contains the full transcript + explicit step count; tool outputs
  truncated to 500 chars to minimize input tokens
- Output format: compact CSV `P,S,U,P,P,S` (single chars, comma-separated) —
  ~7x fewer output tokens than JSON array of strings
- Python splits on commas, maps `P→PRODUCTIVE`, `S→STUCK`, `U→UNSURE`
- Validates `len(labels) == n_steps` before writing

**API strategy:**
- Uses Anthropic Message Batches API (50% discount, separate rate limits)
- One batch request per session — Sonnet sees full sequential context
- Batches submitted via `src/batch_label.py` (see Orchestrator section)
- Output tokens are cheap; input token compression is the main cost lever

**Cost estimate (validated by experiment):**

Compression experiment on the 53-step pilot session:
- Uncompressed transcript: ~41,740 tokens
- Compressed transcript (500-char output cap): ~5,601 tokens — **87% reduction**
- Label quality: **identical** — same PRODUCTIVE/STUCK split, same transition point
- Subagent call total: 17,635 tokens, but ~11,900 of those were Claude Code system
  prompt overhead (tool definitions etc.) — not present in Batch API calls

With a minimal system prompt (~500 tokens) as used in the Batch API:
- Per session: ~500 (system) + ~5,600 (transcript) + ~125 (output) ≈ **6,225 tokens**
- Per step: ~**120 tokens/step** (vs 338 in the uncompressed pilot)
- All sources: ~71,000 steps × 120 tokens = **~8.5M input tokens**
- At batch pricing ($1.50/MTok input, $7.50/MTok output): **~$13 total**
- Run 5 sessions first to calibrate actual cost before committing credits

**CLI:**
```
python src/label_session.py <session_transcript_path> --source <name> --out data/labels/
```

---

### `src/extract_features.py`

Computes per-step numeric features from a raw session. No LLM calls.

**Input:** raw session file (source-specific format, parsed via source adapter)

**Output:** `data/features/<source>/<session_id>_features.json`

**Idempotent:** skip if feature file exists, `schema_version` matches current,
and `n_steps` in the file matches the session length (guards against partial
writes from interrupted runs). Re-extract if any condition fails.

**CLI:**
```
python src/extract_features.py <raw_session_path> --source <name> --out data/features/
```

---

### `src/merge_session.py`

Merges label file and feature file for one session into training rows.

**Validation:**
- Assert `n_steps` matches between label and feature files
- Assert `schema_version` is current
- Assert label array length == feature step count

**Output:** appends to `data/generated/<source>_v<N>.jsonl`

**CLI:**
```
python src/merge_session.py --labels <path> --features <path> --out <path>
```

---

### `src/batch_label.py`

Submits pending sessions to the Anthropic Message Batches API and retrieves
results. Decoupled from the orchestrator so it can be run independently.

**Behavior:**
1. Scan `data/labels/<source>/` — collect sessions without a complete label file
2. Generate transcripts for pending sessions (tool outputs truncated to 500 chars)
3. Submit up to 10,000 requests per batch (one request per session)
4. Poll until batch completes (up to 24 hours) or save `batch_id` to
   `data/labels/<source>/pending_batch.json` and exit — re-running resumes
5. On completion, parse each response (CSV `P,S,U,...`), validate
   `len(labels) == n_steps`, write label file
6. Sessions that fail validation are marked `failed` and excluded from the
   batch result — re-running will resubmit them

**Resume / rate-limit safety:**
- If a batch is already in-flight (`pending_batch.json` exists), skip
  submission and go straight to polling/retrieval
- Sessions with existing complete label files are never resubmitted
- If the API credit limit is hit mid-batch, the batch continues server-side
  (Anthropic processes it async); re-running after topping up retrieves results

**Cost calibration run:**
```
python src/batch_label.py datasets/nlile/ --max-sessions 5 --dry-run-estimate
```
Prints estimated token count and cost for 5 sessions before submitting,
so you can calibrate the full-run cost before committing credits.

**CLI:**
```
python src/batch_label.py datasets/nlile/ datasets/dataclaw_claude/
python src/batch_label.py datasets/nlile/ --max-sessions 5   # cost calibration
```

---

### `src/orchestrate.py`

Drives the full pipeline for a list of source directories. This is the main
entry point for dataset generation.

**Inputs:**
- One or more source directories, each containing `fetch.json` and
  `filter.json`
- `--force-relabel` to ignore existing label files
- `--schema-version N` to trigger feature re-extraction for stale files

**Behavior:**
1. Read `fetch.json` — download raw sessions if not already present
2. Read `filter.json` — select sessions matching criteria
3. Run `batch_label.py` — submit/retrieve labels for all pending sessions
4. For each labeled session:
   - Run `extract_features.py` (skip if feature file is current and complete)
   - Run `merge_session.py`
   - Mark session as `done` in progress file immediately on success
   - Mark session as `failed` on error, continue with remaining sessions
5. Write `data/generated/<source>_v<N>.jsonl` from all completed sessions
6. Print summary: done / failed / pending

**Resume behavior:** re-running with the same arguments resumes from where
it left off. Sessions with existing label files and feature files are skipped.
In-flight batches are detected via `pending_batch.json` and polled rather than
resubmitted. Pass `--retry-failed` to resubmit previously failed sessions.

**Progress tracking** — `data/generated/<source>_progress.json`:
```json
{
  "total": 5000,
  "done": 3241,
  "failed": 12,
  "pending": 1747,
  "failed_sessions": [
    {"session_id": "nlile_abc...", "error": "label mismatch: got 51 labels for 53 steps"},
    ...
  ]
}
```

**CLI:**
```
python src/orchestrate.py datasets/nlile/ datasets/dataclaw_claude/
```

---

## Session Selection

### `datasets/<source>/filter.json`

```json
{
  "min_steps": 30,
  "max_steps": 200,
  "max_sessions": 5000,
  "folder_limits": [
    {"pattern": "nlile_parquet/data/train-0000*", "max": 500},
    {"pattern": "nlile_parquet/data/train-0001*", "max": 500}
  ]
}
```

**On language filtering (spike result — nlile):** language detection from file
extensions in tool call arguments is feasible: ~96.3% of sessions are
detectable, with only 1.7% undetectable. However, nlile is 96.3% Rust
(the hyperswitch codebase), 1.9% JavaScript, 0.1% Python. There is essentially
nothing non-Rust to filter to — `languages` is not a useful field for this
source. For language diversity beyond Rust, entirely new data sources are
required (Python, TypeScript, C++ Claude Code sessions).

**Recommended approach for nlile:** use `folder_limits` with glob patterns on
parquet filenames to select a coarse subset, or `max_sessions` with random
sampling. For sources where sessions map to known repos or folders, a
`folder_limits` list provides the most control over which subset is included.

Selection is applied before labeling. The orchestrator logs how many sessions
were filtered out and why.

### `datasets/<source>/fetch.json`

```json
{
  "type": "parquet",
  "path": "data/separate/nlile_parquet/data/",
  "parser": "nlile",
  "description": "Anthropic internal Claude Code sessions"
}
```

For public datasets:
```json
{
  "type": "huggingface",
  "repo": "owner/dataset-name",
  "split": "train",
  "parser": "nlile",
  "description": "Public Claude Code session dataset"
}
```

For proprietary datasets where raw sessions cannot be shared:
```json
{
  "type": "proprietary",
  "path": "data/separate/work_embedded_c/",
  "parser": "dataclaw",
  "artifact": "data/sources/work_embedded_c_labeled.jsonl.gz",
  "description": "Proprietary embedded C sessions — raw sessions local only"
}
```

When `type` is `proprietary`, `generate.py` runs the full pipeline (label +
extract features) using the local raw sessions, then writes a `.gz` artifact
to the path specified in `artifact`. The artifact contains one row per step
with labels + features baked in — no raw session content. It can be committed
to the repo and shared without exposing the raw sessions.

Re-running `generate.py` skips sessions already present in the artifact.

For consuming a pre-existing artifact (no raw sessions available):
```json
{
  "type": "labeled_gz",
  "path": "data/sources/work_embedded_c_labeled.jsonl.gz",
  "description": "Proprietary embedded C sessions — pre-labeled, raw sessions not available"
}
```

When `type` is `labeled_gz`, `generate.py` skips labeling and feature
extraction entirely. The pipeline is:
1. Decompress `.gz` → read existing labeled rows
2. Check `schema_version` of each row against current version
3. If stale → run feature migration (see below)
4. Write to `data/generated/<source>_v<N>.jsonl`

`filter.json` is ignored for `labeled_gz` sources — the dataset is fixed.

---

## Feature Migration

When the feature schema changes (new feature added, feature dropped, normalization
changed), existing labeled `.gz` files and cached feature files need migration
rather than full re-extraction.

### `src/migrate_features.py`

Applies incremental migrations to bring rows from any older `schema_version`
to the current version. Each version bump has a corresponding migration
function:

```python
MIGRATIONS = {
    # (from_version, to_version): migration_fn
    (1, 2): migrate_v1_to_v2,   # e.g. add self_similarity, drop output_diversity
    (2, 3): migrate_v2_to_v3,   # e.g. add circular_lang
}
```

A migration function receives a step dict and returns an updated step dict.
Migrations are chained — a v1 row runs `v1→v2` then `v2→v3` to reach v3.

**Key constraint:** migrations for `labeled_gz` sources can only add or
transform features computable from existing fields. If a new feature requires
raw session data (e.g. full output text), it cannot be migrated for proprietary
sources — those features are unavailable. The migration must either:
- Compute a reasonable default (e.g. `0.0`)
- Mark the feature as `null` and let the training script handle it
- Document the limitation in the migration function

**CLI:**
```
python src/migrate_features.py data/sources/work_embedded_c_labeled.jsonl.gz --to-version 2
```

---

## Training Manifest — `training_manifest.json`

```json
{
  "schema_version": 1,
  "datasets": [
    {"path": "data/generated/nlile_v1.jsonl",         "weight": 1.0},
    {"path": "data/generated/dataclaw_v1.jsonl",       "weight": 5.0},
    {"path": "data/generated/work_embedded_c_v1.jsonl","weight": 3.0}
  ]
}
```

`weight` controls oversampling at training time. The training script reads
this file exclusively — it has no knowledge of sources or filters.

---

## Training Script — `src/train.py`

Reads `training_manifest.json`, loads all datasets with their weights, trains
the model.

- Weighted sampling applies `weight` as an oversample multiplier
- Outputs model artifacts to `proxy/` (weights, config, checkpoint)
- Logs train/test split metrics and benchmark eval

```
python src/train.py --manifest training_manifest.json
```

---

## Model Architecture Roadmap

### Current (v4)
- Per-window classification (10-step sliding window, stride 5)
- CNN: tool_embed → conv1d(k=3,16) + conv1d(k=5,16) → maxpool → fc(16) → sigmoid
- ~3,100 params, ~22 µs/window inference in JS

### Next (v5) — temporal
- Per-step classification, no windowing
- Input at step T: step features(T) + [score(T-1)...score(T-N)] + [features(T-1)...features(T-N)]
- Previous scores and features carried in a small ring buffer — free to compute,
  negligible memory
- Drop `steps_since_*` features (replaced by direct access to previous step features)
- Architecture: MLP or small CNN over the concatenated current + N previous
  feature vectors, single sigmoid output per step
- ~10,000–15,000 params, still fits in L2 cache, sub-microsecond per step
- Training: sequences of (features, label) pairs per session; loss computed
  per step; teacher forcing during training (ground truth previous labels as
  input rather than model predictions)
- Inference: stateful — model keeps a ring buffer of its own previous outputs
  and feature vectors, updated after each tool call
- Exposure bias mitigation: previous scores fed into the ring buffer are
  quantized to the nearest of {0.0, 0.5, 1.0} (thresholds at 0.25 and 0.75)
  before being stored, ensuring identical input distribution at training and
  inference — ground truth labels are already in this set, so training is
  unaffected

The per-step label format from this pipeline maps directly to this architecture
with no additional processing.

---

## Codebase Cleanup

When the new pipeline is implemented and validated, delete the old window-based
pipeline files. They are replaced entirely by the new modules.

### Files to delete

| File | Replaced by |
|---|---|
| `src/label_sessions.py` | `src/batch_label.py` + `src/label_session.py` + `src/parse_nlile.py` |
| `src/run_review.py` | `src/batch_label.py` |
| `src/review_sonnet.py` | `src/batch_label.py` |
| `src/review_opus.py` | `src/batch_label.py` (no Opus escalation in new pipeline) |
| `src/sample_productive.py` | one-off audit script, no longer needed |
| `src/migrate_labels.py` | window label migration, not applicable to per-step labels |
| `src/migrate_add_prior_output.py` | superseded by `src/migrate_features.py` |
| `src/migrate_fix_features.py` | superseded by `src/migrate_features.py` |
| `src/merge_sources.py` | replaced by `src/merge_session.py` + `src/orchestrate.py` |
| `src/abstract_trajectory.py` | windowing logic, not needed for per-step pipeline |

### Files to keep and update

| File | Status |
|---|---|
| `src/parse_dataclaw.py` | keep — reused as parser for dataclaw_claude + masterclass |
| `src/eval_benchmark.py` | keep — benchmark evaluation unchanged |
| `src/train_cnn_oversample.py` | rename → `src/train.py`, update for per-step format |

### New files to create

```
src/
  pipeline/               # Python package — dataset generation + labeling
    __init__.py
    batch_label.py        # Batch API submission, polling, retrieval
    label_session.py      # transcript formatter, label file writer
    extract_features.py   # per-step numeric feature extraction
    merge_session.py      # zip labels + features → training JSONL rows
    migrate_features.py   # chained schema version migrations
    parsers/
      __init__.py
      nlile.py            # nlile parquet parser (from label_sessions.py)
      dataclaw.py         # reuse/refactor src/parse_dataclaw.py
      claudeset.py        # claudeset-community parser (different turn schema)
  training/               # Python package — model training + evaluation
    __init__.py
    train.py              # renamed/updated from train_cnn_oversample.py
    eval_benchmark.py     # unchanged

generate.py               # entry point: fetch → label → extract → merge
train.py                  # entry point: load manifest → train → eval
```

### Cleanup timing

Do not delete old files until the new pipeline has produced a complete labeled
dataset and a trained model that meets or exceeds v4 benchmark performance.
Run old and new in parallel during transition if needed.
