# Data Generation and Training Pipeline

## Overview

The pipeline is fully decoupled into independent stages. Labels, features, and
training are separate concerns that can evolve independently and be rerun in
isolation. All stages are idempotent — safe to interrupt and resume.

```
Raw sessions
    │
    ├── [fetch.json]  ──────────────────────── download / locate raw data
    ├── [filter.json] ──────────────────────── select sessions for labeling
    │
    ├── src/label_session.py  ──────────────── per-step labels  (Sonnet via claude -p)
    ├── src/extract_features.py  ───────────── per-step features (pure computation)
    │
    ├── src/merge_session.py  ──────────────── zip labels + features → training rows
    │
    └── src/orchestrate.py  ────────────────── drives all of the above
            │
            └── data/generated/<source>_v<N>.jsonl   (one file per source)
                        │
                        └── [training_manifest.json]
                                    │
                                    └── src/train.py
```

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

Labels one session using `claude -p` with full sequential context.

**Input:** path to a human-readable session transcript (JSON array of steps
with tool name, cmd, file, output snippet)

**Output:** `data/labels/<source>/<session_id>_labels.json`

**Idempotent:** if the label file already exists, skip. Pass `--force` to
re-label.

**Prompt contract:**
- System prompt explains the PRODUCTIVE / STUCK / UNSURE definitions
- User message contains the full transcript + explicit step count
- Instructs the model: "output a JSON array of exactly N strings"
- Python validates `len(labels) == n_steps` before writing; retries once on
  mismatch before marking the session as failed

**CLI:**
```
python src/label_session.py <session_transcript_path> --source <name> --out data/labels/
```

---

### `src/extract_features.py`

Computes per-step numeric features from a raw session. No LLM calls.

**Input:** raw session file (source-specific format, parsed via source adapter)

**Output:** `data/features/<source>/<session_id>_features.json`

**Idempotent:** skip if feature file exists and `schema_version` matches
current. Re-extract automatically if version is stale.

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

### `src/orchestrate.py`

Drives the full pipeline for a list of source directories. This is the main
entry point for dataset generation.

**Inputs:**
- One or more source directories, each containing `fetch.json` and
  `filter.json`
- `--workers N` for parallel labeling (default: 5)
- `--force-relabel` to ignore existing label files
- `--schema-version N` to trigger feature re-extraction for stale files

**Behavior:**
1. Read `fetch.json` — download raw sessions if not already present
2. Read `filter.json` — select sessions matching criteria
3. For each selected session (in parallel up to `--workers`):
   - Generate transcript if not cached
   - Run `label_session.py` (skip if label file exists)
   - Run `extract_features.py` (skip if feature file is current)
   - Run `merge_session.py`
4. Write `data/generated/<source>_v<N>.jsonl`
5. Update `progress.json` with per-session status (done / pending / failed)

**Progress tracking** — `data/generated/<source>_progress.json`:
```json
{
  "total": 5000,
  "done": 3241,
  "failed": 12,
  "pending": 1747,
  "failed_sessions": ["nlile_abc...", ...]
}
```

**CLI:**
```
python src/orchestrate.py datasets/nlile/ datasets/dataclaw/ --workers 10
```

---

## Session Selection

### `datasets/<source>/filter.json`

```json
{
  "min_steps": 30,
  "max_steps": 200,
  "languages": ["rust", "python", "typescript", "javascript", "c", "cpp"],
  "max_per_language": 1000,
  "max_sessions": 5000,
  "require_stuck": false
}
```

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

For proprietary pre-labeled datasets (no raw sessions available):
```json
{
  "type": "labeled_gz",
  "path": "data/sources/work_embedded_c_labeled.jsonl.gz",
  "description": "Proprietary embedded C sessions — pre-labeled, raw sessions not available"
}
```

When `type` is `labeled_gz`, the orchestrator skips labeling and feature
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
- Per-step classification with temporal context
- Input: step features + last N window scores + last N fc1 activations
- Drop `steps_since_*` features (replaced by temporal memory)
- Small RNN or MLP temporal head on top of CNN
- ~10,000–15,000 params, still fits in L2 cache
- Training data: per-step labels from this pipeline (not windowed)

The per-step label format produced by this pipeline is directly compatible
with both window-based and per-step model training — windowed training just
groups consecutive step labels into a window label (majority vote or max).
