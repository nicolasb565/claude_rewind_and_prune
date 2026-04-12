# Data Generation and Training Pipeline

## Quickstart (reproducing the model from scratch)

```bash
# Step 1 — generate labeled dataset (fetch → label → extract → merge)
# Set up API key first (copy .env.example → .env and fill in your key)
cp .env.example .env   # then edit .env: ANTHROPIC_API_KEY=sk-ant-...

# Cost calibration: runs 5 sessions and prints token/cost estimate
python generate.py --max-sessions 5 --dry-run-estimate

# Then run for real (idempotent — safe to re-run, resumes from where it left off)
python generate.py

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

| Source | HF repo / path | Sessions | Primary languages |
|---|---|---|---|
| `nlile` | internal parquet | ~5,000 | Rust 96% (hyperswitch) |
| `dataclaw_claude` | woctordho/dataclaw | 69 | C 40%, C++ 19%, Python 18% |
| `masterclass` | gutenbergpbc/john-masterclass-cc | 141 | Python 56%, HTML 10% |
| `claudeset` | lelouch0110/claudeset-community | 101 | PHP 41%, Java 13%, JS 11%, TS 2.5% |
| `work_embedded_c` | proprietary .gz artifact | ~2,590 | C (embedded) |

**Language coverage:** Rust (nlile), C/C++ (dataclaw_claude + work_embedded_c),
Python (dataclaw_claude + masterclass), PHP/Java/JS/TS (claudeset). Missing: Go.

**Per-source notes:**

- `dataclaw_claude` — woctordho/dataclaw is 85% non-Claude (Gemini, GPT). Use
  `model_filter` in fetch.json to keep only the 69 Claude sessions. Same schema
  and parser as the original dataclaw.
- `masterclass` — all sessions are Claude; no model filter needed. Uses the same
  schema as woctordho/dataclaw, so the same `dataclaw` parser applies.
- `claudeset` — lelouch0110/claudeset-community contains 13 synthetic sessions
  (model field `<synthetic>`); use `model_filter` to exclude them. Different turn
  schema from dataclaw — requires its own `claudeset` parser. Turn format:
  `{type: exchange|compact, user, assistant: {thinking, text, tool_calls}}` with
  `tool_calls: [{tool, input, output}]`.
- `work_embedded_c` — proprietary C sessions. Raw files exist locally on the
  workstation. On any other machine, consume the committed `.gz` artifact directly
  (`type: labeled_gz` in fetch.json). See Proprietary Dataset Workflow.

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

**Labeling guidance — PRODUCTIVE / STUCK / UNSURE:**

The system prompt defines three labels:

- **PRODUCTIVE** — the step is advancing the work. Exploring new approaches,
  writing code, reading docs, testing a hypothesis for the first time. The model
  is making forward progress, even if that includes occasional errors.
- **STUCK** — the step is part of a recognizable loop. The same command,
  the same error, the same file edit, repeated without new information or a
  changed approach. The work has stopped moving forward.
- **UNSURE** — genuine ambiguity that cannot be resolved from the transcript.
  Use sparingly. UNSURE is not a default — it is a last resort.

**When in doubt, do not default to PRODUCTIVE or STUCK.** If you cannot tell
whether a step is advancing work or repeating a failed approach, label it UNSURE.
UNSURE (encoded as 0.5 during training) provides a calibrated gradient signal
rather than forcing an incorrect binary label.

**Common patterns:**
- First attempt at a command → PRODUCTIVE
- Same command with identical arguments and same error → STUCK (by second or third repeat)
- Exploring a different file or different approach after an error → PRODUCTIVE
- Reading a file already read earlier, with no new information → STUCK
- A pause before a major transition (e.g. reading docs before a big edit) → PRODUCTIVE
- Compiling/testing in a tight loop with the same failure → STUCK

**Transition labeling:** the first step after entering a stuck loop is
still PRODUCTIVE; label the step where repetition begins as STUCK.
The first step after escaping a stuck loop (new approach, new tool, new file)
is PRODUCTIVE again.

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

Supported `type` values:
- `parquet` — local parquet files (nlile)
- `huggingface` — download from HuggingFace Hub
- `proprietary` — local raw sessions that cannot be shared; pipeline runs
  label + extract and writes a `.gz` artifact. Re-running appends new sessions,
  skips sessions already in the artifact.
- `labeled_gz` — consume a pre-existing `.gz` artifact directly; skips labeling
  and feature extraction. `filter.json` is ignored. Pipeline:
  decompress → check `schema_version` → migrate if stale → write to `generated/`.

`model_filter` is an optional allowlist of model name strings. Sessions whose
`model` field is not in the list are skipped. Omit to accept all sessions.

`parser` names the source adapter: `nlile`, `dataclaw`, or `claudeset`.

**Actual fetch.json for each source:**

`datasets/nlile/fetch.json`:
```json
{
  "type": "parquet",
  "path": "data/separate/nlile_parquet/data/",
  "parser": "nlile",
  "description": "Anthropic internal Claude Code sessions (hyperswitch, mostly Rust)"
}
```

`datasets/dataclaw_claude/fetch.json`:
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

`datasets/masterclass/fetch.json`:
```json
{
  "type": "huggingface",
  "repo": "gutenbergpbc/john-masterclass-cc",
  "split": "train",
  "parser": "dataclaw",
  "description": "Claude Code ML research sessions — Python-heavy"
}
```

`datasets/claudeset/fetch.json`:
```json
{
  "type": "huggingface",
  "repo": "lelouch0110/claudeset-community",
  "split": "train",
  "parser": "claudeset",
  "model_filter": ["claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929",
                   "claude-sonnet-4-6"],
  "description": "Claude Code sessions — PHP/Java/JS coverage"
}
```

`datasets/work_embedded_c/fetch.json` (workstation — raw sessions present):
```json
{
  "type": "proprietary",
  "path": "data/separate/work_embedded_c/",
  "parser": "dataclaw",
  "artifact": "data/sources/work_embedded_c_labeled.jsonl.gz",
  "description": "Proprietary embedded C sessions — raw sessions local only"
}
```

`datasets/work_embedded_c/fetch.json` (other machines — artifact only):
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

## Proprietary Dataset Workflow

For datasets where raw sessions exist locally but cannot be shared (e.g.
internal company codebases, client work), the workflow is:

**First time — generating the artifact:**
```bash
# 1. Point fetch.json at local raw sessions with type=proprietary and artifact path
# 2. Run generate.py normally — it labels, extracts features, and writes the .gz
#    (ANTHROPIC_API_KEY must be set in .env)
python generate.py datasets/work_embedded_c/

# 3. Commit the artifact — it contains only labels + features, no raw session content
git add data/sources/work_embedded_c_labeled.jsonl.gz
git commit -m "data: add work_embedded_c labeled artifact"
```

**Subsequently — anyone cloning the repo:**
```bash
# Change fetch.json type to labeled_gz (or keep proprietary — generate.py detects
# that the artifact already covers all sessions and skips labeling)
python generate.py datasets/work_embedded_c/   # reads artifact, skips labeling
```

**When the feature schema changes:**
```bash
# Migrate the artifact in-place to the new schema version
python src/pipeline/migrate_features.py data/sources/work_embedded_c_labeled.jsonl.gz \
    --to-version 2
git add data/sources/work_embedded_c_labeled.jsonl.gz
git commit -m "data: migrate work_embedded_c artifact to schema v2"
```

**Key properties of the `.gz` artifact:**
- One row per step: `{session_id, step, schema_version, label, ...features}`
- Label provenance preserved: `label_source` field (`sonnet`, `heuristic`, etc.)
- No raw session content (no tool outputs, no file contents)
- Committed to the repo — enables reproducibility without raw sessions

**Artifact lifecycle — handling new, deleted, and evolved sessions:**

*New sessions (raw files added since last run):*
`generate.py` compares session IDs in the artifact against sessions found by
the parser. Any session not yet in the artifact is treated as pending and runs
the full label + extract pipeline. Results are appended to the artifact.
Re-running two months later with new raw sessions just works.

*Deleted/lost raw sessions (raw file gone, artifact row present):*
The artifact row is preserved as-is — it is the only surviving record of that
session's labels and features. `generate.py` logs a warning for each missing
raw file but continues normally. The session remains in training data.
Pass `--drop-missing` to explicitly remove orphaned rows from the artifact
(use with caution — labels are lost permanently).

*Feature schema change with raw sessions available:*
Re-extraction from raw sessions gives exact feature values. Run:
```bash
python generate.py datasets/work_embedded_c/ --schema-version 2
```
`generate.py` detects sessions in the artifact with stale `schema_version`,
checks if raw session files exist, and re-extracts features for those sessions.
Labels are preserved from the artifact — only features are recomputed.
Sessions whose raw files are missing fall back to migration (approximate values).

*Feature schema change without raw sessions (any machine):*
Migration is the only option — see Feature Migration section below. Values
requiring raw data receive defaults or `null`.

*Verifying artifact integrity:*
```bash
python src/pipeline/migrate_features.py data/sources/work_embedded_c_labeled.jsonl.gz --verify
```
Checks that every session has consistent `n_steps` across all its rows and
that `schema_version` is uniform. Prints a summary of any corrupt sessions.

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

## Train/Test Split

Split is **session-level**, not step-level. All steps from a given session go
entirely into train or entirely into test — this prevents data leakage where
the model memorizes session-specific patterns rather than generalizing.

**Parameters:** 90% train / 10% test, `random.seed(42)`.

The split is computed by `train.py` at training time from the session IDs in
the manifest. It is not baked into the `.jsonl` files. This means:

- Adding new sessions to a source and re-merging does not invalidate the split
  for existing sessions — the same seed maps them to the same partition
- Session IDs are sorted before splitting to ensure determinism regardless of
  filesystem ordering
- The split is logged at training time: total sessions, train sessions, test sessions,
  STUCK prevalence in each split

**Validation:** `test_training.py` asserts that no session ID appears in both
the train and test sets.

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

## Code Quality

**Formatting — `black`**
All Python files formatted with `black`. Non-negotiable, enforced by `run_tests.sh`.

**Linting — `pylint`**
Zero warnings enforced. Project-level `.pylintrc` committed to the repo to set
consistent rules. `run_tests.sh` fails if pylint reports any warnings.

**Type hints**
All public functions have type annotations on parameters and return values.
Makes the codebase readable and catches bugs early without a full type checker.

**Dependencies**
`requirements.txt` with pinned versions for reproducibility:
```
anthropic==...
python-dotenv==...
pyarrow==...
torch==...
numpy==...
scikit-learn==...
pytest==...
black==...
pylint==...
```

**API key — `.env`**
`ANTHROPIC_API_KEY` is loaded from a `.env` file at script startup via
`python-dotenv`. The file is listed in `.gitignore` and never committed.
A `.env.example` is committed with the key name but no value:
```
ANTHROPIC_API_KEY=your_key_here
```

**Logging**
`generate.py` runs for up to 24h — structured progress output is essential:
- Session counts: pending / labeled / failed at each stage
- Batch status: submitted, polling, retrieved
- Cost estimate printed after transcript generation, before batch submission
- Warnings for missing raw files, stale schema versions, label mismatches
- Final summary: total steps, STUCK/PRODUCTIVE/UNSURE counts per source

**Committed dataset configs**
`datasets/` directory with `fetch.json` and `filter.json` for every source is
committed to the repo. Without these, a user cloning the repo cannot reproduce
the dataset generation. Raw session files referenced in `fetch.json` are not
committed (too large / proprietary) but their expected paths are documented.

---

## Testing

All tests use synthetic fixture sessions — no real parquet files, no API calls.
Batch API calls are mocked. Run with:

```bash
./run_tests.sh   # black --check, pylint, pytest tests/ -v — all must pass
```

### Structure

```
tests/
  fixtures/                     # minimal synthetic sessions, one per parser format
    nlile_session.json
    dataclaw_session.json
    claudeset_session.json
  test_parsers.py
  test_label_session.py
  test_batch_label.py
  test_extract_features.py
  test_merge_session.py
  test_migrate_features.py
  test_artifact_lifecycle.py
  test_filters.py
  test_training.py
  test_integration.py
```

All Batch API interactions use `unittest.mock` — no real API calls are made
in any test. Tests must pass without `ANTHROPIC_API_KEY` set.

### What each test covers

**`test_parsers.py`**
- Each parser produces the expected step format from its fixture session
- Edge cases: empty tool output, missing fields, session below `min_steps`
- `model_filter` excludes non-Claude sessions
- Malformed session (truncated JSON, missing keys) is skipped with a warning, not a crash
- Session with 0 tool calls handled correctly

**`test_label_session.py` — transcript compression**
- Tool output capped at 500 chars, `[...]` suffix appended when truncated
- Output under 500 chars passed through unchanged, no `[...]` appended
- Very long output (1MB+) does not OOM — truncation is applied before any string copy
- Correct step count produced for a known fixture
- Empty output handled gracefully
- Total transcript size is within expected bounds after compression

**`test_batch_label.py` — all Batch API calls mocked**
- CSV `P,S,U,P` correctly maps to `PRODUCTIVE/STUCK/UNSURE`
- Case-insensitive: `p,s,u` accepted
- Whitespace and trailing newlines stripped before parsing
- Trailing comma handled gracefully
- `len(labels) != n_steps` triggers retry logic (mock returns correct count on second attempt)
- Unknown characters in CSV raise an error
- Sonnet returns JSON array instead of CSV — detected and rejected
- Resume: `pending_batch.json` present on startup → polls instead of resubmitting (mock)
- Credit limit hit: batch ID saved to `pending_batch.json`, re-run retrieves results (mock)
- Partial batch results: failed sessions marked `failed`, successful ones written

**`test_extract_features.py`**
- Known fixture session produces expected feature values (regression test)
- `schema_version` and `n_steps` written correctly
- Idempotency: re-running produces identical output
- Interrupted write (partial file): detected by `n_steps` mismatch, re-extracted

**`test_merge_session.py`**
- Matching label and feature files merge into correct JSONL rows
- Mismatched `n_steps` raises an error
- Label encoding: `PRODUCTIVE→0.0`, `STUCK→1.0`, `UNSURE→0.5`
- Stale `schema_version` raises an error

**`test_migrate_features.py`**
- v1→v2 migration produces expected fields
- Chained v1→v3 applies both functions in order
- Artifact with mixed `schema_version` rows (partially migrated) detected by `--verify`
- Migration is idempotent — running v1→v2 twice does not corrupt rows
- Missing raw session falls back to default value, not a crash
- `--verify` detects inconsistent `n_steps` within a session

**`test_artifact_lifecycle.py`**
- New session appended to existing artifact
- Session already in artifact is skipped (idempotency)
- Deleted raw file: artifact row preserved, warning logged
- `--drop-missing` removes orphaned rows
- Concurrent append safety: two writes do not corrupt the `.gz`

**`test_filters.py`**
- `min_steps`/`max_steps` excludes out-of-range sessions
- `max_sessions` caps the count
- `folder_limits` glob patterns match correctly
- Empty source (0 sessions after filtering) fails with a clear error, not silently

**`test_training.py`**
- Training manifest loads with correct weights
- Weighted sampler produces correct oversampling ratio
- Label encoding consistent between merge output and training loader
- Session-level train/test split: no session appears in both splits (no leakage)
- Class balance reported correctly after weighted sampling
- Training with an empty source (0 steps after filtering) fails with a clear error

**`test_integration.py` — end-to-end**
- Full `generate.py` pipeline on a synthetic source with fixture sessions
- Verifies final `.jsonl` contains correct rows, labels, features, and step count
- Re-running produces identical output (full pipeline idempotency)
- New session added to source → re-running appends it, existing rows unchanged

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
