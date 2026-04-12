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

**Idempotent / validation:** on startup, if a label file exists:
1. Parse it as JSON — if invalid (truncated write, corrupt file), delete and re-label.
2. Check `len(labels) == n_steps` — if the count is wrong, delete and re-label.
3. If both checks pass, skip (already complete).

Re-labeling always submits the full session — Sonnet needs full context to label
any step, so there is no partial completion for labels. Pass `--force` to force
re-labeling even on a valid file.

**Prompt contract:**
- System prompt explains the PRODUCTIVE / STUCK / UNSURE definitions — keep
  short (~500 tokens); it is sent with every request
- User message contains the full transcript + explicit step count; tool outputs
  truncated to 500 chars to minimize input tokens
- Output format: compact CSV `P,S,U,P,P,S` (single chars, comma-separated) —
  ~7x fewer output tokens than JSON array of strings
- Python splits on commas, maps `P→PRODUCTIVE`, `S→STUCK`, `U→UNSURE`
- Validates `len(labels) == n_steps` before writing

**System prompt (draft — tune after calibration run):**

```
You are labeling steps in a Claude Code session. Each step is one tool call.
Classify each step as P (productive), S (stuck), or U (unsure).

PRODUCTIVE: the session is making forward progress. Exploring a new approach,
writing code, reading a file for the first time, testing a hypothesis.
Errors are fine — what matters is that something new is being attempted.

STUCK: the session is in a loop. The same command, the same error, the same
edit repeated without a changed approach or new information. The work has
stopped moving forward.

UNSURE: genuine ambiguity that you cannot resolve from the transcript.
Use sparingly — not as a default.

Common patterns:
- First attempt at any command → P
- Same command, same error, second or third time → S
- Trying a different file, flag, or approach after failure → P
- Reading a file already read, with no new context → S
- Tight compile/test loop with unchanged failure → S

Transition rules:
- The first step of a repeating pattern is still P; label S when repetition begins
- The first step after escaping a loop (new approach, new tool) is P again

Output: one label per step, comma-separated, nothing else.
Example: P,P,S,S,S,P,P,S,P
```

No project context is given to the model — it does not need to know this is for
a stuck detector or a CNN. The transcript truncation is not disclosed either;
Sonnet just sees shorter outputs, which is normal variation in real sessions.

**Calibration — verify and tune the system prompt before the full run:**

Run **5 sessions per source** (~25 sessions total, ~$0.25) before committing
to the full run. Prefer sessions that contain at least one known stuck episode
— the benchmark sessions are ideal since you already know where the loops are.

```bash
python generate.py datasets/nlile/          --max-sessions 5
python generate.py datasets/dataclaw_claude/ --max-sessions 5
python generate.py datasets/masterclass/    --max-sessions 5
python generate.py datasets/claudeset/      --max-sessions 5
```

Manually inspect the label files against the raw sessions for each source:
- Are STUCK transitions happening at the right step, or too early / too late?
- Is UNSURE appearing at all? If so, on what kind of steps?
- Are any clearly repetitive steps labeled PRODUCTIVE?
- Are any clearly novel steps labeled STUCK?
- Are labels consistent in style across sources (same pattern, different language)?

5 sessions per source matters because stuck episodes are rare (~5–15% of steps)
— with a single source you might see only 1–2 stuck episodes, which is not enough
to judge transition accuracy. Cross-source consistency also needs to be checked
since Rust, Python, and PHP sessions look different on the surface.

If the labels look wrong for a systematic reason, adjust the system prompt
(tighten the transition rule, add or remove a pattern example) and re-run
the calibration. Only commit API credits to the full run once the output
looks correct across all sources.

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
- Run 5 sessions per source first (~25 total, ~$0.25) to calibrate before committing credits

**CLI:**
```
python src/label_session.py <session_transcript_path> --source <name> --out data/labels/
```

---

### `src/extract_features.py`

Computes per-step numeric features from a raw session. No LLM calls.

**Input:** raw session file (source-specific format, parsed via source adapter)

**Output:** `data/features/<source>/<session_id>_features.json`

**Idempotent / validation:** on startup, if a feature file exists:
1. Parse it as JSON — if invalid (truncated write, corrupt file), delete and re-extract all steps.
2. Check `len(steps) == n_steps` (expected step count from the raw session) — if
   fewer steps are present, re-extract all steps from scratch. Feature extraction
   is stateful (e.g. `steps_since_same_tool` depends on all prior steps), so
   completing missing steps requires replaying from step 0 anyway.
3. Check `schema_version` matches current — if stale, re-extract or migrate
   (see Feature Migration).
4. If all checks pass, skip (already complete).

Pass `--force` to re-extract even on a valid file.

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
4. Poll until batch completes or save `batch_id` to
   `data/labels/<source>/pending_batch.json` and exit — re-running resumes
5. On completion, parse each response (CSV `P,S,U,...`), validate
   `len(labels) == n_steps`, write label file
6. Sessions that fail validation are marked `failed` and excluded from the
   batch result — re-running will resubmit them

**Resume / error handling:**

*In-flight batch (normal resume):*
If `pending_batch.json` exists on startup, skip submission and go straight
to polling/retrieval. Sessions with existing complete label files are never
resubmitted.

*Batch expired (24h timeout):*
The Batch API expires a batch after 24 hours if not completed. On retrieval,
check the batch status — if `expired`, print a clear warning:
```
WARNING: batch <id> expired before completing. X/N requests were processed.
Deleting pending_batch.json — affected sessions will be resubmitted on next run.
```
Then delete `pending_batch.json` and write label files for any requests that
did complete before expiry. Re-running resubmits only the remaining sessions.

*Retry with exponential backoff — HTTP-level calls only:*
Batch submission and polling are quick synchronous HTTP calls where a transient
529/500 typically resolves within seconds. Retry these with exponential backoff:
- Up to 4 retries: delays of 1s, 2s, 4s, 8s (total ~15s max wait)
- Retry on: `429`, `500`, `529`
- Do not retry on: `400`, `401`, `402` (non-recoverable, fail immediately)
- If all retries exhausted: abort with a clear error and non-zero exit code

*Per-request errors inside a completed batch:*
These cannot be retried in-place — the batch is already done server-side and
individual failed requests are final. Re-running the script is the mechanism:
it submits a new batch containing only the unlabeled sessions.

On retrieval, check each request's `result.type` — if `errored`, classify the
error code:

Recoverable (leave unlabeled, exit 0 — re-run resubmits automatically):
- `529 overloaded`, `500 server error` — transient backend issue during processing
- `429 rate limit` — will succeed when rate resets

Non-recoverable (leave unlabeled, abort with non-zero exit code):
- `401 invalid API key` → ERROR: check ANTHROPIC_API_KEY in .env
- `402 insufficient credits` → ERROR: top up your account at console.anthropic.com
- `400 bad request` → ERROR: malformed transcript — bug in transcript generation

```
WARNING: 8 sessions failed with transient errors (529 overloaded).
Re-run to resubmit them automatically.

ERROR: 12 sessions failed with insufficient credits (402).
Top up your account at console.anthropic.com, then re-run.
```

If a batch contains both recoverable and non-recoverable errors, non-recoverable
takes priority: abort and report the non-recoverable issue first.

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

### `generate.py`

Top-level entry point. Drives the full pipeline for one or more source
directories: fetch → label → extract features → merge.

**CLI:**
```
python generate.py datasets/nlile/ datasets/dataclaw_claude/
python generate.py datasets/nlile/ --max-sessions 5   # calibration run
```

**Flags:**
- `--max-sessions N` — process at most N sessions per source (calibration)
- `--force-relabel` — re-label even if a valid label file exists
- `--schema-version N` — trigger feature re-extraction for files at older version
- `--retry-failed` — resubmit sessions previously marked as failed
- `--dry-run-estimate` — print token/cost estimate without submitting

**Behavior:**
1. For each source directory: read `fetch.json` — download raw sessions if not present
2. Read `filter.json` — select sessions matching criteria
3. Run `pipeline/batch_label.py` — submit/retrieve labels for all pending sessions
4. For each labeled session:
   - Run `pipeline/extract_features.py` (skip if feature file is current and complete)
   - Run `pipeline/merge_session.py`
   - Mark session as `done` in progress file immediately on success
   - Mark session as `failed` on error, continue with remaining sessions
5. Write `data/generated/<source>_v<N>.jsonl` from all completed sessions
6. Print summary: done / failed / pending per source

**Resume behavior:** re-running with the same arguments resumes from where
it left off. Sessions with existing valid label and feature files are skipped.
In-flight batches are detected via `pending_batch.json` and polled rather than
resubmitted.

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

**`test_label_session.py` — transcript compression + file validation**
- Tool output capped at 500 chars, `[...]` suffix appended when truncated
- Output under 500 chars passed through unchanged, no `[...]` appended
- Very long output (1MB+) does not OOM — truncation is applied before any string copy
- Correct step count produced for a known fixture
- Empty output handled gracefully
- Total transcript size is within expected bounds after compression
- Existing valid label file with correct count → skipped (no API call)
- Existing label file with invalid JSON → deleted and re-labeled
- Existing label file with wrong label count (too few) → deleted and re-labeled
- `--force` flag → re-labels even when file is valid

**`test_batch_label.py` — all Batch API calls mocked**
- CSV `P,S,U,P` correctly maps to `PRODUCTIVE/STUCK/UNSURE`
- Case-insensitive: `p,s,u` accepted
- Whitespace and trailing newlines stripped before parsing
- Trailing comma handled gracefully
- `len(labels) != n_steps` triggers retry logic (mock returns correct count on second attempt)
- Unknown characters in CSV raise an error
- Sonnet returns JSON array instead of CSV — detected and rejected
- Resume: `pending_batch.json` present on startup → polls instead of resubmitting (mock)
- Batch expired (status=`expired`): warning printed, `pending_batch.json` deleted,
  completed requests written, remaining sessions left unlabeled for resubmission (mock)
- HTTP 529 on batch submission: retried with exponential backoff, succeeds on
  third attempt (mock)
- HTTP 529 on batch submission: all 4 retries exhausted → abort with non-zero
  exit code (mock)
- HTTP 400/401/402 on submission: no retry, immediate abort (mock)
- Recoverable per-request errors (529, 500, 429) in completed batch: warning
  printed, sessions left unlabeled for resubmission, exit code 0 (mock)
- Non-recoverable per-request errors (401, 402, 400) in completed batch: error
  message printed with actionable guidance, exit code non-zero (mock)
- Mix of recoverable and non-recoverable errors: non-recoverable takes priority,
  abort with non-zero exit code (mock)
- Partial batch results: mix of succeeded/errored/expired requests handled correctly

**`test_extract_features.py`**
- Known fixture session produces expected feature values (regression test)
- `schema_version` and `n_steps` written correctly
- Existing valid feature file with correct step count and current schema → skipped
- Existing feature file with invalid JSON → deleted and re-extracted from scratch
- Existing feature file with fewer steps than the session → re-extracted from scratch
- Existing feature file with stale `schema_version` → re-extracted (or migrated)
- `--force` flag → re-extracts even when file is valid
- Idempotency: re-running produces identical output

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
| `src/merge_sources.py` | replaced by `src/pipeline/merge_session.py` + `generate.py` |
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
