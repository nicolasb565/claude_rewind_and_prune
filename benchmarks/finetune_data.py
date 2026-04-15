#!/usr/bin/env python3
"""
Build fine-tuning dataset for the stuck-detection classifier.

Extracts labeled sessions from the existing v6 training jsonl files (which
contain session_id, step, label triples but no raw text), joins them with the
raw transcripts from each source corpus, parses steps, and emits multi-turn
chat format JSONL for SFTTrainer.

Output format (one JSON object per session, per line):
  {
    "session_id": "nlile_xxx",
    "source": "nlile",
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "<rendered step 0>"},
      {"role": "assistant", "content": "P"},
      {"role": "user", "content": "<rendered step 1>"},
      {"role": "assistant", "content": "P"},
      ...
    ]
  }

Causal attention during training ensures each assistant label only sees
prior user+assistant turns, not future ones — so this format is causally
correct for a real-time deployment target. Loss is computed only on
assistant label tokens via a collator or completion_only_loss flag.

Usage:
  .venv/bin/python benchmarks/finetune_data.py --out data/generated/finetune_chat.jsonl
  .venv/bin/python benchmarks/finetune_data.py --inspect  # sample a few examples
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.pipeline.parsers import nlile as nlile_parser  # noqa: E402
from src.pipeline.parsers import dataclaw as dataclaw_parser  # noqa: E402
from src.pipeline.parsers import claudeset as claudeset_parser  # noqa: E402
from src.pipeline.label_session import _render_step  # noqa: E402

# ── Corpora config ────────────────────────────────────────────────────────

V6_JSONLS = {
    "nlile": REPO / "data" / "generated" / "nlile_v6.jsonl",
    "dataclaw_claude": REPO / "data" / "generated" / "dataclaw_claude_v6.jsonl",
    "masterclass": REPO / "data" / "generated" / "masterclass_v6.jsonl",
    "claudeset": REPO / "data" / "generated" / "claudeset_v6.jsonl",
}

HF_REPOS = {
    "dataclaw_claude": "woctordho/dataclaw",
    "masterclass": "gutenbergpbc/john-masterclass-cc",
    "claudeset": "lelouch0110/claudeset-community",
}
NLILE_PARQUET_DIR = REPO / "data" / "separate" / "nlile_parquet" / "data"

# System prompt. Kept short — the model learns the task from the training
# examples. The per-step rendering (the user messages) is more important than
# the system prompt for discrimination, since Sonnet saw the same per-step
# format when it produced the labels.
SYSTEM_PROMPT = """\
You are labeling tool-call steps in a Claude Code session.
For each tool call, reply with exactly one letter:
  P = productive (new approach, first attempt, or iteration making progress)
  S = stuck (same command or same error repeating without visible progress)
  U = unsure (genuine ambiguity)
Output format: a single letter, nothing else."""


# ── Helpers ───────────────────────────────────────────────────────────────

def _label_token(label_float: float) -> str:
    if label_float >= 0.9:
        return "S"
    if label_float <= 0.1:
        return "P"
    return "U"


# Use _render_step DIRECTLY (imported from src/pipeline/label_session.py).
# This is the exact rendering Sonnet saw when producing the labels:
#   [N] ToolName
#     key: value
#     → output[:500] + "[...]"
# Matching this exactly means training-time content is byte-identical to
# labeling-time content, so there's no format drift between what the label
# was assigned to and what the model sees during fine-tuning.
def _render_step_for_training(step: dict, i: int) -> str:
    return _render_step(step, i)


def load_labels(source: str) -> dict[str, list[tuple[int, float]]]:
    """Return {session_id_without_prefix: [(step, label), ...]} for the given source."""
    path = V6_JSONLS[source]
    if not path.exists():
        print(f"  [skip] {source}: {path} missing")
        return {}
    by_session: dict[str, list[tuple[int, float]]] = defaultdict(list)
    prefix = f"{source}_"
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid_full = row.get("session_id", "")
            if sid_full.startswith(prefix):
                sid = sid_full[len(prefix):]
            else:
                sid = sid_full
            step = int(row.get("step", 0))
            label = float(row.get("label", 0.5))
            by_session[sid].append((step, label))
    # Sort by step
    for sid in by_session:
        by_session[sid].sort()
    return dict(by_session)


def load_raw_transcripts_nlile() -> dict[str, list[dict]]:
    """Return {nlile_id: [parsed_steps]} from local parquet shards."""
    import pyarrow.parquet as pq
    out: dict[str, list[dict]] = {}
    shards = sorted(NLILE_PARQUET_DIR.glob("train-*.parquet"))
    print(f"  [nlile] reading {len(shards)} parquet shards")
    for shard in shards:
        tbl = pq.read_table(shard, columns=["id", "messages_json"]).to_pylist()
        for row in tbl:
            try:
                msgs = json.loads(row["messages_json"])
                steps = nlile_parser.parse_session(msgs)
            except Exception:
                continue
            out[row["id"]] = steps
    return out


def load_raw_transcripts_hf(source: str) -> dict[str, list[dict]]:
    """Return {hf_id: [parsed_steps]} from the cached HF dataset."""
    try:
        import datasets as hf_datasets  # pylint: disable=no-name-in-module
    except ImportError:
        print(f"  [{source}] datasets package not installed, skipping")
        return {}
    repo = HF_REPOS[source]
    print(f"  [{source}] loading HF dataset {repo}")
    try:
        ds = hf_datasets.load_dataset(repo, split="train")
    except Exception as e:
        print(f"  [{source}] load_dataset failed: {e}")
        return {}

    out: dict[str, list[dict]] = {}
    parser_name = "claudeset" if source == "claudeset" else "dataclaw"
    for row in ds:
        sid = row.get("session_id", row.get("id", ""))
        try:
            if parser_name == "claudeset":
                steps = claudeset_parser.parse_session(row.get("turns", []))
            else:
                steps = dataclaw_parser.parse_session(row.get("messages", []))
        except Exception:
            continue
        out[sid] = steps
    return out


# ── Build chat sessions ───────────────────────────────────────────────────

def build_chat_session(
    session_id: str,
    source: str,
    steps: list[dict],
    labeled_steps: list[tuple[int, float]],
    drop_unsure: bool = True,
) -> dict | None:
    """
    Render one session as multi-turn chat. Returns None if no valid labels.

    Each labeled step becomes a user message (rendered tool call) + assistant
    message (single letter label). Steps that are compact blocks or that don't
    have matching labels are skipped. Unsure labels are dropped by default
    (drop the whole turn, not the session — we just don't teach on that step).
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Filter out compact blocks (dict-typed steps with type=="compact")
    # parse_session already does this for claudeset; dataclaw/nlile produce
    # only tool_use steps, but be defensive.
    real_steps = [s for s in steps if isinstance(s, dict) and not s.get("type") == "compact"]

    label_map = dict(labeled_steps)  # {step_index: label_float}
    n = min(len(real_steps), max(label_map.keys()) + 1 if label_map else 0)

    n_labeled = 0
    for i in range(n):
        if i not in label_map:
            continue
        lbl_token = _label_token(label_map[i])
        if drop_unsure and lbl_token == "U":
            continue
        step_text = _render_step_for_training(real_steps[i], i)
        if not step_text.strip():
            continue
        messages.append({"role": "user", "content": step_text})
        messages.append({"role": "assistant", "content": lbl_token})
        n_labeled += 1

    if n_labeled == 0:
        return None

    return {
        "session_id": f"{source}_{session_id}",
        "source": source,
        "messages": messages,
        "n_labeled": n_labeled,
    }


# ── Main extraction ───────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-train", default="data/generated/finetune_train.jsonl")
    ap.add_argument("--out-val", default="data/generated/finetune_val.jsonl")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--sources", nargs="+",
                    default=["nlile", "dataclaw_claude", "masterclass", "claudeset"])
    ap.add_argument("--inspect", action="store_true",
                    help="print 3 sample sessions and exit")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    all_sessions: list[dict] = []
    stats: dict[str, dict] = {}
    label_counts: Counter = Counter()

    for source in args.sources:
        print(f"\n=== {source} ===")
        labels = load_labels(source)
        if not labels:
            continue
        print(f"  {len(labels)} labeled sessions in v6 jsonl")

        if source == "nlile":
            raw = load_raw_transcripts_nlile()
        else:
            raw = load_raw_transcripts_hf(source)
        print(f"  {len(raw)} raw sessions loaded")

        # Join
        joined = 0
        missing = 0
        corpus_sessions = []
        for sid, labeled_steps in labels.items():
            if sid not in raw:
                missing += 1
                continue
            sess = build_chat_session(sid, source, raw[sid], labeled_steps)
            if sess is None:
                continue
            corpus_sessions.append(sess)
            joined += 1
            for msg in sess["messages"]:
                if msg["role"] == "assistant":
                    label_counts[msg["content"]] += 1
        print(f"  joined: {joined}, missing raw: {missing}")

        stats[source] = {
            "labeled_in_v6": len(labels),
            "raw_loaded": len(raw),
            "joined": joined,
            "missing": missing,
        }
        all_sessions.extend(corpus_sessions)

    print(f"\n=== total ===")
    print(f"  sessions: {len(all_sessions)}")
    print(f"  label distribution: {dict(label_counts)}")
    if label_counts:
        total_labels = sum(label_counts.values())
        print(f"  P fraction: {label_counts['P'] / total_labels:.3f}")
        print(f"  S fraction: {label_counts['S'] / total_labels:.3f}")

    if args.inspect:
        print("\n=== sample sessions ===")
        for sess in random.sample(all_sessions, min(3, len(all_sessions))):
            print(f"\n--- {sess['session_id']} (n_labeled={sess['n_labeled']}) ---")
            for i, msg in enumerate(sess["messages"][:8]):
                content = msg["content"][:300]
                print(f"[{msg['role']}] {content}")
            if len(sess["messages"]) > 8:
                print(f"... ({len(sess['messages']) - 8} more messages)")
        return 0

    # Shuffle sessions and split train/val at session level
    random.shuffle(all_sessions)
    n_val = max(1, int(len(all_sessions) * args.val_fraction))
    val = all_sessions[:n_val]
    train = all_sessions[n_val:]

    out_train = REPO / args.out_train
    out_val = REPO / args.out_val
    out_train.parent.mkdir(parents=True, exist_ok=True)

    with open(out_train, "w") as f:
        for sess in train:
            f.write(json.dumps(sess) + "\n")
    with open(out_val, "w") as f:
        for sess in val:
            f.write(json.dumps(sess) + "\n")

    print(f"\n=== wrote ===")
    print(f"  train: {out_train}  ({len(train)} sessions)")
    print(f"  val:   {out_val}  ({len(val)} sessions)")
    print(f"\n=== per-source stats ===")
    for s, d in stats.items():
        print(f"  {s}: {d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
