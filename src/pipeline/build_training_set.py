#!/usr/bin/env python3
"""
Scale the Sonnet annotator over a target-count of sessions to build a
hygiene-training corpus.

Input: streams masterclass-cc + dataclaw HF datasets, filters to sessions
with 20-150 tool steps (cost-capped and long enough to have meaningful
checkpoints), normalizes via the same parser used by
sample_test_sessions.py.

Output: data/generated/hygiene_v1.jsonl — one annotated Session per line,
each with `checkpoints` (Sonnet-labeled) and `expire_step_ids` fields.

Resumable: every completed session is flushed immediately. Re-running
the script with an existing output file skips already-annotated
session_ids.

Cost control:
  --target-count N       how many sessions to produce (default 100)
  --max-steps N          skip sessions above this many tool steps (default 150)
  --dry-run              print what would be annotated without calling API

Usage (v1 smoke):
  set -a; source .env; set +a
  .venv/bin/python -m src.pipeline.build_training_set --target-count 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from src.pipeline.hygiene_types import Session
from src.pipeline.sample_test_sessions import normalize_masterclass_dataclaw
from src.pipeline.annotate_sonnet import call_sonnet


REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "data" / "generated"


# (dataset name, messages field). Ordered so we iterate through cheaper
# masterclass first before spending on long dataclaw sessions.
DATASETS = [
    ("gutenbergpbc/john-masterclass-cc", "messages"),
    ("woctordho/dataclaw", "messages"),
]


def load_done_ids(out_file: Path) -> set[str]:
    if not out_file.exists():
        return set()
    done = set()
    with open(out_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = d.get("session_id")
            if sid:
                done.add(str(sid))
    return done


def iter_candidates(min_steps: int, max_steps: int, max_scan_per_dataset: int = 30000):
    """Yield (dataset_label, session_id, Session) tuples."""
    from datasets import load_dataset

    for name, field in DATASETS:
        ds = load_dataset(name, split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= max_scan_per_dataset:
                break
            msgs = row.get(field) or []
            if not isinstance(msgs, list):
                continue
            row_id = row.get("id") or row.get("session_id") or f"row_{i}"
            source = f"{name}#{i}:{row_id}"
            session = normalize_masterclass_dataclaw(str(row_id), source, msgs)
            if session is None:
                continue
            tool_steps = sum(1 for s in session.steps if s.role == "tool")
            if not (min_steps <= tool_steps <= max_steps):
                continue
            yield name, str(row_id), session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-count", type=int, default=100)
    ap.add_argument("--min-steps", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--output", type=Path, default=OUT_DIR / "hygiene_v1.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        env_file = REPO / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(args.output)
    if done:
        print(f"[resume] {len(done)} sessions already annotated in {args.output.name}")

    produced = len(done)
    skipped = 0
    cost_estimate_usd = 0.0
    start = time.time()

    with open(args.output, "a") as out:
        for dataset_name, session_id, session in iter_candidates(args.min_steps, args.max_steps):
            if produced >= args.target_count:
                break
            if session_id in done:
                skipped += 1
                continue
            tool_steps = sum(1 for s in session.steps if s.role == "tool")
            elapsed = time.time() - start
            print(f"[{produced+1}/{args.target_count}] {dataset_name}#{session_id[:20]} "
                  f"steps={len(session.steps)}({tool_steps} tool)  elapsed={elapsed:.0f}s  "
                  f"cost≈${cost_estimate_usd:.2f}", file=sys.stderr)

            if args.dry_run:
                produced += 1
                continue

            try:
                checkpoints, expire_ids = call_sonnet(session)
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}", file=sys.stderr)
                # Back off a bit on any error (rate limit / network)
                time.sleep(3)
                continue

            session.set_annotations(checkpoints, expire_ids)
            out.write(json.dumps(session.to_dict()) + "\n")
            out.flush()

            # Rough cost update — call_sonnet already prints true tokens to
            # stderr; approximate from session size here.
            rendered_chars = sum(
                len(s.output) + len(s.cmd) + len(s.text) for s in session.steps
            )
            cost_estimate_usd += (rendered_chars / 4) / 1_000_000 * 3.0  # $3/M input
            cost_estimate_usd += 1000 / 1_000_000 * 15.0                  # ~1K output

            produced += 1
            done.add(session_id)

    print(f"\ndone: produced={produced} skipped_already_done={skipped} "
          f"elapsed={time.time() - start:.0f}s est_cost=${cost_estimate_usd:.2f}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
