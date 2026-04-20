#!/usr/bin/env python3
"""
Sample 3 Claude Code sessions (short / medium / long) from cached HF
datasets and normalize into the Session schema for annotator testing.

Targets:
  short  — lelouch0110/claudeset-community  (aim ~6–20 steps)
  medium — gutenbergpbc/john-masterclass-cc (aim ~25–60 steps)
  long   — woctordho/dataclaw               (aim ~100–200 steps)

Output: data/annotate_test/{short,medium,long}.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from src.pipeline.hygiene_types import Session, Step

REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "data" / "annotate_test"


def _cmd_from_input(tool: str, inp: dict) -> str:
    """Pull a short primary-key identifier out of a tool's input dict."""
    if not isinstance(inp, dict):
        return ""
    for key in ("command", "file_path", "path", "pattern", "query"):
        v = inp.get(key)
        if v:
            return str(v)
    # fallback — short repr
    s = str(inp)
    return s[:300]


def _output_text(out: Any) -> str:
    """Coerce a tool's output field (dict | str | list) into a single string."""
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        return str(out.get("text", out.get("content", "")))
    if isinstance(out, list):
        parts = []
        for b in out:
            if isinstance(b, dict):
                parts.append(str(b.get("text", b.get("content", ""))))
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(parts)
    return ""


def normalize_masterclass_dataclaw(row_id: str, source_label: str, messages: list[dict]) -> Session | None:
    """Parser for gutenbergpbc/masterclass-cc + woctordho/dataclaw.

    Shape: list of {role, content | tool_uses, timestamp}. `tool_uses` is
    a list of {tool, input, output} — pre-paired, not Anthropic-format.
    """
    steps: list[Step] = []
    idx = 0
    for msg in messages:
        role = msg.get("role")
        if msg.get("tool_uses"):
            for tu in msg["tool_uses"]:
                if not isinstance(tu, dict):
                    continue
                tool = tu.get("tool", "")
                inp = tu.get("input", {}) or {}
                cmd = _cmd_from_input(tool, inp)
                out = _output_text(tu.get("output", ""))
                steps.append(Step(
                    idx=idx, role="tool",
                    tool_name=tool, cmd=cmd, output=out,
                    input_file=inp.get("file_path") or inp.get("path") if isinstance(inp, dict) else None,
                ))
                idx += 1
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            steps.append(Step(
                idx=idx,
                role=("user_text" if role == "user" else "assistant_text"),
                text=content.strip(),
            ))
            idx += 1

    if not any(s.role == "tool" for s in steps):
        return None
    return Session(session_id=str(row_id), source=source_label, steps=steps)


def sample_from_dataset(name: str, messages_field: str, target_min: int, target_max: int,
                        max_scan: int = 500) -> Session | None:
    """Scan the streaming dataset, pick a row whose step-count (after parsing)
    falls in the target range. target_min/max are in *parsed steps*, not
    raw message count."""
    from datasets import load_dataset

    ds = load_dataset(name, split="train", streaming=True)
    rng = random.Random(42)
    candidates: list[tuple[int, Session]] = []
    for i, row in enumerate(ds):
        if i >= max_scan:
            break
        msgs = row.get(messages_field) or []
        if not isinstance(msgs, list):
            continue
        row_id = row.get("id") or row.get("session_id") or f"row_{i}"
        source_label = f"{name}#{i}:{row_id}"
        session = normalize_masterclass_dataclaw(str(row_id), source_label, msgs)
        if session is None:
            continue
        tool_steps = sum(1 for s in session.steps if s.role == "tool")
        if target_min <= tool_steps <= target_max:
            candidates.append((i, session))
        if len(candidates) >= 20:
            break

    if not candidates:
        print(f"WARN: no sessions in {name} matching [{target_min},{target_max}] tool-steps in first {max_scan} rows", file=sys.stderr)
        return None

    i, session = rng.choice(candidates)
    return session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing samples")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # claudeset-community turns are pre-summarized (not raw tool calls), unusable.
    # Pull short from masterclass too; medium/long from masterclass/dataclaw.
    targets = [
        ("short",  "gutenbergpbc/john-masterclass-cc",   "messages",  5,  20),
        ("medium", "gutenbergpbc/john-masterclass-cc",   "messages", 30,  60),
        ("long",   "woctordho/dataclaw",                 "messages", 100, 200),
    ]

    for label, dataset, field, lo, hi in targets:
        out = OUT_DIR / f"{label}.json"
        if out.exists() and not args.force:
            print(f"[skip] {out} exists (use --force to overwrite)")
            continue
        print(f"[{label}] sampling from {dataset} target_steps=[{lo},{hi}]")
        session = sample_from_dataset(dataset, field, lo, hi)
        if session is None:
            print(f"  ERROR: no session sampled")
            continue
        with open(out, "w") as f:
            json.dump(session.to_dict(), f, indent=2)
        print(f"  saved {len(session.steps)} steps → {out}")


if __name__ == "__main__":
    main()
