#!/usr/bin/env python3
"""
Render a single session from hygiene_v1.jsonl with its annotations
interleaved inline, for human quality review.

Usage:
  .venv/bin/python -m src.pipeline.review_session --line 3
  .venv/bin/python -m src.pipeline.review_session --line 7 --file data/generated/hygiene_v1.jsonl
  .venv/bin/python -m src.pipeline.review_session --line 22 --full-output
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def wrap_text(text: str, width: int = 110, indent: str = "    ") -> str:
    import textwrap
    lines = []
    for para in text.splitlines():
        if len(para) <= width:
            lines.append(para)
        else:
            lines.extend(textwrap.wrap(para, width=width, subsequent_indent=""))
    return f"\n{indent}".join(lines)


def render_step(step: dict, max_out_chars: int, ttl_marked: bool) -> str:
    role = step["role"]
    idx = step["idx"]
    ttl_badge = f"  ◀── EXPIRE (step {idx})" if ttl_marked else ""
    if role == "user_text":
        text = step.get("text", "")
        snippet = text[:300] + (f"  (…{len(text)-300} more)" if len(text) > 300 else "")
        return f"[{idx:3d}] USER TEXT{ttl_badge}\n    {wrap_text(snippet)}"
    if role == "assistant_text":
        text = step.get("text", "")
        snippet = text[:300] + (f"  (…{len(text)-300} more)" if len(text) > 300 else "")
        return f"[{idx:3d}] ASST TEXT{ttl_badge}\n    {wrap_text(snippet)}"
    # tool step
    tool = step.get("tool_name", "?")
    cmd = step.get("cmd", "")
    out = step.get("output", "")
    cmd_short = cmd[:200] + (f"  (…{len(cmd)-200} more)" if len(cmd) > 200 else "")
    if max_out_chars > 0:
        out_short = out[:max_out_chars] + (f"  (…{len(out)-max_out_chars} more chars)" if len(out) > max_out_chars else "")
    else:
        out_short = ""
    lines = [f"[{idx:3d}] TOOL  {tool}{ttl_badge}",
             f"    cmd: {wrap_text(cmd_short)}"]
    if out_short:
        lines.append(f"    out: {wrap_text(out_short)}")
    return "\n".join(lines)


def render_checkpoint(cp: dict, after_step: int) -> str:
    ptype = cp.get("progress_type", "?")
    finding = cp.get("finding", "")
    evidence = cp.get("evidence", "")
    nxt = cp.get("next_direction", "")
    banner = f"━━━━━━━━━━  CHECKPOINT after step {after_step}  [{ptype}]  ━━━━━━━━━━"
    return (f"\n{banner}\n"
            f"  finding:   {wrap_text(finding, indent='             ')}\n"
            f"  evidence:  {wrap_text(evidence, indent='             ')}\n"
            f"  next:      {wrap_text(nxt, indent='             ')}\n"
            f"{'━' * len(banner)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=REPO / "data" / "generated" / "hygiene_v1.jsonl")
    ap.add_argument("--line", type=int, required=True, help="0-based session index")
    ap.add_argument("--max-out-chars", type=int, default=400,
                    help="truncate tool outputs to N chars (0 to hide, default 400)")
    ap.add_argument("--full-output", action="store_true",
                    help="don't truncate tool outputs")
    args = ap.parse_args()

    with open(args.file) as f:
        for i, line in enumerate(f):
            if i == args.line:
                row = json.loads(line)
                break
        else:
            raise SystemExit(f"line {args.line} not found")

    print(f"═══ session {row['session_id']}  (line {args.line}) ═══")
    print(f"source: {row['source']}")
    steps = row["steps"]
    events = row["events"]
    expire_set = {e["expire"] for e in events if "expire" in e}
    ckpts_by_after = {}
    for e in events:
        if "checkpoint" in e:
            ckpts_by_after.setdefault(e["checkpoint"]["after_step"], []).append(e["checkpoint"])

    n_ckpts = sum(len(v) for v in ckpts_by_after.values())
    print(f"steps: {len(steps)}   checkpoints: {n_ckpts}   expire: {len(expire_set)}")
    if expire_set:
        print(f"expire step ids: {sorted(expire_set)}")
    if ckpts_by_after:
        print(f"ckpt step ids: {sorted(ckpts_by_after.keys())}")
    print()

    max_chars = 0 if args.max_out_chars == 0 else (10_000_000 if args.full_output else args.max_out_chars)

    for step in steps:
        print(render_step(step, max_chars, step["idx"] in expire_set))
        for cp in ckpts_by_after.get(step["idx"], []):
            print(render_checkpoint(cp, step["idx"]))
        print()


if __name__ == "__main__":
    main()
