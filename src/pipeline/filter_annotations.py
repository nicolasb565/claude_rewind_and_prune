#!/usr/bin/env python3
"""
Post-process hygiene_v1.jsonl to drop unsafe `expire` annotations:

  1) expire events targeting non-tool steps (user_text / assistant_text) —
     there is no tool output to expire on these; Sonnet sometimes marked
     long text turns anyway. Drop.
  2) expire events targeting Read/Edit/Write-category tools — file-based
     tool outputs are load-bearing when the agent later modifies the
     same file. Sonnet cannot reliably catch this. Drop.

Keeps expire on Bash/Grep/Glob-class tools where Sonnet's judgment is
reliable (verified: 90%+ of Glob/Grep expires, 77% of Bash expires, no
file-state hazard).

Checkpoint annotations pass through unchanged.

Usage:
  .venv/bin/python -m src.pipeline.filter_annotations \\
      --in data/generated/hygiene_v1.jsonl \\
      --out data/generated/hygiene_v1_filtered.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Tool-name categorization. Cover canonical Claude Code + common variants
# seen in the cached datasets. Anything not listed falls into "other" and
# is treated as Bash-like (safe to expire by default — the conservative
# decision is to expire less, but the hazard only applies when a file is
# later modified, which is specifically Read/Edit/Write).
FILE_BASED_TOOLS = {
    # Reading
    "Read", "read_file", "read_many_files", "list_directory",
    # Editing / writing
    "Edit", "Write", "MultiEdit", "replace", "write_file", "apply_patch",
}


def is_file_based_tool(tool_name: str) -> bool:
    return tool_name in FILE_BASED_TOOLS


def filter_events(row: dict) -> tuple[dict, dict]:
    """Return (new_row, drop_counts)."""
    steps = row.get("steps", [])
    by_idx = {s["idx"]: s for s in steps}
    drops = {"non_tool": 0, "file_based": 0}

    new_events = []
    for ev in row.get("events", []):
        if "expire" in ev:
            i = ev["expire"]
            step = by_idx.get(i)
            if step is None:
                # Out-of-range — would also be caught by a validator;
                # drop silently.
                drops.setdefault("out_of_range", 0)
                drops["out_of_range"] += 1
                continue
            if step.get("role") != "tool":
                drops["non_tool"] += 1
                continue
            if is_file_based_tool(step.get("tool_name", "")):
                drops["file_based"] += 1
                continue
        new_events.append(ev)

    new_row = dict(row)
    new_row["events"] = new_events
    return new_row, drops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", dest="outp", type=Path, required=True)
    args = ap.parse_args()

    args.outp.parent.mkdir(parents=True, exist_ok=True)
    totals = {"rows": 0, "non_tool": 0, "file_based": 0, "out_of_range": 0}
    before_events = 0
    after_events = 0
    with args.inp.open() as fin, args.outp.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            before_events += len(row.get("events", []))
            new_row, drops = filter_events(row)
            after_events += len(new_row["events"])
            fout.write(json.dumps(new_row) + "\n")
            totals["rows"] += 1
            for k, v in drops.items():
                totals[k] = totals.get(k, 0) + v

    print(f"processed {totals['rows']} sessions")
    print(f"  events before: {before_events}")
    print(f"  events after:  {after_events}")
    print(f"  dropped (non-tool step):   {totals['non_tool']}")
    print(f"  dropped (file-based tool): {totals['file_based']}")
    print(f"  dropped (out-of-range):    {totals.get('out_of_range', 0)}")


if __name__ == "__main__":
    main()
