#!/usr/bin/env python3
"""
Sonnet-based annotator for hygiene training data.

One API call per session. We render the session into a compact form
(compressed tool outputs, no thinking blocks) and ask Sonnet to identify
every checkpoint moment plus its type, finding, evidence, and
next_direction — matching the same schema the MCP tool uses.

Prompt-token budget: ~20K per session after compression. At ~$3/M input
and ~$15/M output, a single call costs <$0.10.

Usage:
  ANTHROPIC_API_KEY=... .venv/bin/python benchmarks/annotate_sonnet.py \\
      --in data/annotate_test/short.json \\
      --out data/annotate_test/short.sonnet.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from src.pipeline.hygiene_types import Checkpoint, Session, render_session_for_prompt


MODEL = os.environ.get("SONNET_MODEL", "claude-sonnet-4-6")
MAX_TOKENS_OUT = 4096


PROMPT = """You are annotating a Claude Code trajectory to produce two kinds of labels for training a small model on context-hygiene behavior:

1) CHECKPOINT moments — where the agent should consolidate progress into a short summary and drop prior exploration from its context.

   A checkpoint is one of two types:
   - milestone_achieved — the agent confirmed something concrete (bug identified, test passes, fix verified, subsystem understood).
   - approach_eliminated — a specific hypothesis or approach was ruled out by a specific observation (failed test, reverted edit, matching error repeated).

   Do NOT mark a checkpoint for "still exploring and unsure" — only when something concrete happened that the agent should now lock in.

2) EXPIRE step ids — step indices whose tool output is safe to expire from later context once the agent has moved on.

   This applies to any tool (Bash, Read, Grep, Glob, Edit, Write, MultiEdit). The decision is semantic, not tool-type-based: which outputs are load-bearing downstream, which are not.

   Safe to EXPIRE means: the information the agent needed from that output was already extracted into the agent's reasoning or absorbed into a subsequent decision, and leaving the raw output in context adds no downstream signal.

   Examples of safe-to-EXPIRE:
   - A 300-line `make` log whose only load-bearing signal was "build failed with error X" — the agent noted it and moved on.
   - An early Read of a file the agent re-Reads later (the later read supersedes; the earlier is redundant).
   - A Grep that was navigational — used to locate a function, the agent then Edited it, the grep results aren't referenced again.
   - A dir listing that pointed at the next file to open, after which the listing is unused.

   Do NOT EXPIRE a step whose output contains load-bearing info the agent will clearly revisit:
   - The error/stack trace that pinpointed the bug (if the fix comes from that specific information).
   - The test-pass confirmation that validated a fix.
   - A Read whose line numbers or specific content the agent later references indirectly (e.g., Edits referencing a line the Read revealed).
   - A Grep result the agent uses for subsequent navigation without re-running.
   - A config value the agent is copying elsewhere.

   Be especially careful with Read/Grep: they often ARE load-bearing because Edits reference their content by line number or structure. Only EXPIRE a Read/Grep if you can see evidence the agent no longer uses it (superseded by a later Read of the same file, or the relevant file was closed/abandoned).

   List every step id that is safe to expire. Size is not the criterion — load-bearingness is.

Output a single JSON object with one field `events`: an ordered list where each element is EITHER a EXPIRE event OR a checkpoint event. Order events by the step index they reference (ascending).

EXPIRE event shape:
  {"expire": <int>}     // the step id whose output can be expired

Checkpoint event shape:
  {"checkpoint": {
    "after_step": <int>,
    "progress_type": "milestone_achieved" | "approach_eliminated",
    "finding": "<1 sentence — what is now true>",
    "evidence": "<1 sentence — the concrete observation; reference specific tool output, file, command, or test>",
    "next_direction": "<1 sentence — what to do next>"
  }}

Each element must have exactly ONE key (`expire` OR `checkpoint`), never both.

Full output shape:
{
  "events": [
    {"expire": 2},
    {"expire": 12},
    {"checkpoint": {"after_step": 38, "progress_type": "milestone_achieved", ...}},
    {"expire": 41}
  ]
}

If no events, return "events": [].

Session:
----
{session_text}
----
Return ONLY the JSON object."""


def call_sonnet(session: Session) -> tuple[list[Checkpoint], list[int]]:
    """Returns (checkpoints, preserve_step_ids)."""
    import anthropic
    client = anthropic.Anthropic()
    rendered = render_session_for_prompt(session, max_output_chars=1500)
    prompt = PROMPT.replace("{session_text}", rendered)
    # Ballpark the token cost so the user can see it upfront.
    approx_in = len(prompt) // 4
    print(f"  [sonnet] ~{approx_in} input tokens (est ${approx_in/1_000_000*3:.3f})", file=sys.stderr)

    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_OUT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")

    # Usage reporting
    try:
        u = resp.usage
        print(f"  [sonnet] in={u.input_tokens} out={u.output_tokens} "
              f"cache_creation={getattr(u,'cache_creation_input_tokens',0)} "
              f"cache_read={getattr(u,'cache_read_input_tokens',0)}", file=sys.stderr)
    except Exception:
        pass

    # Extract top-level JSON object — allow markdown fencing.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"no JSON object in Sonnet output:\n{text[:500]}")
    raw = json.loads(m.group(0))

    events_raw = raw.get("events", [])
    checkpoints: list[Checkpoint] = []
    expire_ids: list[int] = []
    for ev in events_raw:
        if not isinstance(ev, dict):
            continue
        # Element must have exactly one known key.
        known_keys = [k for k in ("expire", "checkpoint") if k in ev]
        if len(known_keys) != 1:
            print(f"  [sonnet] WARN: malformed event {ev}", file=sys.stderr)
            continue
        key = known_keys[0]
        if key == "expire":
            try:
                expire_ids.append(int(ev["expire"]))
            except (TypeError, ValueError):
                print(f"  [sonnet] WARN: non-int expire {ev}", file=sys.stderr)
        else:  # checkpoint
            cp = ev["checkpoint"]
            if not isinstance(cp, dict):
                continue
            try:
                checkpoints.append(Checkpoint(
                    after_step=int(cp["after_step"]),
                    progress_type=cp["progress_type"],
                    finding=str(cp.get("finding", "")),
                    evidence=str(cp.get("evidence", "")),
                    next_direction=str(cp.get("next_direction", "")),
                ))
            except (KeyError, TypeError, ValueError) as e:
                print(f"  [sonnet] WARN: malformed checkpoint {e}: {cp}", file=sys.stderr)
    return checkpoints, expire_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    # Convenience: source .env if ANTHROPIC_API_KEY isn't set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        env_file = Path(__file__).resolve().parent.parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
                    break

    with open(args.inp) as f:
        session = Session.from_dict(json.load(f))

    checkpoints, expire_ids = call_sonnet(session)
    session.set_annotations(checkpoints, expire_ids)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(session.to_dict(), f, indent=2)
    print(f"{args.inp.name}: {len(checkpoints)} checkpoint(s), {len(expire_ids)} expire(s) → {args.out.name}")
    for cp in checkpoints:
        print(f"  step {cp.after_step:3d}  {cp.progress_type:22s}  {cp.finding[:100]}")
    if expire_ids:
        print(f"  expire_step_ids: {expire_ids}")


if __name__ == "__main__":
    main()
