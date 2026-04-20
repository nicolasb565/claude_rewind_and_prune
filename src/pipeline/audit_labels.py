#!/usr/bin/env python3
"""
Audit under-labeling rate in Sonnet's checkpoint annotations.

Strategy: re-annotate a small sample of sessions with a STRICT coverage
prompt ("label every plausible milestone including minor subtask
completions"), diff against the original labels, quantify the gap.

Compares on:
- Sessions in v13b's val split (where FP rate was 91% / 10 of 11)
  → these are the sessions the adapter was accused of over-emitting on
- Random sample for comparison

Cost: ~$0.05 per session × 5-10 sessions = $0.25-0.50.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

from src.pipeline.hygiene_types import Session, render_session_for_prompt


MODEL = os.environ.get("SONNET_MODEL", "claude-sonnet-4-6")
MAX_TOKENS_OUT = 4096


STRICT_PROMPT = """You are auditing a Claude Code trajectory for potential checkpoint moments. A checkpoint is a moment where the agent could usefully consolidate progress into a short summary — it's useful both at MAJOR milestones and at MINOR subtask completions.

Examples of things that DO qualify as checkpoint moments:
- A file edit was applied and verified (cat of the change confirms)
- A test passed after a fix
- A specific hypothesis was ruled out by a specific observation
- A subdirectory was added to a build system
- A decorator was modified and the change was visible in the file
- A benchmark run completed and output was saved
- Any discrete piece of "state has changed, we've learned / done something concrete, we can now move on"

Be COMPREHENSIVE. A session that does 20 discrete things should have ~20 checkpoint moments, not 3 "major" ones. Granularity matters less than concreteness — if something concrete was achieved and the agent moved on, that's a checkpoint.

Checkpoint types:
- milestone_achieved: a concrete completion
- approach_eliminated: a specific approach was ruled out

Output a JSON object with one field `events`: an ordered list of checkpoint events (no expire events for this audit).

Checkpoint shape:
  {"checkpoint": {
    "after_step": <int>,
    "progress_type": "milestone_achieved" | "approach_eliminated",
    "finding": "<1 sentence — what is now true>",
    "evidence": "<1 sentence — the concrete observation>",
    "next_direction": "<1 sentence — what to do next>"
  }}

Session to annotate:
---
{rendered}
---

Output JSON only, no prose.
"""


MIDDLE_PROMPT = """You are annotating a Claude Code trajectory for context-hygiene training. The goal: label moments where the agent could usefully compress its working memory — points where prior exploration details are no longer load-bearing because a coherent subtask has reached stable completion.

A CHECKPOINT is a point where ALL THREE are true:
1. A discrete subtask with a clear outcome has just completed — not a single tool call, but a coherent unit of work
2. The exploration details leading up to it (Bash outputs, intermediate edits, early reads, failed attempts within that subtask) are no longer needed for downstream work
3. The agent is transitioning to meaningfully different work — not continuing the same thread

Test: "if we compressed all exploration prior to this moment into a 1-sentence summary, would we lose anything important for the rest of the session?" If NO → checkpoint. If YES (we still need details from earlier) → not a checkpoint yet.

Checkpoint types:
- milestone_achieved: a concrete coherent subtask completed
- approach_eliminated: a specific hypothesis ruled out by a specific observation, and the agent has moved on to a different approach

DO mark these as checkpoints:
- A feature was implemented AND verified (edit succeeded + test/build confirms)
- A bug was diagnosed (root cause identified, the investigation itself can be compressed)
- An approach failed AND the agent pivoted to a new direction (not just encountered an error and retried)
- A subsystem was understood well enough that the exploration of it is no longer needed
- A deliverable artifact was completed (README written, config finalized, benchmark report saved)

DO NOT mark these as checkpoints:
- Single file edits within a larger task
- Intermediate tool calls in an ongoing investigation
- Reads or greps that produced info but didn't close out a subtask
- Minor tweaks (cosmetic changes, formatting) unless they were the WHOLE subtask
- Moments where we learned something but are still in the same thread of work
- Errors that the agent is still actively debugging

Aim for granularity that matches "coherent subtask" — a typical 50-step session might have 4-10 checkpoints, not 2 and not 20.

Output a JSON object with one field `events`: an ordered list of checkpoint events.

Checkpoint shape:
  {"checkpoint": {
    "after_step": <int>,
    "progress_type": "milestone_achieved" | "approach_eliminated",
    "finding": "<1 sentence — what is now true>",
    "evidence": "<1 sentence — the concrete observation>",
    "next_direction": "<1 sentence — what to do next>"
  }}

Session to annotate:
---
{rendered}
---

Output JSON only, no prose.
"""


PROMPTS = {"strict": STRICT_PROMPT, "middle": MIDDLE_PROMPT}


def call_strict_sonnet(session: Session, prompt_style: str = "strict") -> list[dict]:
    """Re-annotate a session with the chosen prompt. Return new checkpoint events."""
    import anthropic
    client = anthropic.Anthropic()

    rendered = render_session_for_prompt(session)
    prompt_text = PROMPTS[prompt_style].replace("{rendered}", rendered)

    msg = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS_OUT,
        messages=[{"role": "user", "content": prompt_text}],
    )
    usage = msg.usage
    print(f"  [sonnet] in={usage.input_tokens} out={usage.output_tokens}", file=sys.stderr)

    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        print(f"  WARN no JSON in output", file=sys.stderr)
        return []
    try:
        obj = json.loads(m.group(0))
    except Exception as e:
        print(f"  WARN parse error: {e}", file=sys.stderr)
        return []
    events = obj.get("events", [])
    return [e["checkpoint"] for e in events if isinstance(e, dict) and "checkpoint" in e]


def original_checkpoints(session_raw: dict) -> list[dict]:
    cps = []
    for ev in session_raw.get("events", []):
        if isinstance(ev, dict) and "checkpoint" in ev:
            cps.append(ev["checkpoint"])
    return cps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path,
                    default=Path("data/generated/hygiene_v1.jsonl"))
    ap.add_argument("--session-ids", type=Path, default=None,
                    help="File with one session_id per line (focus audit here)")
    ap.add_argument("--n", type=int, default=5,
                    help="How many sessions to audit (default 5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("data/generated/label_audit.jsonl"))
    ap.add_argument("--prompt-style", choices=list(PROMPTS.keys()), default="strict")
    args = ap.parse_args()

    # Load env
    if not os.environ.get("ANTHROPIC_API_KEY"):
        env = Path(".env")
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"')

    # Load all sessions
    sessions_raw = []
    with args.inp.open() as f:
        for line in f:
            sessions_raw.append(json.loads(line))
    print(f"loaded {len(sessions_raw)} annotated sessions", file=sys.stderr)

    # Filter to target sessions if session-ids provided
    if args.session_ids and args.session_ids.exists():
        wanted = set(args.session_ids.read_text().strip().split("\n"))
        sessions_raw = [s for s in sessions_raw if s.get("session_id") in wanted]
        print(f"filtered to {len(sessions_raw)} target sessions", file=sys.stderr)

    # Random sample
    random.Random(args.seed).shuffle(sessions_raw)
    sample = sessions_raw[:args.n]

    results = []
    orig_totals = 0
    strict_totals = 0
    for i, raw in enumerate(sample):
        sid = raw.get("session_id")
        print(f"\n[{i+1}/{len(sample)}] auditing {sid}", file=sys.stderr)
        try:
            session = Session.from_dict(raw)
        except Exception as e:
            print(f"  skip, parse error: {e}", file=sys.stderr)
            continue

        orig_cps = original_checkpoints(raw)
        print(f"  original labels: {len(orig_cps)} checkpoints", file=sys.stderr)

        try:
            strict_cps = call_strict_sonnet(session, prompt_style=args.prompt_style)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        print(f"  strict labels:   {len(strict_cps)} checkpoints", file=sys.stderr)

        # Diff by after_step
        orig_steps = {c["after_step"] for c in orig_cps}
        strict_steps = {c["after_step"] for c in strict_cps}
        added = strict_steps - orig_steps
        kept = strict_steps & orig_steps
        dropped = orig_steps - strict_steps
        print(f"  diff: +{len(added)} new / {len(kept)} kept / -{len(dropped)} dropped", file=sys.stderr)

        orig_totals += len(orig_cps)
        strict_totals += len(strict_cps)
        results.append({
            "session_id": sid,
            "n_steps": len(session.steps),
            "original_checkpoints": len(orig_cps),
            "strict_checkpoints": len(strict_cps),
            "added_steps": sorted(added),
            "kept_steps": sorted(kept),
            "dropped_steps": sorted(dropped),
            "strict_events": strict_cps,
        })

    # Summary
    print("\n" + "=" * 60, file=sys.stderr)
    print(f"AUDIT RESULTS ({len(results)} sessions)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"original total: {orig_totals} checkpoints", file=sys.stderr)
    print(f"strict total:   {strict_totals} checkpoints", file=sys.stderr)
    if orig_totals:
        print(f"ratio: {strict_totals / orig_totals:.2f}× more labels under strict prompt", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(results)} audit rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
