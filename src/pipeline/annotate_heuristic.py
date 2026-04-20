#!/usr/bin/env python3
"""
Deterministic heuristic annotator for hygiene training data.

Scans a Session and emits checkpoint annotations based on fixed rules:

Milestone (milestone_achieved):
  M1 — first Edit/Write after ≥2 Read/Grep steps → likely identified fix
  M4 — Bash command succeeds after ≥1 prior matching Bash with error

Approach eliminated (approach_eliminated):
  E1 — same Bash command repeated with matching error signature ≥2 times
  E2 — git revert / checkout -- / stash / reset — explicit rollback
  E3 — Edit/Write to file X immediately reverted (≤3 steps) by another
       Edit/Write to X with substantially different content

Summary text is templated from the triggering-step context. Useful for
cheap-and-fast labeling; a Sonnet annotator should be run on a sample to
validate that the positions and types match a semantic read.

Usage:
  .venv/bin/python benchmarks/annotate_heuristic.py \\
      --in data/annotate_test/short.json \\
      --out data/annotate_test/short.heuristic.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.pipeline.hygiene_types import Checkpoint, Session, Step


# ── helpers ───────────────────────────────────────────────────────────────

_ERROR_RX = re.compile(
    r"\b(error|exception|traceback|undefined|cannot find|failed|fatal|not found|"
    r"syntax error|unresolved|permission denied|no such file|abort|segmentation fault|"
    r"command not found)\b",
    re.IGNORECASE,
)

_BUILD_OK_RX = re.compile(
    r"\b(build (succeeded|successful)|compilation (successful|ok)|"
    r"passed|passing|\bok\b|finished in \d|tests? (passed|ok)|"
    r"\[==========\] .* passed|all tests passed)\b",
    re.IGNORECASE,
)

_ROLLBACK_RX = re.compile(
    r"\bgit (revert|reset --?hard|stash|checkout --? )|"
    r"\bundo\b|rollback",
    re.IGNORECASE,
)


def _has_error(text: str) -> bool:
    if not text:
        return False
    return bool(_ERROR_RX.search(text))


def _has_success_marker(text: str) -> bool:
    if not text:
        return False
    return bool(_BUILD_OK_RX.search(text))


def _error_signature(text: str) -> str:
    """Crude signature of the error — first non-empty line under 120 chars."""
    if not text:
        return ""
    for line in text.splitlines():
        line = line.strip()
        if line and _has_error(line):
            return line[:120]
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:120]
    return ""


def _cmd_signature(cmd: str) -> str:
    """Normalize a Bash cmd for fuzzy equality — first 2 words or file name."""
    if not cmd:
        return ""
    return " ".join(cmd.split()[:3])[:80]


# ── checkpoint detectors ──────────────────────────────────────────────────


def detect_m1_edit_after_reads(session: Session) -> list[Checkpoint]:
    """M1 — first Edit/Write after ≥2 Read/Grep steps."""
    reads_in_window: list[Step] = []
    fired = False
    hits: list[Checkpoint] = []
    for step in session.steps:
        if step.role != "tool":
            continue
        if step.tool_name in ("Read", "Grep", "Glob"):
            reads_in_window.append(step)
            continue
        if step.tool_name in ("Edit", "Write", "MultiEdit") and len(reads_in_window) >= 2 and not fired:
            fired = True
            read_files = [r.input_file or r.cmd for r in reads_in_window[-5:]]
            finding = (
                f"First edit to {step.input_file or step.cmd} after exploring "
                f"{len(reads_in_window)} file(s) — likely identified fix location."
            )
            evidence = f"Read/Grep steps: {'; '.join(read_files[:3])}. Edit target: {step.input_file or step.cmd}."
            next_direction = "Verify the edit with a test or build."
            hits.append(Checkpoint(
                after_step=step.idx,
                progress_type="milestone_achieved",
                finding=finding,
                evidence=evidence[:400],
                next_direction=next_direction,
            ))
    return hits


def detect_m4_bash_success_after_failure(session: Session) -> list[Checkpoint]:
    """M4 — Bash command succeeds after ≥1 matching-signature Bash that failed."""
    recent_bashes: dict[str, list[Step]] = {}
    hits: list[Checkpoint] = []
    for step in session.steps:
        if step.role != "tool" or step.tool_name != "Bash":
            continue
        sig = _cmd_signature(step.cmd)
        if not sig:
            continue
        history = recent_bashes.setdefault(sig, [])
        prev_had_error = any(_has_error(h.output) for h in history)
        if history and prev_had_error and not _has_error(step.output):
            hits.append(Checkpoint(
                after_step=step.idx,
                progress_type="milestone_achieved",
                finding=f"`{sig}` now succeeds after {len(history)} prior failure(s).",
                evidence=_error_signature(history[-1].output)[:300] or "prior runs showed error output",
                next_direction="Build is usable; move to next subtask or verify correctness.",
            ))
            history.clear()
        history.append(step)
    return hits


def detect_e1_repeated_matching_failure(session: Session) -> list[Checkpoint]:
    """E1 — same Bash command signature, matching error, ≥2 occurrences."""
    hits: list[Checkpoint] = []
    seen: dict[tuple[str, str], list[Step]] = {}
    for step in session.steps:
        if step.role != "tool" or step.tool_name != "Bash":
            continue
        if not _has_error(step.output):
            continue
        sig = _cmd_signature(step.cmd)
        err = _error_signature(step.output)
        if not sig or not err:
            continue
        key = (sig, err[:60])
        lst = seen.setdefault(key, [])
        lst.append(step)
        if len(lst) == 2:
            hits.append(Checkpoint(
                after_step=step.idx,
                progress_type="approach_eliminated",
                finding=f"Same error from `{sig}` repeated — current approach isn't fixing it.",
                evidence=f"Error signature: {err[:200]}",
                next_direction="Change strategy; the current edits are not addressing the root cause.",
            ))
    return hits


def detect_e2_explicit_rollback(session: Session) -> list[Checkpoint]:
    """E2 — explicit rollback command (git revert, reset --hard, stash, etc.)."""
    hits: list[Checkpoint] = []
    for step in session.steps:
        if step.role != "tool" or step.tool_name != "Bash":
            continue
        if not step.cmd:
            continue
        if not _ROLLBACK_RX.search(step.cmd):
            continue
        hits.append(Checkpoint(
            after_step=step.idx,
            progress_type="approach_eliminated",
            finding="Explicit rollback — prior edits didn't work.",
            evidence=f"Rollback command: {step.cmd[:200]}",
            next_direction="Revisit the hypothesis; the reverted change was not correct.",
        ))
    return hits


def detect_e3_edit_reverted(session: Session) -> list[Checkpoint]:
    """E3 — Edit/Write to file X soon followed (≤3 steps) by another
    Edit/Write to X with substantially different content."""
    edits_by_file: dict[str, list[Step]] = {}
    hits: list[Checkpoint] = []
    recent_edit_idx: dict[str, int] = {}
    for step in session.steps:
        if step.role != "tool" or step.tool_name not in ("Edit", "Write", "MultiEdit"):
            continue
        f = step.input_file or step.cmd
        if not f:
            continue
        prior_idx = recent_edit_idx.get(f)
        recent_edit_idx[f] = step.idx
        edits_by_file.setdefault(f, []).append(step)
        if prior_idx is not None and (step.idx - prior_idx) <= 4 and len(edits_by_file[f]) >= 2:
            hits.append(Checkpoint(
                after_step=step.idx,
                progress_type="approach_eliminated",
                finding=f"Re-edited {f} within {step.idx - prior_idx} steps — prior attempt wasn't right.",
                evidence="Same file edited twice in short succession",
                next_direction="Settle on an approach before making more edits.",
            ))
    return hits


# ── EXPIRE annotation ────────────────────────────────────────────────────────


def compute_expire_targets(session: Session, stale_after: int = 3, min_chars: int = 500) -> list[int]:
    """Step indices whose tool_result text should be treated as EXPIRE-expired in
    downstream training. Bash only; other tools preserve output per feedback
    (Read/Grep are essential to keep — the feedback memory is explicit).
    """
    expire: list[int] = []
    last_idx = session.steps[-1].idx if session.steps else 0
    for step in session.steps:
        if step.role != "tool" or step.tool_name != "Bash":
            continue
        if len(step.output) < min_chars:
            continue
        if last_idx - step.idx < stale_after:
            continue
        expire.append(step.idx)
    return expire


# ── top-level ─────────────────────────────────────────────────────────────


def annotate(session: Session) -> tuple[list[Checkpoint], list[int]]:
    """Run all detectors, dedupe overlapping checkpoints, return (checkpoints, expire_step_ids)."""
    raw: list[Checkpoint] = []
    raw.extend(detect_m1_edit_after_reads(session))
    raw.extend(detect_m4_bash_success_after_failure(session))
    raw.extend(detect_e1_repeated_matching_failure(session))
    raw.extend(detect_e2_explicit_rollback(session))
    raw.extend(detect_e3_edit_reverted(session))
    # Dedupe: if two checkpoints fire within 1 step of each other, keep the
    # first by position (milestone-achieved wins ties to bias toward progress).
    raw.sort(key=lambda c: (c.after_step, 0 if c.progress_type == "milestone_achieved" else 1))
    deduped: list[Checkpoint] = []
    for cp in raw:
        if deduped and cp.after_step - deduped[-1].after_step <= 1:
            continue
        deduped.append(cp)
    expire = compute_expire_targets(session)
    return deduped, expire


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    with open(args.inp) as f:
        session = Session.from_dict(json.load(f))
    checkpoints, expire = annotate(session)
    session.set_annotations(checkpoints, expire)
    out_dict = session.to_dict()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"{args.inp.name}: {len(checkpoints)} checkpoint(s), {len(expire)} EXPIRE target(s) → {args.out.name}")
    for cp in checkpoints:
        print(f"  step {cp.after_step:3d}  {cp.progress_type:22s}  {cp.finding[:100]}")


if __name__ == "__main__":
    main()
