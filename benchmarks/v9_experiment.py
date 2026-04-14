#!/usr/bin/env python3
"""
v9 relational features — LR prototype + evaluation vs Sonnet labels.

This is a self-contained experiment to validate the v9 feature design
before committing to full pipeline changes. If LR AUC on Sonnet labels
beats v5-baseline features with this 34-feature relational shape, we
port to extract_features.py + train.py as schema 6.

Feature layout (see discussion):

  Input: 34 features = 5 previous steps (6 features each) + current (4 features)

  Previous step features (6 × 5 history = 30):
    action_match         — 1.0 if prior.action == current.action
    target_file_match    — 1.0 if prior.target_file == current.target_file
    target_scope_match   — 1.0 if prior.target_scope == current.target_scope
    output_similarity    — Jaccard of prior's output vs its own last match (self-relative)
    output_length        — log1p(lines) of prior's output
    is_error             — prior had error indicators

  Current step features (4):
    output_length                — log1p(lines) of current output
    is_error                     — current has error indicators
    output_similarity_vs_match   — Jaccard of current output vs last
                                   (action, target_file) match in history
    consecutive_match_count      — normalized count of last 5 steps with
                                   (action_match AND target_file_match)

All features are RELATIONAL or intrinsic to the current step. No absolute
cmd_hash or file_hash. Match detection is precomputed rather than learned
from CRC32-normalized pseudo-random floats.

Usage:
  .venv/bin/python benchmarks/v9_experiment.py
  .venv/bin/python benchmarks/v9_experiment.py --task 03_llvm_loop_vec --verbose
  .venv/bin/python benchmarks/v9_experiment.py --compare  # vs v5 baseline features
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.pipeline.parsers.nlile import parse_session  # noqa: E402

# ─── Constants & helpers shared with extract_features.py ───────────────────

SILENT_CMD_RE = re.compile(r"^(cd|pushd|popd|source|export|set|unset|alias|ulimit|umask)\b")
FILE_EXT_RE = re.compile(r"\.[a-zA-Z]{1,8}$")
SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL | re.I)
ERROR_PATTERNS = re.compile(
    r"(error|traceback|exception|failed|failure|fatal|cannot|unable to|not found|permission denied"
    r"|segmentation fault|core dumped|FAIL|ModuleNotFoundError|ImportError|SyntaxError"
    r"|TypeError|ValueError|KeyError|AttributeError|RuntimeError|FileNotFoundError)",
    re.I,
)
PATH_TOKEN_RE = re.compile(r"(?:/?[\w@.\-]+/)+[\w@.\-]+(?:\.[a-zA-Z0-9_]{1,8})?|[\w@.\-]+\.[a-zA-Z0-9_]{1,8}")
MAX_OUTPUT_LINES = 100
N_HISTORY = 5
SCOPE_DEPTH = 5  # /a/b/c/d → 5 components including empty head from leading slash


# ─── Core v9 feature extraction ────────────────────────────────────────────

def _strip_reminders(text: str) -> str:
    if not text or "<system-reminder" not in text:
        return text
    return SYSTEM_REMINDER_RE.sub("", text)


def _normalize_to_set(output: str) -> frozenset:
    if not output:
        return frozenset()
    lines = output.strip().split("\n")[:MAX_OUTPUT_LINES]
    out = set()
    for line in lines:
        line = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", line)
        line = re.sub(r"\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}", "TIMESTAMP", line)
        line = re.sub(r"pid[=: ]\d+", "pid=PID", line, flags=re.I)
        line = re.sub(r"/tmp/[^\s]+", "/tmp/TMPFILE", line)
        line = re.sub(r"\d+\.\d{3,}s", "N.NNNs", line)
        line = line.strip()
        if line:
            out.add(line)
    return frozenset(out)


def _jaccard(a: frozenset, b: frozenset | None) -> float:
    if not b:
        return 0.0
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


# Subcommand tokens must look like program names or identifiers, nothing
# with quotes / special chars / path separators. Catches `git log`, `make test`,
# `ninja opt`, `cargo build`, `npm run`. Rejects `node -e "const qs..."` where
# the -e flag value is a code string.
_SUBCOMMAND_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_\-]*$")

# Programs that take a code/script argument after a short flag. When we see
# one of these with the flag, treat the program alone as the action — don't
# try to extract a subcommand from the code string.
_PROGS_WITH_INLINE_SCRIPT = {"node", "python", "python3", "ruby", "perl",
                             "sh", "bash", "zsh", "fish", "awk", "sed", "tclsh"}
_INLINE_SCRIPT_FLAGS = {"-e", "-c", "--command", "--eval", "-p", "-P"}


def _action_of(step: dict) -> tuple:
    """Return a hashable 'action' tuple. Used for action_match comparison.

    Non-bash tools: (tool_category, tool_name) — e.g. ('search', 'Grep').
    Grep, Glob, Read, Edit, etc. match on tool identity regardless of
    their pattern/file arguments.

    Bash: ('bash', program, subcommand?) where:
      - program is the basename of the first non-silent token in the first
        pipeline of the command
      - subcommand is the first non-flag, non-path, identifier-shaped token
        after the program, when one exists and isn't a script argument.
        Examples: ('bash', 'git', 'log'), ('bash', 'make', 'test').
      - Programs that take inline scripts via -e/-c (node -e "code",
        python -c "code") return just ('bash', program) to avoid matching
        on fragments of the quoted script.
    """
    tool = step.get("tool", "other")
    cmd = step.get("cmd", "") or ""
    if tool != "bash":
        return (tool, (step.get("tool_name") or tool))

    parts = re.split(r"\s*(?:&&|;)\s*", cmd.strip())
    real = [p for p in parts if p.strip() and not SILENT_CMD_RE.match(p.strip())]
    if not real:
        tokens = cmd.strip().split()
        return ("bash", tokens[0] if tokens else "")

    first_pipe = re.split(r"\s*\|\s*", real[0].strip())[0]
    tokens = first_pipe.strip().split()
    if not tokens:
        return ("bash", "")

    prog = tokens[0].rsplit("/", 1)[-1]

    # Programs that take inline code via flag: skip subcommand detection.
    if prog in _PROGS_WITH_INLINE_SCRIPT:
        has_inline_flag = any(t in _INLINE_SCRIPT_FLAGS for t in tokens[1:])
        if has_inline_flag:
            return ("bash", prog)

    # Subcommand detection: only accept if it comes IMMEDIATELY after the
    # program, before any flag. This avoids mistaking flag arguments for
    # subcommands (e.g. `ninja -C build opt` → ('bash', 'ninja'), not
    # ('bash', 'ninja', 'build'), because `-C` takes `build` as its value).
    # Side effect: loses subcommand info when flags come first, e.g.
    # `make -j4 test` → ('bash', 'make'). That's an acceptable loss — we
    # gain robustness against flag-arg misidentification.
    subcommand = None
    if len(tokens) >= 2:
        tok = tokens[1]
        if (not tok.startswith("-")
                and "/" not in tok
                and "." not in tok
                and _SUBCOMMAND_RE.match(tok)):
            subcommand = tok
    return ("bash", prog, subcommand) if subcommand else ("bash", prog)


def _target_file_of(step: dict) -> str | None:
    """Return the primary resource identifier for match_file comparison.
    Prefers the structured `file` field; falls back to the first path-like
    token in the command string; falls back to the raw cmd for Grep/Glob.
    """
    if step.get("file"):
        return str(step["file"])
    tool = step.get("tool", "other")
    cmd = step.get("cmd", "") or ""
    if tool == "bash":
        m = PATH_TOKEN_RE.search(cmd)
        if m:
            return m.group(0)
        return None
    # Native tool with no file field (e.g. Grep with pattern only)
    return cmd or None


def _target_scope_of(step: dict) -> str | None:
    """Return a coarser scope identifier — directory prefix, depth SCOPE_DEPTH.
    Catches 'same subtree via different files'.
    """
    tf = _target_file_of(step)
    if not tf:
        return None
    if "/" not in tf:
        return None  # no directory info
    parts = tf.split("/")
    # Walk up from the file to the last directory component, then take a
    # prefix of depth SCOPE_DEPTH. For /scratch/llvm/lib/Transforms/Vectorize/VPlan.cpp
    # this gives /scratch/llvm/lib/Transforms/Vectorize.
    if len(parts) <= SCOPE_DEPTH:
        # Drop the filename, keep the directory path
        return "/".join(parts[:-1]) if len(parts) > 1 else None
    return "/".join(parts[:SCOPE_DEPTH])


def _has_error(output: str) -> bool:
    if not output:
        return False
    return bool(ERROR_PATTERNS.search(output[:2000]))


@dataclass
class StepInfo:
    """Everything we need to know about a step to compute features for it
    and for steps that reference it later."""
    action: tuple
    target_file: str | None
    target_scope: str | None
    output_set: frozenset
    output_length: float
    is_error: float
    # Self-relative: the Jaccard of THIS step's output_set vs the most recent
    # prior step that matched on (action, target_file). Computed at the moment
    # this step was processed — captures 'was this step itself a repeat'.
    self_relative_sim: float


def compute_v9_features(steps: list[dict]) -> list[list[float]]:
    """Produce one 34-dim feature vector per step.

    Side effects: none — pure function of the step sequence.
    """
    # First pass: normalize each step into a StepInfo with its self-relative
    # output similarity computed against its own history.
    history_by_match_key: dict[tuple, list[frozenset]] = {}  # (action, target_file) -> recent outputs
    infos: list[StepInfo] = []

    for step in steps:
        action = _action_of(step)
        target_file = _target_file_of(step)
        target_scope = _target_scope_of(step)
        clean = _strip_reminders(step.get("output", "") or "")
        output_set = _normalize_to_set(clean)
        match_key = (action, target_file)
        priors = history_by_match_key.get(match_key, [])
        self_sim = 0.0
        for p in priors:
            j = _jaccard(output_set, p)
            if j > self_sim:
                self_sim = j
                if self_sim >= 1.0:
                    break
        infos.append(StepInfo(
            action=action,
            target_file=target_file,
            target_scope=target_scope,
            output_set=output_set,
            output_length=math.log1p(clean.count("\n")),
            is_error=1.0 if _has_error(clean) else 0.0,
            self_relative_sim=self_sim,
        ))
        # Update history AFTER recording so self_sim doesn't match against self
        slots = history_by_match_key.setdefault(match_key, [])
        slots.append(output_set)
        if len(slots) > 5:
            slots.pop(0)

    # Second pass: for each step T, build its 34-dim vector by looking at
    # infos[T-1..T-5] and comparing against infos[T].
    result: list[list[float]] = []
    for t, cur in enumerate(infos):
        vec: list[float] = []
        # 5 prior slots, most recent first (T-1, T-2, ..., T-5)
        prior_action_match = [0.0] * N_HISTORY
        prior_file_match = [0.0] * N_HISTORY
        for slot in range(N_HISTORY):
            idx = t - 1 - slot
            if idx < 0:
                # Zero padding: all 6 features 0
                vec.extend([0.0] * 6)
                continue
            prior = infos[idx]
            am = 1.0 if prior.action == cur.action else 0.0
            fm = 1.0 if (prior.target_file is not None and
                         prior.target_file == cur.target_file) else 0.0
            sm = 1.0 if (prior.target_scope is not None and
                         prior.target_scope == cur.target_scope) else 0.0
            prior_action_match[slot] = am
            prior_file_match[slot] = fm
            vec.extend([
                am,
                fm,
                sm,
                prior.self_relative_sim,
                prior.output_length,
                prior.is_error,
            ])

        # Current-step features (4):
        #   output_length, is_error, output_similarity_vs_match, consecutive_match_count
        vec.append(cur.output_length)
        vec.append(cur.is_error)

        # output_similarity_vs_match: Jaccard of current output vs the most
        # recent prior step that matched BOTH action AND target_file.
        out_sim_match = 0.0
        for slot in range(N_HISTORY):
            idx = t - 1 - slot
            if idx < 0:
                break
            prior = infos[idx]
            if prior.action == cur.action and prior.target_file == cur.target_file and prior.target_file is not None:
                out_sim_match = _jaccard(cur.output_set, prior.output_set)
                break
        vec.append(out_sim_match)

        # consecutive_match_count: normalized count of last 5 steps matching on BOTH
        matches = sum(
            1 for slot in range(N_HISTORY)
            if prior_action_match[slot] == 1.0 and prior_file_match[slot] == 1.0
        )
        vec.append(matches / N_HISTORY)

        result.append(vec)

    return result


V9_FEATURE_COUNT = 6 * N_HISTORY + 4  # 34
V9_FEATURE_NAMES = (
    [f"p{i+1}_{k}" for i in range(N_HISTORY)
     for k in ("act_match", "file_match", "scope_match", "self_sim", "out_len", "is_err")]
    + ["cur_out_len", "cur_is_err", "cur_sim_vs_match", "cur_consec_match"]
)
assert len(V9_FEATURE_NAMES) == V9_FEATURE_COUNT


# ─── Transcript parsing & evaluation ───────────────────────────────────────

def parse_transcript_to_steps(path: Path) -> list[dict]:
    """stream-json transcript → list of normalized step dicts."""
    messages = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") in ("user", "assistant"):
            msg = ev.get("message", {})
            if isinstance(msg, dict):
                messages.append(msg)
    return parse_session(messages)


def logreg_eval(features_by_task, labels_by_task):
    """Pool features across tasks, train LR, report AUC + weights.

    Returns (auc, per_feature_weights_dict).
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X, y = [], []
    for t, feats in features_by_task.items():
        for f, lbl in zip(feats, labels_by_task[t]):
            if lbl == "UNSURE":
                continue
            X.append(f)
            y.append(1 if lbl == "STUCK" else 0)
    if not X:
        return float("nan"), {}
    Xn = np.array(X, dtype=float)
    yn = np.array(y, dtype=int)
    if yn.sum() == 0 or yn.sum() == len(yn):
        return float("nan"), {}
    maxes = np.maximum(np.abs(Xn).max(axis=0), 1e-9)
    Xn = Xn / maxes
    lr = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
    lr.fit(Xn, yn)
    probs = lr.predict_proba(Xn)[:, 1]
    auc = roc_auc_score(yn, probs)
    weights = dict(zip(V9_FEATURE_NAMES, lr.coef_[0].tolist()))
    return float(auc), weights


def per_task_logreg(features_by_task, labels_by_task):
    """Train LR on the pooled set, evaluate per-task AUC separately."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # Pool
    X, y, task_idx = [], [], []
    for i, (t, feats) in enumerate(features_by_task.items()):
        for f, lbl in zip(feats, labels_by_task[t]):
            if lbl == "UNSURE":
                continue
            X.append(f)
            y.append(1 if lbl == "STUCK" else 0)
            task_idx.append(t)
    Xn = np.array(X, dtype=float)
    yn = np.array(y, dtype=int)
    maxes = np.maximum(np.abs(Xn).max(axis=0), 1e-9)
    Xn = Xn / maxes
    lr = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
    lr.fit(Xn, yn)
    probs = lr.predict_proba(Xn)[:, 1]
    results = {}
    for tname in sorted(set(task_idx)):
        mask = np.array([i for i in range(len(task_idx)) if task_idx[i] == tname])
        if len(mask) == 0:
            continue
        tl = yn[mask]
        tp = probs[mask]
        if tl.sum() == 0 or tl.sum() == len(tl):
            results[tname] = ("no stuck/all stuck", int(tl.sum()), len(tl))
            continue
        auc = roc_auc_score(tl, tp)
        results[tname] = (f"{auc:.4f}", int(tl.sum()), len(tl))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="benchmarks/results/comparison_off")
    ap.add_argument("--task", default="all")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    all_task_dirs = sorted(d for d in run_dir.iterdir()
                           if d.is_dir() and (d / "sonnet_labels.json").exists())
    if args.task != "all":
        all_task_dirs = [d for d in all_task_dirs if d.name == args.task]

    features_by_task: dict[str, list[list[float]]] = {}
    labels_by_task: dict[str, list[str]] = {}

    for td in all_task_dirs:
        steps = parse_transcript_to_steps(td / "transcript_1.jsonl")
        if not steps:
            continue
        feats = compute_v9_features(steps)
        labels = json.loads((td / "sonnet_labels.json").read_text())["labels"]
        n = min(len(feats), len(labels))
        features_by_task[td.name] = feats[:n]
        labels_by_task[td.name] = labels[:n]

        if args.verbose:
            stuck_count = sum(1 for l in labels[:n] if l == "STUCK")
            print(f"{td.name:<25} n={n:>4} stuck={stuck_count:>3}")

    print(f"\nParsed {len(features_by_task)} tasks, {sum(len(f) for f in features_by_task.values())} total steps")
    print(f"v9 feature count: {V9_FEATURE_COUNT}")
    print()

    # Pooled LR AUC
    auc, weights = logreg_eval(features_by_task, labels_by_task)
    print(f"POOLED LR AUC (balanced): {auc:.4f}")
    print()

    # Top-magnitude weights
    sorted_w = sorted(weights.items(), key=lambda kv: -abs(kv[1]))
    print("Top 15 weights by |coefficient|:")
    for name, w in sorted_w[:15]:
        bar = "+" if w > 0 else "-"
        print(f"  {bar} {name:<28} {w:+.3f}")
    print()

    # Per-task
    per_task = per_task_logreg(features_by_task, labels_by_task)
    print("Per-task AUC (LR trained pooled, evaluated per task):")
    print(f"  {'task':<25}{'AUC':>10}{'stuck':>8}{'total':>8}")
    for tname, (auc_str, stk, tot) in per_task.items():
        print(f"  {tname:<25}{auc_str:>10}{stk:>8}{tot:>8}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
