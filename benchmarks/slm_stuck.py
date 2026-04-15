#!/usr/bin/env python3
"""
Small-LM zero-shot / few-shot stuck classifier evaluation.

For each step in the OOD benchmark, builds a prompt containing:
  - Task description (same framing as the Sonnet labeler prompt)
  - Optional few-shot examples
  - Last N context steps
  - The target step to classify
and calls ollama to get a P/S/U label. Reports pooled P/R/F1/AUC vs
Sonnet ground truth.

Usage:
  # Zero-shot, all sizes
  .venv/bin/python benchmarks/slm_stuck.py --models qwen3:0.6b qwen3:1.7b qwen3:4b qwen3:8b

  # Few-shot with 8 examples
  .venv/bin/python benchmarks/slm_stuck.py --models qwen3:1.7b --few-shot 8

  # Single task only (fast iteration)
  .venv/bin/python benchmarks/slm_stuck.py --models qwen3:1.7b --task 03_llvm_loop_vec
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.pipeline.parsers.nlile import parse_session  # noqa: E402
from src.pipeline.label_session import _render_step  # noqa: E402

RUN_DIR = REPO / "benchmarks" / "results" / "comparison_off"
# OpenAI-compatible endpoint exposed by llama-server from a local
# llama.cpp build. Ollama's /api/chat has well-documented sampling bugs on
# Qwen3.5 / Gemma4 / other "new architecture" models; we stopped trusting it.
LLAMA_SERVER_URL = os.environ.get(
    "LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1/chat/completions"
)

CONTEXT_STEPS = 5           # last N steps shown before the target
MAX_OUTPUT_CHARS = 400      # truncate each step's output in the prompt
REQUEST_TIMEOUT = 120       # seconds per ollama call

# ── Prompt assembly ────────────────────────────────────────────────────────

_TASK_DESCRIPTION_SHORT = """\
You are evaluating a Claude Code coding session for stuck detection.

Classify the TARGET step (the last step shown) as exactly one letter:
  P - productive: the agent is making progress (new approach, first-time \
action, finding new bugs, legitimate iterative build/test loops with fixes \
in between)
  S - stuck: the agent is in a loop (same command/error/edit repeated \
without progress, trying the same thing from different angles)
  U - unsure: genuine ambiguity that cannot be resolved from the context

Rules:
- First attempt at any command → P
- Legitimate build/test iteration where fixes are being made between runs → P
- Same command with same error twice or more, no visible change → S
- Different approach or tool after a failure → P (first step of the new approach)
- Agent hitting the same underlying error from different files/angles → S

Output format: reply with a single character — P, S, or U — and NOTHING ELSE."""

# Adapted from src/pipeline/label_session.py SYSTEM_PROMPT (the one Sonnet
# used to produce ground-truth labels). Only the output format line was
# changed — original asked for comma-separated per-step labels across the
# whole transcript, we adapt to single-step windowed classification.
_TASK_DESCRIPTION_SONNET = """\
You are labeling steps in a Claude Code session. Each step is one tool call.
Classify the TARGET step (the last one shown) as P (productive), S (stuck), or U (unsure).

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
- Reading a file already read (same path appears earlier in the transcript) → S
- Tight compile/test loop with unchanged failure → S

Transition rules:
- The first step of a repeating pattern is still P; label S when repetition begins
- The first step after escaping a loop (new approach, new tool) is P again

Output: one letter — P, S, or U — nothing else."""

# Default — can be swapped via --system-prompt CLI flag
_TASK_DESCRIPTION = _TASK_DESCRIPTION_SHORT

# Whole-session prompt: matches Sonnet's exact task framing. Ask the model
# to label every step in the transcript in one response, comma-separated.
_WHOLE_SESSION_TASK = """\
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
- Reading a file already read (same path appears earlier in the transcript) → S
- Tight compile/test loop with unchanged failure → S

Transition rules:
- The first step of a repeating pattern is still P; label S when repetition begins
- The first step after escaping a loop (new approach, new tool) is P again

Output: one label per step, comma-separated, nothing else.
Example: P,P,S,S,S,P,P,S,P"""

_FEW_SHOT_HEADER = "Here are labeled examples to calibrate your judgment:\n"


def _render_step_compact(step: dict, i: int, include_index: bool = True) -> str:
    """Compact single-step rendering, truncated output.

    When include_index is False, the leading '[N] Tool' line becomes just
    'Tool' — the numbered form confuses small models into outputting step
    indices ('1,2,3,...') instead of labels ('P,S,U,...') when asked for
    a comma-separated list.
    """
    rendered = _render_step(step, i)
    lines = rendered.split("\n")
    out_lines = []
    for ln in lines:
        if not include_index and ln.startswith("["):
            # "[0] Bash" → "Bash"
            close = ln.find("]")
            if close != -1:
                ln = ln[close + 1:].lstrip()
        if ln.startswith("  → ") and len(ln) > MAX_OUTPUT_CHARS:
            out_lines.append(ln[:MAX_OUTPUT_CHARS] + " [...]")
        else:
            out_lines.append(ln)
    return "\n".join(out_lines)


def build_prompt(
    context_steps: list[dict],
    target_step: dict,
    target_index: int,
    few_shot_examples: list[dict] | None = None,
) -> str:
    """Build the full prompt string for one classification decision.

    Each few_shot_example is a dict: {"context": [steps], "target": step, "label": "P"|"S"|"U"}
    """
    parts = [_TASK_DESCRIPTION, ""]

    if few_shot_examples:
        parts.append(_FEW_SHOT_HEADER)
        for ex_idx, ex in enumerate(few_shot_examples):
            parts.append(f"--- Example {ex_idx + 1} ---")
            if ex.get("context"):
                parts.append("Context:")
                for i, s in enumerate(ex["context"]):
                    parts.append(_render_step_compact(s, i))
            parts.append("TARGET step:")
            parts.append(_render_step_compact(ex["target"], len(ex.get("context", []))))
            parts.append(f"Label: {ex['label']}")
            parts.append("")
        parts.append("--- Now classify this session ---")
        parts.append("")

    if context_steps:
        parts.append("Context (prior steps):")
        start = target_index - len(context_steps)
        for offset, s in enumerate(context_steps):
            parts.append(_render_step_compact(s, start + offset))
    parts.append("")
    parts.append("TARGET step:")
    parts.append(_render_step_compact(target_step, target_index))
    parts.append("")
    parts.append("Your single-character label (P, S, or U):")
    return "\n".join(parts)


# ── Ollama client ──────────────────────────────────────────────────────────

def build_whole_session_prompt(steps: list[dict]) -> str:
    """Build a prompt asking the model to label every step in the session at once.

    Each step is separated by '--- STEP N ---' marker to make the per-step
    correspondence explicit, and the output format is demanded with high
    specificity to prevent the model from outputting step indices instead
    of labels.
    """
    parts = [_WHOLE_SESSION_TASK, "", f"Transcript ({len(steps)} tool calls):", ""]
    for i, s in enumerate(steps):
        parts.append(f"--- STEP {i+1} ---")
        parts.append(_render_step_compact(s, i, include_index=False))
    parts.append("")
    parts.append(f"Now output exactly {len(steps)} labels (P, S, or U), one for "
                 f"each of the {len(steps)} steps above, in order, separated by commas.")
    parts.append("Do NOT output step numbers. Each label is a single letter: P, S, or U.")
    parts.append(f"Example for a 5-step session: P,P,S,S,P")
    parts.append("")
    parts.append(f"Your {len(steps)} labels:")
    return "\n".join(parts)


def call_ollama(model: str, prompt: str, num_predict: int = 16,
                num_ctx: int = 32768) -> str:
    """Call llama-server /v1/chat/completions (OpenAI-compatible).

    Function name preserved for backwards compat in the rest of the file.
    `model` is ignored — the server loads one model at startup and serves it.
    """
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_k": 1,
        "max_tokens": num_predict,
        "repetition_penalty": 1.0,
    }
    r = requests.post(LLAMA_SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    choices = data.get("choices", [])
    if not choices:
        return ""
    return (choices[0].get("message", {}).get("content") or "").strip()


def parse_label(response: str) -> str:
    """Extract P/S/U label from ollama response (first matching letter)."""
    if not response:
        return "U"
    # Strip thinking tags if present (qwen3 often thinks)
    if "</think>" in response:
        response = response.split("</think>", 1)[1]
    upper = response.upper()
    for ch in upper:
        if ch in ("P", "S", "U"):
            return ch
    return "U"


def parse_label_sequence(response: str, expected_count: int) -> list[str]:
    """Extract a sequence of P/S/U labels from a comma-separated response.

    Returns exactly `expected_count` labels, padding with 'U' if short or
    truncating if long.
    """
    if not response:
        return ["U"] * expected_count
    if "</think>" in response:
        response = response.split("</think>", 1)[1]
    upper = response.upper()
    # Extract all P/S/U characters in order (ignore commas, whitespace, garbage)
    labels: list[str] = []
    for ch in upper:
        if ch in ("P", "S", "U"):
            labels.append(ch)
            if len(labels) >= expected_count:
                break
    while len(labels) < expected_count:
        labels.append("U")
    return labels


# ── Benchmark loading ──────────────────────────────────────────────────────

def parse_transcript(path: Path) -> list[dict]:
    messages = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") in ("user", "assistant"):
            m = ev.get("message", {})
            if isinstance(m, dict):
                messages.append(m)
    return parse_session(messages)


def load_ood_tasks(task_filter: str | None = None):
    """Return list of (task_name, steps, labels)."""
    tasks = []
    for td in sorted(RUN_DIR.iterdir()):
        if not td.is_dir():
            continue
        if task_filter and task_filter != td.name:
            continue
        t = td / "transcript_1.jsonl"
        lp = td / "sonnet_labels.json"
        if not (t.exists() and lp.exists()):
            continue
        steps = parse_transcript(t)
        labels = json.loads(lp.read_text())["labels"]
        n = min(len(steps), len(labels))
        tasks.append((td.name, steps[:n], labels[:n]))
    return tasks


# ── Few-shot example bank (hand-picked from common patterns) ──────────────

# Tiny curated few-shot set. Each example is a 1-step-context + target.
# Hand-constructed to illustrate the core patterns without needing TF-IDF
# retrieval on first pass. Can be replaced with nlile-retrieved examples.
FEW_SHOT_BANK: list[dict] = [
    {
        "context": [
            {"tool_name": "Bash", "tool": "bash",
             "cmd": "make", "output": "gcc -c foo.c\nfoo.c:10:5: error: 'x' undeclared"},
        ],
        "target": {"tool_name": "Bash", "tool": "bash",
                   "cmd": "make", "output": "gcc -c foo.c\nfoo.c:10:5: error: 'x' undeclared"},
        "label": "S",
    },
    {
        "context": [
            {"tool_name": "Bash", "tool": "bash",
             "cmd": "make", "output": "gcc -c foo.c\nfoo.c:10:5: error: 'x' undeclared"},
            {"tool_name": "Edit", "tool": "edit",
             "cmd": "foo.c", "output": "edit applied"},
        ],
        "target": {"tool_name": "Bash", "tool": "bash",
                   "cmd": "make", "output": "gcc -c foo.c\nfoo.c:12:3: error: missing semicolon"},
        "label": "P",
    },
    {
        "context": [
            {"tool_name": "Grep", "tool": "search",
             "cmd": "fn parse_json", "output": "src/parser.rs:42: fn parse_json..."},
        ],
        "target": {"tool_name": "Read", "tool": "view",
                   "cmd": "src/parser.rs", "output": "pub fn parse_json..."},
        "label": "P",
    },
    {
        "context": [
            {"tool_name": "Bash", "tool": "bash",
             "cmd": "cargo test", "output": "test foo::bar ... FAILED\nexpected 1, got 2"},
            {"tool_name": "Bash", "tool": "bash",
             "cmd": "cargo test", "output": "test foo::bar ... FAILED\nexpected 1, got 2"},
        ],
        "target": {"tool_name": "Bash", "tool": "bash",
                   "cmd": "cargo test", "output": "test foo::bar ... FAILED\nexpected 1, got 2"},
        "label": "S",
    },
    {
        "context": [
            {"tool_name": "Read", "tool": "view",
             "cmd": "src/main.py", "output": "def main():\n    ..."},
            {"tool_name": "Bash", "tool": "bash",
             "cmd": "python src/main.py", "output": "ModuleNotFoundError: No module named 'foo'"},
        ],
        "target": {"tool_name": "Bash", "tool": "bash",
                   "cmd": "pip install foo", "output": "Successfully installed foo-1.2"},
        "label": "P",
    },
    {
        "context": [
            {"tool_name": "Grep", "tool": "search",
             "cmd": "TODO", "output": "src/a.py:1: TODO"},
            {"tool_name": "Grep", "tool": "search",
             "cmd": "FIXME", "output": "src/a.py:2: FIXME"},
        ],
        "target": {"tool_name": "Grep", "tool": "search",
                   "cmd": "TODO", "output": "src/a.py:1: TODO"},
        "label": "S",
    },
    {
        "context": [
            {"tool_name": "Bash", "tool": "bash",
             "cmd": "cmake --build build", "output": "Building target foo... 45% done"},
        ],
        "target": {"tool_name": "Bash", "tool": "bash",
                   "cmd": "cmake --build build", "output": "Building target bar... 67% done"},
        "label": "P",
    },
    {
        "context": [
            {"tool_name": "Edit", "tool": "edit",
             "cmd": "foo.py", "output": "edit applied"},
            {"tool_name": "Bash", "tool": "bash",
             "cmd": "pytest", "output": "test_foo ... FAIL: AssertionError"},
            {"tool_name": "Edit", "tool": "edit",
             "cmd": "foo.py", "output": "edit applied"},
            {"tool_name": "Bash", "tool": "bash",
             "cmd": "pytest", "output": "test_foo ... FAIL: AssertionError"},
        ],
        "target": {"tool_name": "Edit", "tool": "edit",
                   "cmd": "foo.py", "output": "edit applied"},
        "label": "S",
    },
]


# ── Evaluation loop ────────────────────────────────────────────────────────

def evaluate_model_whole_session(model: str, tasks, verbose: bool = False):
    """Label each full transcript in one shot (matches Sonnet's exact task)."""
    print(f"\n========== {model} (WHOLE_SESSION, tasks={len(tasks)}) ==========")

    all_pred, all_gold = [], []
    per_task = []
    t_start = time.time()
    n_calls = 0

    for task_name, steps, labels in tasks:
        prompt = build_whole_session_prompt(steps)
        # Budget: ~2 tokens per step (letter + comma) plus slack
        num_pred = max(64, 4 * len(steps))
        try:
            resp = call_ollama(model, prompt, num_predict=num_pred)
        except requests.RequestException as e:
            print(f"  ERROR calling ollama on {task_name}: {e}")
            resp = ""
        task_pred = parse_label_sequence(resp, len(steps))
        n_calls += 1

        tp = sum(1 for p, g in zip(task_pred, labels) if p == "S" and g == "STUCK")
        fp = sum(1 for p, g in zip(task_pred, labels) if p == "S" and g == "PRODUCTIVE")
        fn = sum(1 for p, g in zip(task_pred, labels) if p != "S" and g == "STUCK")
        n_stk = sum(1 for g in labels if g == "STUCK")
        per_task.append({
            "task": task_name, "n": len(steps), "stk": n_stk,
            "tp": tp, "fp": fp, "fn": fn,
        })
        for p, g in zip(task_pred, labels):
            if g == "UNSURE":
                continue
            all_pred.append(1 if p == "S" else 0)
            all_gold.append(1 if g == "STUCK" else 0)

    elapsed = time.time() - t_start
    print(f"{'task':<22}{'n':>5}{'stk':>5}{'tp':>5}{'fp':>5}{'fn':>5}")
    for r in per_task:
        print(f"{r['task']:<22}{r['n']:>5}{r['stk']:>5}"
              f"{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}")

    arr_p = np.array(all_pred); arr_g = np.array(all_gold)
    if len(arr_g) > 0 and 0 < arr_g.sum() < len(arr_g):
        auc = roc_auc_score(arr_g, arr_p)
    else:
        auc = float("nan")
    tp = int(((arr_p == 1) & (arr_g == 1)).sum())
    fp = int(((arr_p == 1) & (arr_g == 0)).sum())
    fn = int(((arr_p == 0) & (arr_g == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    s_per_session = elapsed / max(n_calls, 1)
    print(f"\nPOOLED  AUC={auc:.4f}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
          f"TP={tp} FP={fp} FN={fn}  ({s_per_session:.1f}s/session)")

    return {
        "model": model, "few_shot": 0, "mode": "whole_session",
        "auc": float(auc), "precision": float(prec), "recall": float(rec),
        "f1": float(f1), "tp": tp, "fp": fp, "fn": fn,
        "ms_per_call": s_per_session * 1000,
    }


def evaluate_model(model: str, tasks, few_shot_n: int, verbose: bool = False):
    few_shot = FEW_SHOT_BANK[:few_shot_n] if few_shot_n > 0 else None
    print(f"\n========== {model} "
          f"(few_shot={few_shot_n}, tasks={len(tasks)}) ==========")

    all_pred, all_gold = [], []
    per_task = []
    t_start = time.time()
    n_calls = 0

    for task_name, steps, labels in tasks:
        task_pred = []
        for i, step in enumerate(steps):
            context = steps[max(0, i - CONTEXT_STEPS):i]
            prompt = build_prompt(context, step, i, few_shot)
            try:
                resp = call_ollama(model, prompt)
            except requests.RequestException as e:
                print(f"  ERROR calling ollama on step {i}: {e}")
                resp = ""
            label = parse_label(resp)
            task_pred.append(label)
            n_calls += 1
            if verbose:
                gold = labels[i]
                mark = "✓" if (label == "S" and gold == "STUCK") or \
                              (label == "P" and gold == "PRODUCTIVE") else "✗"
                print(f"  [{task_name}:{i}] pred={label} gold={gold} {mark}")

        # Per-task metrics
        tp = sum(1 for p, g in zip(task_pred, labels) if p == "S" and g == "STUCK")
        fp = sum(1 for p, g in zip(task_pred, labels) if p == "S" and g == "PRODUCTIVE")
        fn = sum(1 for p, g in zip(task_pred, labels) if p != "S" and g == "STUCK")
        n_stk = sum(1 for g in labels if g == "STUCK")
        per_task.append({
            "task": task_name, "n": len(steps), "stk": n_stk,
            "tp": tp, "fp": fp, "fn": fn,
        })
        for p, g in zip(task_pred, labels):
            if g == "UNSURE":
                continue
            all_pred.append(1 if p == "S" else 0)
            all_gold.append(1 if g == "STUCK" else 0)

    elapsed = time.time() - t_start
    print(f"{'task':<22}{'n':>5}{'stk':>5}{'tp':>5}{'fp':>5}{'fn':>5}")
    for r in per_task:
        print(f"{r['task']:<22}{r['n']:>5}{r['stk']:>5}"
              f"{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}")

    arr_p = np.array(all_pred); arr_g = np.array(all_gold)
    if len(arr_g) > 0 and 0 < arr_g.sum() < len(arr_g):
        auc = roc_auc_score(arr_g, arr_p)
    else:
        auc = float("nan")
    tp = int(((arr_p == 1) & (arr_g == 1)).sum())
    fp = int(((arr_p == 1) & (arr_g == 0)).sum())
    fn = int(((arr_p == 0) & (arr_g == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    ms_per_call = 1000 * elapsed / max(n_calls, 1)
    print(f"\nPOOLED  AUC={auc:.4f}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
          f"TP={tp} FP={fp} FN={fn}  ({ms_per_call:.0f}ms/call)")

    return {
        "model": model, "few_shot": few_shot_n, "auc": float(auc),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "tp": tp, "fp": fp, "fn": fn,
        "ms_per_call": ms_per_call,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="ollama model tags, e.g. qwen3:0.6b qwen3:1.7b")
    ap.add_argument("--few-shot", type=int, default=0,
                    help="number of few-shot examples from FEW_SHOT_BANK (0-8)")
    ap.add_argument("--task", default=None,
                    help="run on a single task for fast iteration")
    ap.add_argument("--whole-session", action="store_true",
                    help="label entire transcripts in one call (matches Sonnet framing)")
    ap.add_argument("--system-prompt", choices=["short", "sonnet"], default="short",
                    help="short (default): current short prompt; "
                         "sonnet: the full Sonnet labeling prompt adapted to single-step output")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    global _TASK_DESCRIPTION
    if args.system_prompt == "sonnet":
        _TASK_DESCRIPTION = _TASK_DESCRIPTION_SONNET
    else:
        _TASK_DESCRIPTION = _TASK_DESCRIPTION_SHORT
    print(f"System prompt: {args.system_prompt} ({len(_TASK_DESCRIPTION)} chars)")

    tasks = load_ood_tasks(args.task)
    print(f"Loaded {len(tasks)} OOD tasks, "
          f"{sum(len(s) for _, s, _ in tasks)} total steps")

    results = []
    for model in args.models:
        try:
            if args.whole_session:
                r = evaluate_model_whole_session(model, tasks, args.verbose)
            else:
                r = evaluate_model(model, tasks, args.few_shot, args.verbose)
            results.append(r)
        except Exception as e:
            print(f"SKIP {model}: {e}")

    print("\n" + "=" * 95)
    print(f"HEAD-TO-HEAD (few_shot={args.few_shot})")
    print("=" * 95)
    print(f"{'model':<28}{'AUC':>8}{'P':>8}{'R':>8}{'F1':>8}"
          f"{'TP':>5}{'FP':>5}{'FN':>5}{'ms/call':>10}")
    for r in results:
        print(f"{r['model']:<28}{r['auc']:>8.4f}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['f1']:>8.3f}"
              f"{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}{r['ms_per_call']:>10.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
