#!/usr/bin/env python3
"""
Error analysis for LR core4 on 03_llvm_loop_vec.

Scores every step, prints rows where the LR disagrees with Sonnet
(FN = missed stuck, FP = false alarm). For each row, print:
  - step index
  - Sonnet label
  - LR score
  - action tuple + cmd + target
  - feature values

Goal: identify whether there's a transferable pattern the 4 features
can't express, or whether Sonnet and the classifier are seeing the
same stuck loops at different thresholds.

Usage:
  .venv/bin/python benchmarks/v9_tier3_error_analysis.py
  .venv/bin/python benchmarks/v9_tier3_error_analysis.py --task 33_geometry
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.pipeline.parsers.nlile import parse_session  # noqa: E402
from src.pipeline.extract_features import compute_step_features  # noqa: E402
from benchmarks.v9_tier1_train import (  # noqa: E402
    compute_tier1_features,
    load_and_annotate,
    session_split,
    build_inputs,
)

FEATURES = [
    "match_ratio_5",
    "self_sim_max",
    "repeat_no_error",
    "cur_bash_and_match_ratio",
]
RUN_DIR = REPO / "benchmarks" / "results" / "comparison_off"


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
            msg = ev.get("message", {})
            if isinstance(msg, dict):
                messages.append(msg)
    return parse_session(messages)


def fit_lr():
    print("Fitting LR core4 (mult=1)...")
    all_rows = load_and_annotate("training_manifest_v6.json")
    train_rows, _ = session_split(all_rows, seed=42)
    X, y = build_inputs(train_rows, FEATURES)
    mean = X.mean(axis=0); std = X.std(axis=0).clip(min=1e-6)
    Xn = (X - mean) / std
    num_pos = int(y.sum()); num_neg = len(y) - num_pos
    pos_w = num_neg / max(num_pos, 1)
    lr = LogisticRegression(
        C=1.0, class_weight={0: 1.0, 1: pos_w},
        max_iter=2000, solver="lbfgs",
    )
    lr.fit(Xn, y)
    return lr, mean, std


def analyze_task(task_name: str, lr, mean, std, threshold: float):
    td = RUN_DIR / task_name
    t = td / "transcript_1.jsonl"
    lp = td / "sonnet_labels.json"
    steps = parse_transcript(t)
    feats = compute_step_features(steps)
    labels = json.loads(lp.read_text())["labels"]
    n = min(len(feats), len(labels))
    feats = feats[:n]; labels = labels[:n]
    compute_tier1_features(feats)

    X = np.array([[float(r[k]) for k in FEATURES] for r in feats], dtype=np.float64)
    Xn = (X - mean) / std
    scores = lr.predict_proba(Xn)[:, 1]
    pred = (scores >= threshold).astype(int)
    gold = np.array([1 if lbl == "STUCK" else 0 for lbl in labels])

    tp_rows = [(i, scores[i]) for i in range(n) if pred[i] == 1 and gold[i] == 1]
    fn_rows = [(i, scores[i]) for i in range(n) if pred[i] == 0 and gold[i] == 1]
    fp_rows = [(i, scores[i]) for i in range(n) if pred[i] == 1 and gold[i] == 0]

    print(f"\n========== {task_name} (n={n}, threshold={threshold}) ==========")
    print(f"TP={len(tp_rows)}  FN={len(fn_rows)}  FP={len(fp_rows)}")

    def print_row(i: int, score: float):
        step = steps[i] if i < len(steps) else {}
        raw_step = step.get("raw", {}) if isinstance(step, dict) else {}
        action = step.get("action", "?") if isinstance(step, dict) else "?"
        target = (step.get("target_file") or step.get("target_scope") or "")[:50]
        cmd = (step.get("base_cmd") or "")[:40]
        fvals = {k: feats[i][k] for k in FEATURES}
        print(f"  step {i:3d}  score={score:.3f}  gold={labels[i]}  "
              f"action={action}  cmd={cmd}  target={target}")
        print(f"         mr5={fvals['match_ratio_5']:.2f} "
              f"ssmax={fvals['self_sim_max']:.2f} "
              f"rne={fvals['repeat_no_error']:.2f} "
              f"bXr={fvals['cur_bash_and_match_ratio']:.2f}")

    if fn_rows:
        print(f"\n-- MISSED STUCK ({len(fn_rows)}) --")
        for i, s in fn_rows:
            print_row(i, s)

    if fp_rows:
        print(f"\n-- FALSE ALARMS ({len(fp_rows)}) --")
        for i, s in sorted(fp_rows, key=lambda x: -x[1]):
            print_row(i, s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="03_llvm_loop_vec")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    lr, mean, std = fit_lr()
    print(f"LR weights: intercept={lr.intercept_[0]:+.3f}")
    for name, w in zip(FEATURES, lr.coef_[0]):
        print(f"  {name:<28}{w:+.3f}")
    analyze_task(args.task, lr, mean, std, args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
