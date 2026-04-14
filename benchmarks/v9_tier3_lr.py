#!/usr/bin/env python3
"""
Sanity check: does an LR match the MLP on the 6-feature minimal set?

If yes — the MLP's hidden layers aren't contributing anything and we
should prefer LR for interpretability, smaller footprint, and better
OOD generalization properties under distribution shift.

Uses sklearn's LogisticRegression with L2 + class_weight, trained
on the same session split + normalized features as the MLP.

Usage:
  .venv/bin/python benchmarks/v9_tier3_lr.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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

import argparse

FEATURE_SETS = {
    "minimal6": [
        "is_bash",
        "match_ratio_5",
        "repeat_no_error",
        "self_sim_max",
        "match_ratio_recent_3",
        "cur_bash_and_match_ratio",
    ],
    "core4": [
        "match_ratio_5",
        "self_sim_max",
        "repeat_no_error",
        "cur_bash_and_match_ratio",
    ],
}
FEATURES = FEATURE_SETS["minimal6"]

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


def build_ood_features():
    X, y = [], []
    for td in sorted(RUN_DIR.iterdir()):
        if not td.is_dir():
            continue
        t = td / "transcript_1.jsonl"
        lp = td / "sonnet_labels.json"
        if not (t.exists() and lp.exists()):
            continue
        steps = parse_transcript(t)
        feats = compute_step_features(steps)
        labels = json.loads(lp.read_text())["labels"]
        n = min(len(feats), len(labels))
        feats = feats[:n]; labels = labels[:n]
        compute_tier1_features(feats)
        for r, lbl in zip(feats, labels):
            if lbl == "UNSURE":
                continue
            X.append([float(r[k]) for k in FEATURES])
            y.append(1 if lbl == "STUCK" else 0)
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)


def metrics_at(scores, labels, threshold):
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1, tp, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=list(FEATURE_SETS.keys()),
                    default="minimal6")
    args = ap.parse_args()
    global FEATURES
    FEATURES = FEATURE_SETS[args.features]
    print(f"Feature set: {args.features} ({len(FEATURES)} features)")
    print(f"  {FEATURES}")

    print("\nLoading + annotating training corpus...")
    all_rows = load_and_annotate("training_manifest_v6.json")
    train_rows, test_rows = session_split(all_rows, seed=42)
    X_train, y_train = build_inputs(train_rows, FEATURES)
    X_test, y_test = build_inputs(test_rows, FEATURES)
    print(f"  train={len(X_train)} stuck={int(y_train.sum())}")
    print(f"  test={len(X_test)} stuck={int(y_test.sum())}")

    # Same normalization as the MLP uses
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0).clip(min=1e-6)
    Xtr = (X_train - mean) / std
    Xte = (X_test - mean) / std

    X_ood, y_ood = build_ood_features()
    Xood_n = (X_ood - mean) / std
    print(f"  OOD={len(X_ood)} stuck={int(y_ood.sum())}")

    print("\n== LR sweep over class_weight multipliers ==")
    print(f"{'mult':>6}{'ind_AUC':>10}{'ind_F1':>10}"
          f"{'ood_AUC':>10}{'ood_P':>8}{'ood_R':>8}{'ood_F1':>8}"
          f"{'TP':>5}{'FP':>5}{'FN':>5}")
    for mult in [1.0, 2.0, 3.0, 5.0]:
        num_pos = int(y_train.sum())
        num_neg = len(y_train) - num_pos
        pos_w = (num_neg / max(num_pos, 1)) * mult
        lr = LogisticRegression(
            C=1.0,
            class_weight={0: 1.0, 1: pos_w},
            max_iter=2000,
            solver="lbfgs",
        )
        lr.fit(Xtr, y_train)
        s_te = lr.predict_proba(Xte)[:, 1]
        auc_id = roc_auc_score(y_test, s_te)
        p, r, f1, _, _, _ = metrics_at(s_te, y_test, 0.5)
        s_ood = lr.predict_proba(Xood_n)[:, 1]
        auc_ood = roc_auc_score(y_ood, s_ood)
        op, orr, of1, tp, fp, fn = metrics_at(s_ood, y_ood, 0.5)
        print(f"{mult:>6.1f}{auc_id:>10.4f}{f1:>10.4f}"
              f"{auc_ood:>10.4f}{op:>8.3f}{orr:>8.3f}{of1:>8.3f}"
              f"{tp:>5}{fp:>5}{fn:>5}")

    # Also print weights for best mult (we'll pick mult=3 for alignment)
    lr = LogisticRegression(
        C=1.0,
        class_weight={0: 1.0, 1: (num_neg / max(num_pos, 1)) * 3.0},
        max_iter=2000,
        solver="lbfgs",
    )
    lr.fit(Xtr, y_train)
    print(f"\nLR weights (mult=3) on normalized features:")
    print(f"  intercept: {lr.intercept_[0]:+.4f}")
    for name, w in zip(FEATURES, lr.coef_[0]):
        print(f"  {name:<28}{w:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
