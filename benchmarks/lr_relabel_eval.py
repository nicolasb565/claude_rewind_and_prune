#!/usr/bin/env python3
"""
Train LR on N-consecutive relabeled training data and evaluate on OOD
against BOTH the original Sonnet labels and the N-consecutive relabeled
version of the OOD labels.

The hypothesis: training on relabeled data (where "stuck" means "has been
stuck for N consecutive steps") should teach the classifier a causally
learnable function. At eval time we report:
  - Performance against the ORIGINAL benchmark labels (the "real" task)
  - Performance against the RELABELED benchmark labels (the "achievable" task)

The gap between the two tells us how much the causal ceiling was limiting
the classifier's apparent F1 on the real task.

Usage:
  .venv/bin/python benchmarks/lr_relabel_eval.py --n 5
  .venv/bin/python benchmarks/lr_relabel_eval.py --n 3 --task 03_llvm_loop_vec
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def relabel_session(step_label_pairs: list[tuple[int, float]], n: int) -> dict[int, float]:
    """Given (step, label) pairs for one session (label is 0.0/0.5/1.0),
    apply N-consecutive relabeling. A step stays at 1.0 (STUCK) iff the
    last N consecutive steps (including this one) were all 1.0. Otherwise
    becomes 0.0 (PRODUCTIVE). 0.5 (UNSURE) is left alone.
    """
    # Sort by step
    pairs = sorted(step_label_pairs, key=lambda x: x[0])
    steps = [s for s, _ in pairs]
    labels = [l for _, l in pairs]
    new = {}
    for i in range(len(labels)):
        if labels[i] == 0.5:
            new[steps[i]] = 0.5
            continue
        start = max(0, i - n + 1)
        window = labels[start:i + 1]
        if len(window) >= n and all(l >= 0.9 for l in window):
            new[steps[i]] = 1.0
        else:
            new[steps[i]] = 0.0
    return new


def relabel_rows(rows: list[dict], n: int) -> list[dict]:
    """Return a new list of rows with the label field updated per
    N-consecutive hindsight rule. Session structure is inferred from
    session_id + step fields in each row.
    """
    by_session = defaultdict(list)
    for r in rows:
        by_session[r["session_id"]].append((r["step"], r["label"]))

    new_label_for = {}  # (session_id, step) -> new_label
    for sid, pairs in by_session.items():
        new_labels = relabel_session(pairs, n)
        for step, lbl in new_labels.items():
            new_label_for[(sid, step)] = lbl

    out = []
    for r in rows:
        key = (r["session_id"], r["step"])
        if key in new_label_for:
            r2 = dict(r)
            r2["label"] = new_label_for[key]
            out.append(r2)
        else:
            out.append(r)
    return out


def build_xy(rows, features):
    import numpy as np
    usable = [r for r in rows if r["label"] in (0.0, 1.0)]
    X = np.array([[r[k] for k in features] for r in usable], dtype=np.float64)
    y = np.array([1 if r["label"] >= 0.9 else 0 for r in usable], dtype=np.int32)
    sessions = [r["session_id"] for r in usable]
    return X, y, sessions


def metrics_at_threshold(scores, labels, threshold):
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return p, r, f1, tp, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="N consecutive rule")
    ap.add_argument("--cache", default="data/generated/content_prototype.json")
    ap.add_argument("--task", default="03_llvm_loop_vec",
                    help="OOD task to focus on for per-task breakdown")
    args = ap.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        precision_recall_curve,
        roc_auc_score,
    )

    FEATS = [
        "match_ratio_5", "self_sim_max", "repeat_no_error",
        "cur_bash_and_match_ratio",
        "unique_err_sigs_6", "new_token_ratio_vs_5",
        "has_success_marker", "err_volume_ratio_vs_5",
    ]

    cache_path = REPO / args.cache
    if not cache_path.exists():
        print(f"ERROR: {cache_path} missing")
        return 1
    d = json.loads(cache_path.read_text())
    indist = d["indist"]
    ood = d["ood"]

    print(f"=== N = {args.n} ===")
    print(f"Original: train={len(indist)}  ood={len(ood)}")

    # Relabel train and ood under N-consecutive rule
    indist_re = relabel_rows(indist, args.n)
    ood_re = relabel_rows(ood, args.n)

    # Build four datasets:
    #   X_tr_orig / y_tr_orig : original training labels
    #   X_tr_new  / y_tr_new  : N-relabeled training labels
    #   X_ood_orig / y_ood_orig : original ood labels (eval set A)
    #   X_ood_new  / y_ood_new  : N-relabeled ood labels (eval set B)
    X_tr_orig, y_tr_orig, _ = build_xy(indist, FEATS)
    X_tr_new, y_tr_new, _ = build_xy(indist_re, FEATS)
    X_ood_orig, y_ood_orig, sess_orig = build_xy(ood, FEATS)
    X_ood_new, y_ood_new, sess_new = build_xy(ood_re, FEATS)

    print(f"\ntrain:")
    print(f"  original: {len(y_tr_orig)} rows, {int(y_tr_orig.sum())} stuck ({100*y_tr_orig.mean():.1f}%)")
    print(f"  relabeled: {len(y_tr_new)} rows, {int(y_tr_new.sum())} stuck ({100*y_tr_new.mean():.1f}%)")
    print(f"ood:")
    print(f"  original: {len(y_ood_orig)} rows, {int(y_ood_orig.sum())} stuck ({100*y_ood_orig.mean():.1f}%)")
    print(f"  relabeled: {len(y_ood_new)} rows, {int(y_ood_new.sum())} stuck ({100*y_ood_new.mean():.1f}%)")

    # Normalize features using ORIGINAL training data stats (same as production LR)
    mean = X_tr_orig.mean(axis=0)
    std = X_tr_orig.std(axis=0).clip(min=1e-6)

    def norm(X):
        return (X - mean) / std

    X_tr_orig_n = norm(X_tr_orig)
    X_tr_new_n = norm(X_tr_new)
    X_ood_orig_n = norm(X_ood_orig)
    X_ood_new_n = norm(X_ood_new)

    # ── Train TWO LR models ──────────────────────────────────────────────
    print(f"\n=== Training two LR models ===")
    # Model A: LR trained on original labels (baseline)
    pos_a = int(y_tr_orig.sum())
    neg_a = len(y_tr_orig) - pos_a
    lr_orig = LogisticRegression(
        C=1.0,
        class_weight={0: 1.0, 1: neg_a / max(pos_a, 1)},
        max_iter=2000,
        solver="lbfgs",
    )
    lr_orig.fit(X_tr_orig_n, y_tr_orig)
    print("  lr_orig weights:",
          " ".join(f"{name}={w:+.3f}" for name, w in zip(FEATS, lr_orig.coef_[0])))

    # Model B: LR trained on N-relabeled labels
    pos_b = int(y_tr_new.sum())
    neg_b = len(y_tr_new) - pos_b
    lr_new = LogisticRegression(
        C=1.0,
        class_weight={0: 1.0, 1: neg_b / max(pos_b, 1)},
        max_iter=2000,
        solver="lbfgs",
    )
    lr_new.fit(X_tr_new_n, y_tr_new)
    print(f"  lr_new  weights:",
          " ".join(f"{name}={w:+.3f}" for name, w in zip(FEATS, lr_new.coef_[0])))

    # ── 4 evaluations: 2 models × 2 OOD label sets ──────────────────────
    def eval_model(name, model, X_eval_orig, y_eval_orig, X_eval_new, y_eval_new):
        print(f"\n{'='*70}")
        print(f"=== {name} ===")
        print(f"{'='*70}")
        # Predictions don't depend on label set, only the model & features
        s_orig = model.predict_proba(X_eval_orig)[:, 1]
        s_new = model.predict_proba(X_eval_new)[:, 1]
        # Use the features from the original OOD set (they're the same rows
        # in the same order, only labels differ — but since build_xy filters
        # UNSURE which is label-dependent, we need to be careful)
        for label_name, s, y in [
            ("ORIGINAL OOD labels", s_orig, y_eval_orig),
            (f"RELABELED (N={args.n}) OOD labels", s_new, y_eval_new),
        ]:
            print(f"\n  -- {label_name} --")
            if y.sum() == 0 or y.sum() == len(y):
                print("    (degenerate: no positives or no negatives)")
                continue
            auc = roc_auc_score(y, s)
            p, r, f1, tp, fp, fn = metrics_at_threshold(s, y, 0.5)
            print(f"    AUC={auc:.4f}  @0.5: P={p:.3f} R={r:.3f} F1={f1:.3f}  "
                  f"TP={tp} FP={fp} FN={fn}")

            # Recall @ precision thresholds
            precisions, recalls, thresholds = precision_recall_curve(y, s)
            print(f"    {'target_P':>10} {'actual_P':>10} {'recall':>10} "
                  f"{'thresh':>10} {'TP':>5} {'FP':>5} {'FN':>5}")
            for target in [0.50, 0.70, 0.80, 0.90]:
                valid = precisions >= target
                if not valid.any():
                    print(f"    {target:>10.2f}  (unreachable)")
                    continue
                valid_recalls = np.where(valid, recalls, -1)
                best_idx = int(valid_recalls.argmax())
                best_p = precisions[best_idx]
                best_r = recalls[best_idx]
                best_thresh = 1.0 if best_idx >= len(thresholds) else thresholds[best_idx]
                pr = (s >= best_thresh).astype(int)
                ttp = int(((pr == 1) & (y == 1)).sum())
                tfp = int(((pr == 1) & (y == 0)).sum())
                tfn = int(((pr == 0) & (y == 1)).sum())
                print(f"    {target:>10.2f} {best_p:>10.3f} {best_r:>10.3f} "
                      f"{best_thresh:>10.4f} {ttp:>5} {tfp:>5} {tfn:>5}")

    # Build a 4-way comparison. Use X_ood_orig (labels filtered against
    # original UNSURE) for consistency — but we need y values for both
    # label sets aligned to the same rows. Let's drop UNSURE from BOTH
    # label sets against the ORIGINAL row set so the comparison is apples
    # to apples.
    orig_rows = [r for r in ood if r["label"] in (0.0, 1.0)]
    re_by_key = {(r["session_id"], r["step"]): r["label"]
                 for r in ood_re if r["label"] in (0.0, 1.0)}
    # Keep only rows present in both
    keep_rows = [r for r in orig_rows if (r["session_id"], r["step"]) in re_by_key]
    X_common = np.array([[r[k] for k in FEATS] for r in keep_rows], dtype=np.float64)
    X_common_n = norm(X_common)
    y_common_orig = np.array(
        [1 if r["label"] >= 0.9 else 0 for r in keep_rows], dtype=np.int32
    )
    y_common_new = np.array(
        [1 if re_by_key[(r["session_id"], r["step"])] >= 0.9 else 0 for r in keep_rows],
        dtype=np.int32,
    )

    print(f"\n{'='*70}")
    print(f"Common eval set: {len(keep_rows)} rows")
    print(f"  original stuck: {int(y_common_orig.sum())}")
    print(f"  relabeled stuck: {int(y_common_new.sum())}")
    print(f"{'='*70}")

    for model_name, model in [("LR trained on ORIGINAL labels", lr_orig),
                              (f"LR trained on RELABELED N={args.n} labels", lr_new)]:
        eval_model(model_name, model, X_common_n, y_common_orig, X_common_n, y_common_new)

    # ── Per-task focus on --task (default 03_llvm_loop_vec) ─────────────
    task_key = f"bench_{args.task}"
    task_rows = [r for r in keep_rows if r["session_id"] == task_key]
    task_idx = [i for i, r in enumerate(keep_rows) if r["session_id"] == task_key]
    if task_rows:
        print(f"\n{'='*70}")
        print(f"=== Per-task focus: {args.task} ({len(task_rows)} rows) ===")
        print(f"{'='*70}")
        X_task = X_common_n[task_idx]
        y_task_orig = y_common_orig[task_idx]
        y_task_new = y_common_new[task_idx]
        print(f"  original stuck: {int(y_task_orig.sum())}")
        print(f"  relabeled stuck: {int(y_task_new.sum())}")
        print()

        for model_name, model in [("lr_orig", lr_orig), (f"lr_new(N={args.n})", lr_new)]:
            scores = model.predict_proba(X_task)[:, 1]
            for label_name, y in [
                ("original", y_task_orig),
                (f"relabeled", y_task_new),
            ]:
                if y.sum() == 0:
                    continue
                p, r, f1, tp, fp, fn = metrics_at_threshold(scores, y, 0.5)
                auc = roc_auc_score(y, scores) if 0 < y.sum() < len(y) else float('nan')
                print(f"  {model_name:<20} vs {label_name:<10} "
                      f"AUC={auc:.3f} P={p:.3f} R={r:.3f} F1={f1:.3f} "
                      f"TP={tp} FP={fp} FN={fn}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
