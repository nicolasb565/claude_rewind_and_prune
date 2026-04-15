#!/usr/bin/env python3
"""
Sweep N-consecutive post-filters on top of LR scores.

Motivation: LR alone on OOD tops out around F1=0.326 because single-step FPs
drown out the signal. The product only cares about *sustained* stuck, so a
trailing-window filter should crush FPs while keeping recall on long loops.

Filter forms (applied to per-step P(stuck) scores in session order):
  all-of-N:   flag iff last N scores are all >= thr
  kofn(k):    flag iff at least k of last N scores are >= thr
  mean-of-N:  flag iff mean of last N scores >= thr

Usage:
  .venv/bin/python benchmarks/lr_filter_sweep.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parent.parent

FEATS = [
    "match_ratio_5", "self_sim_max", "repeat_no_error",
    "cur_bash_and_match_ratio",
    "unique_err_sigs_6", "new_token_ratio_vs_5",
    "has_success_marker", "err_volume_ratio_vs_5",
]


def build_xy(rows):
    usable = [r for r in rows if r["label"] in (0.0, 1.0)]
    X = np.array([[r[k] for k in FEATS] for r in usable], dtype=np.float64)
    y = np.array([1 if r["label"] >= 0.9 else 0 for r in usable], dtype=np.int32)
    return X, y, usable


def confusion(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "p": p, "r": r, "f1": f1}


import math


def _aggregate(window, rule, k=None):
    """Collapse a window of LR probabilities to a single aggregate score."""
    n = len(window)
    if rule == "mean":
        return sum(window) / n
    if rule == "min":
        return min(window)
    if rule == "max":
        return max(window)
    if rule == "median":
        s = sorted(window)
        mid = n // 2
        return s[mid] if n % 2 == 1 else 0.5 * (s[mid - 1] + s[mid])
    if rule == "trim":
        if n < 3:
            return sum(window) / n
        s = sorted(window)[1:-1]  # drop top+bottom
        return sum(s) / len(s)
    if rule == "geom":
        eps = 1e-6
        return math.exp(sum(math.log(max(w, eps)) for w in window) / n)
    if rule == "logit":
        # Average in logit space, then back through sigmoid.
        eps = 1e-6
        s = 0.0
        for w in window:
            w = min(max(w, eps), 1 - eps)
            s += math.log(w / (1 - w))
        avg = s / n
        return 1.0 / (1.0 + math.exp(-avg))
    if rule == "ewma":
        # k carries the α parameter * 100 (so α=0.5 → k=50).
        alpha = (k or 50) / 100.0
        ewma = window[0]
        for w in window[1:]:
            ewma = alpha * w + (1 - alpha) * ewma
        return ewma
    raise ValueError(rule)


def apply_filter(scores_by_step, rule, N, thr, k=None):
    """Return a {step: flagged_bool} dict for one session.

    scores_by_step: dict[step_int, float] in arbitrary order.
    Steps are sorted ascending to build a rolling window.
    """
    steps = sorted(scores_by_step.keys())
    out = {}
    for i, s in enumerate(steps):
        start = max(0, i - N + 1)
        window = [scores_by_step[steps[j]] for j in range(start, i + 1)]
        if len(window) < N:
            out[s] = False
            continue
        if rule == "all":
            out[s] = all(w >= thr for w in window)
        elif rule == "kofn":
            out[s] = sum(1 for w in window if w >= thr) >= k
        else:
            out[s] = _aggregate(window, rule, k=k) >= thr
    return out


def main() -> int:
    d = json.loads((REPO / "data/generated/content_prototype.json").read_text())
    indist = d["indist"]
    ood = d["ood"]

    X_tr, y_tr, _ = build_xy(indist)
    X_ood, y_ood, rows_ood = build_xy(ood)

    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0).clip(min=1e-6)
    X_tr_n = (X_tr - mean) / std
    X_ood_n = (X_ood - mean) / std

    pos = int(y_tr.sum())
    neg = len(y_tr) - pos
    lr = LogisticRegression(
        C=1.0,
        class_weight={0: 1.0, 1: neg / max(pos, 1)},
        max_iter=2000,
        solver="lbfgs",
    )
    lr.fit(X_tr_n, y_tr)
    scores = lr.predict_proba(X_ood_n)[:, 1]

    print(f"=== LR baseline ===")
    print(f"  train rows: {len(y_tr)} ({pos} stuck, {100*pos/len(y_tr):.1f}%)")
    print(f"  ood rows: {len(y_ood)} ({int(y_ood.sum())} stuck, "
          f"{100*y_ood.mean():.1f}%)")
    print(f"  weights:", " ".join(
        f"{name}={w:+.3f}" for name, w in zip(FEATS, lr.coef_[0])))
    print(f"  score range: min={scores.min():.4f} max={scores.max():.4f} "
          f"median={np.median(scores):.4f}")

    # Regroup scores by session for the rolling window
    sess_scores = defaultdict(dict)  # session_id -> {step: score}
    sess_labels = defaultdict(dict)  # session_id -> {step: y}
    sess_task = {}
    for r, score, y in zip(rows_ood, scores, y_ood):
        sid = r["session_id"]
        step = r["step"]
        sess_scores[sid][step] = float(score)
        sess_labels[sid][step] = int(y)
        sess_task[sid] = sid.replace("bench_", "")

    idx_by_step = {}
    for i, r in enumerate(rows_ood):
        idx_by_step[(r["session_id"], r["step"])] = i

    def evaluate(rule, N, thr, k=None):
        """Return pooled confusion + per-task dict (per-STEP metrics)."""
        flags = np.zeros(len(y_ood), dtype=np.int32)
        for sid, steps_dict in sess_scores.items():
            flagged = apply_filter(steps_dict, rule, N, thr, k=k)
            for step, fl in flagged.items():
                flags[idx_by_step[(sid, step)]] = 1 if fl else 0
        pooled = confusion(flags, y_ood)
        by_task = {}
        for sid, steps_dict in sess_scores.items():
            idxs = [idx_by_step[(sid, step)] for step in steps_dict]
            by_task[sess_task[sid]] = confusion(flags[idxs], y_ood[idxs])
        return pooled, by_task

    def evaluate_multi(rule, N, thr, k=None):
        """Return per-step, episode-level, and session-level confusions.

        episode-level:
          - TP = stuck episode with >=1 fire somewhere inside its span
          - FN = stuck episode with no fire inside
          - FP = contiguous fire-cluster with 0 stuck steps inside
                 (a wholly wrong alarm burst)
        session-level:
          - each session with any stuck label: TP if fired at least once, else FN
          - each session with no stuck label: FP if fired at least once, else TN
        """
        step_flags = {}  # (sid, step) -> bool
        for sid, steps_dict in sess_scores.items():
            flagged = apply_filter(steps_dict, rule, N, thr, k=k)
            for step, fl in flagged.items():
                step_flags[(sid, step)] = fl

        # Per-step
        flags = np.zeros(len(y_ood), dtype=np.int32)
        for (sid, step), fl in step_flags.items():
            flags[idx_by_step[(sid, step)]] = 1 if fl else 0
        step_c = confusion(flags, y_ood)

        # Episode & session level
        ep_tp = ep_fn = ep_fp = 0
        sess_tp = sess_fn = sess_fp = sess_tn = 0
        for sid, steps_dict in sess_scores.items():
            steps = sorted(steps_dict.keys())
            labels = [sess_labels[sid][s] for s in steps]
            fires = [1 if step_flags[(sid, s)] else 0 for s in steps]

            # Collect stuck episodes (contiguous runs of label==1)
            stuck_eps = []
            i = 0
            while i < len(labels):
                if labels[i] == 1:
                    j = i
                    while j < len(labels) and labels[j] == 1:
                        j += 1
                    stuck_eps.append((i, j - 1))
                    i = j
                else:
                    i += 1
            # Collect contiguous fire-runs
            fire_runs = []
            i = 0
            while i < len(fires):
                if fires[i] == 1:
                    j = i
                    while j < len(fires) and fires[j] == 1:
                        j += 1
                    fire_runs.append((i, j - 1))
                    i = j
                else:
                    i += 1

            # Episode TP/FN: does each stuck episode contain at least one fire?
            for a, b in stuck_eps:
                if any(fires[k] for k in range(a, b + 1)):
                    ep_tp += 1
                else:
                    ep_fn += 1
            # Episode FP: fire-clusters that contain zero stuck steps
            for a, b in fire_runs:
                if not any(labels[k] == 1 for k in range(a, b + 1)):
                    ep_fp += 1

            # Session level
            session_is_stuck = any(l == 1 for l in labels)
            session_fired = any(fires)
            if session_is_stuck:
                if session_fired:
                    sess_tp += 1
                else:
                    sess_fn += 1
            else:
                if session_fired:
                    sess_fp += 1
                else:
                    sess_tn += 1

        ep_p = ep_tp / max(ep_tp + ep_fp, 1)
        ep_r = ep_tp / max(ep_tp + ep_fn, 1)
        ep_f1 = 2 * ep_p * ep_r / max(ep_p + ep_r, 1e-9)
        ep = {"tp": ep_tp, "fp": ep_fp, "fn": ep_fn, "p": ep_p, "r": ep_r, "f1": ep_f1}

        sess_p = sess_tp / max(sess_tp + sess_fp, 1)
        sess_r = sess_tp / max(sess_tp + sess_fn, 1)
        sess_f1 = 2 * sess_p * sess_r / max(sess_p + sess_r, 1e-9)
        sess_m = {"tp": sess_tp, "fp": sess_fp, "fn": sess_fn, "tn": sess_tn,
                  "p": sess_p, "r": sess_r, "f1": sess_f1}

        return step_c, ep, sess_m

    # ── Sweep 1: N ∈ {1..7}, rule=all-of-N, thresholds chosen to span FP/FN ─
    print(f"\n{'='*78}")
    print("=== Sweep 1: all-of-N filter, threshold grid ===")
    print(f"{'='*78}")
    thr_grid = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    header = f"{'N':>2} {'thr':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6}"
    print(header)
    best = {}  # (N) -> best F1 result
    for N in [1, 2, 3, 4, 5, 7]:
        for thr in thr_grid:
            pooled, _ = evaluate("all", N, thr)
            line = (f"{N:>2} {thr:>5.2f} {pooled['tp']:>4} {pooled['fp']:>4} "
                    f"{pooled['fn']:>4} {pooled['p']:>6.3f} {pooled['r']:>6.3f} "
                    f"{pooled['f1']:>6.3f}")
            print(line)
            key = N
            if key not in best or pooled['f1'] > best[key][0]['f1']:
                best[key] = (pooled, thr)
        print()

    # ── Sweep 2: K-of-N ─────────────────────────────────────────────────────
    print(f"{'='*78}")
    print("=== Sweep 2: K-of-N filter (majority / relaxed) ===")
    print(f"{'='*78}")
    print(f"{'N':>2} {'K':>2} {'thr':>5} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'P':>6} {'R':>6} {'F1':>6}")
    for N, k in [(3, 2), (5, 3), (5, 4), (7, 4), (7, 5), (7, 6),
                 (8, 7), (9, 8), (10, 9), (10, 8)]:
        for thr in [0.40, 0.50, 0.60, 0.70]:
            pooled, _ = evaluate("kofn", N, thr, k=k)
            print(f"{N:>2} {k:>2} {thr:>5.2f} {pooled['tp']:>4} "
                  f"{pooled['fp']:>4} {pooled['fn']:>4} {pooled['p']:>6.3f} "
                  f"{pooled['r']:>6.3f} {pooled['f1']:>6.3f}")
        print()

    # ── Sweep 3: mean-of-N filter ───────────────────────────────────────────
    # For each N, fine threshold sweep and extract three Pareto points:
    #   F1-best, P-best (max P with R ≥ R_FLOOR), R-best (max R with P ≥ P_FLOOR).
    print(f"{'='*78}")
    print("=== Sweep 3: mean-of-N filter — Pareto per N ===")
    print(f"{'='*78}")
    print("Decision: flag iff mean of last N LR scores >= thr")
    R_FLOOR = 0.30
    P_FLOOR = 0.50
    print(f"(best-P subject to R ≥ {R_FLOOR}; best-R subject to P ≥ {P_FLOOR})")
    fine_thrs = [round(0.10 + 0.005 * i, 4) for i in range(141)]  # 0.10..0.80
    n_values = [2, 3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 40, 50]

    rows_by_N = {}
    for N in n_values:
        rows = []
        for thr in fine_thrs:
            pooled, _ = evaluate("mean", N, thr)
            rows.append((thr, pooled))
        rows_by_N[N] = rows

    def pick_f1_best(rows):
        return max(rows, key=lambda x: x[1]["f1"])

    def pick_p_best(rows, r_floor):
        cand = [(t, p) for t, p in rows if p["r"] >= r_floor]
        if not cand:
            return None
        return max(cand, key=lambda x: (x[1]["p"], x[1]["r"]))

    def pick_r_best(rows, p_floor):
        cand = [(t, p) for t, p in rows if p["p"] >= p_floor]
        if not cand:
            return None
        return max(cand, key=lambda x: (x[1]["r"], x[1]["p"]))

    def fmt(entry):
        if entry is None:
            return "   —    —    —    —    —    —    —"
        thr, p = entry
        return (f"{thr:>6.3f} {p['tp']:>4} {p['fp']:>4} {p['fn']:>4} "
                f"{p['p']:>6.3f} {p['r']:>6.3f} {p['f1']:>6.3f}")

    print(f"\n=== F1-best per N ===")
    header = (f"{'N':>3} {'thr':>6} {'TP':>4} {'FP':>4} {'FN':>4} "
              f"{'P':>6} {'R':>6} {'F1':>6}")
    print(header)
    for N in n_values:
        print(f"{N:>3} {fmt(pick_f1_best(rows_by_N[N]))}")

    print(f"\n=== P-best per N (subject to R ≥ {R_FLOOR}) ===")
    print(header)
    for N in n_values:
        print(f"{N:>3} {fmt(pick_p_best(rows_by_N[N], R_FLOOR))}")

    print(f"\n=== R-best per N (subject to P ≥ {P_FLOOR}) ===")
    print(header)
    for N in n_values:
        print(f"{N:>3} {fmt(pick_r_best(rows_by_N[N], P_FLOOR))}")

    # ── Per-task breakdown at a few candidate operating points ──────────────
    candidates = [
        ("baseline N=1",        "all",  1,  0.50, None),
        ("mean-of-10 @ 0.55",   "mean", 10, 0.55, None),
        ("mean-of-12 @ 0.47",   "mean", 12, 0.465, None),
        ("mean-of-15 @ 0.425",  "mean", 15, 0.425, None),
        ("mean-of-15 @ 0.455",  "mean", 15, 0.455, None),
        ("mean-of-18 @ 0.41",   "mean", 18, 0.410, None),
        ("mean-of-20 @ 0.39",   "mean", 20, 0.390, None),
        ("mean-of-25 @ 0.44",   "mean", 25, 0.440, None),
        ("4of7 @ 0.60",         "kofn", 7,  0.60, 4),
        ("5of7 @ 0.60",         "kofn", 7,  0.60, 5),
    ]
    # ── Sweep 4: baseline vs mean-of-2 across fine thresholds ───────────────
    #   Question: does a 2-sample smoother cut spurious fires while holding
    #   session-level recall? Compare against the unfiltered LR directly.
    print(f"{'='*78}")
    print("=== Sweep 4: N=1 vs N=2 threshold sweep, all three metrics ===")
    print(f"{'='*78}")
    fine2 = [round(0.20 + 0.01 * i, 3) for i in range(71)]  # 0.20..0.90
    for N in [1, 2]:
        print(f"\n-- mean-of-{N} --")
        print(f"{'thr':>5} | {'step TP/FP/FN F1':^23} | "
              f"{'epis TP/FP/FN F1':^23} | {'sess TP/FP/FN F1':^23}")
        rows_for_N = []
        for thr in fine2:
            step, ep, sess = evaluate_multi("mean", N, thr)
            rows_for_N.append((thr, step, ep, sess))
            s1 = f"{step['tp']:>3}/{step['fp']:>3}/{step['fn']:>3} {step['f1']:.3f}"
            s2 = f"{ep['tp']:>3}/{ep['fp']:>3}/{ep['fn']:>3} {ep['f1']:.3f}"
            s3 = f"{sess['tp']:>3}/{sess['fp']:>3}/{sess['fn']:>3} {sess['f1']:.3f}"
            print(f"{thr:>5.2f} | {s1:^23} | {s2:^23} | {s3:^23}")

        # Best-per-metric for this N
        best_step = max(rows_for_N, key=lambda x: x[1]["f1"])
        best_ep = max(rows_for_N, key=lambda x: x[2]["f1"])
        best_sess = max(rows_for_N, key=lambda x: x[3]["f1"])

        # Precision-oriented: best session-level with session FP == 0
        zero_fp_sess = [r for r in rows_for_N if r[3]["fp"] == 0]
        if zero_fp_sess:
            best_zero_fp = max(zero_fp_sess, key=lambda x: x[3]["r"])
        else:
            best_zero_fp = None

        print(f"\n  best per-step F1: thr={best_step[0]:.2f}  "
              f"step={best_step[1]['tp']}/{best_step[1]['fp']}/{best_step[1]['fn']} "
              f"F1={best_step[1]['f1']:.3f}")
        print(f"  best episode F1:  thr={best_ep[0]:.2f}  "
              f"ep={best_ep[2]['tp']}/{best_ep[2]['fp']}/{best_ep[2]['fn']} "
              f"F1={best_ep[2]['f1']:.3f}")
        print(f"  best session F1:  thr={best_sess[0]:.2f}  "
              f"sess={best_sess[3]['tp']}/{best_sess[3]['fp']}/{best_sess[3]['fn']} "
              f"F1={best_sess[3]['f1']:.3f}")
        if best_zero_fp is not None:
            print(f"  best R w/ sess FP=0: thr={best_zero_fp[0]:.2f}  "
                  f"sess={best_zero_fp[3]['tp']}/0/{best_zero_fp[3]['fn']} "
                  f"R={best_zero_fp[3]['r']:.3f} F1={best_zero_fp[3]['f1']:.3f}")

    print(f"\n{'='*78}")
    print("=== Multi-metric comparison at candidate operating points ===")
    print(f"{'='*78}")
    print("  per-step = strict (each (session,step) pair)")
    print("  episode  = stuck-run granularity; FP = wholly-wrong fire clusters")
    print("  session  = one count per session (10 total: 7 have stuck, 3 don't)")
    print()
    print(f"{'config':<22} | {'per-step':^25} | {'episode':^25} | {'session':^23}")
    print(f"{'':<22} | {'TP  FP  FN     F1':^25} | "
          f"{'TP  FP  FN     F1':^25} | {'TP  FP  FN     F1':^23}")
    print("-" * 100)
    for name, rule, N, thr, k in candidates:
        step, ep, sess = evaluate_multi(rule, N, thr, k=k)
        step_s = (f"{step['tp']:>3} {step['fp']:>3} {step['fn']:>3}   "
                  f"{step['f1']:>5.3f}")
        ep_s = (f"{ep['tp']:>3} {ep['fp']:>3} {ep['fn']:>3}   "
                f"{ep['f1']:>5.3f}")
        sess_s = (f"{sess['tp']:>3} {sess['fp']:>3} {sess['fn']:>3}   "
                  f"{sess['f1']:>5.3f}")
        print(f"{name:<22} | {step_s:^25} | {ep_s:^25} | {sess_s:^23}")

    # ── Sweep 5: nonlinear aggregators ──────────────────────────────────────
    #   Linear mean hits a ceiling (step P=0.24 at best). Try aggregators
    #   that weight the signal differently: min, median, trimmed mean,
    #   geometric, logit-space, EWMA.
    print(f"\n{'='*90}")
    print("=== Sweep 5: nonlinear aggregators — best operating point per rule ===")
    print(f"{'='*90}")
    nl_configs = [
        # (rule, N, k/alpha_hint)
        ("min",    2, None),
        ("min",    3, None),
        ("min",    4, None),
        ("min",    5, None),
        ("median", 3, None),
        ("median", 5, None),
        ("median", 7, None),
        ("trim",   5, None),
        ("trim",   7, None),
        ("geom",   3, None),
        ("geom",   5, None),
        ("logit",  3, None),
        ("logit",  5, None),
        ("ewma",   5, 30),   # α=0.30
        ("ewma",   5, 50),   # α=0.50
        ("ewma",   10, 30),
        # Reference:
        ("mean",   2, None),
        ("mean",   5, None),
    ]
    nl_thrs = [round(0.05 + 0.005 * i, 4) for i in range(190)]  # 0.05..0.99

    header = (f"{'rule':<8} {'N':>2} {'thr':>6} | "
              f"{'step TP/FP/FN     F1':^26} | "
              f"{'ep TP/FP/FN      F1':^26} | "
              f"{'sess TP/FP/FN    F1':^24}")
    print(header)
    print("-" * len(header))

    def fmt_trip(m):
        return f"{m['tp']:>3}/{m['fp']:>3}/{m['fn']:>3}  {m['f1']:>5.3f}"

    best_rules = {}  # (rule, N, k) -> (best_thr, best_step, best_ep, best_sess)
    for rule, N, k in nl_configs:
        rows_all = []
        for thr in nl_thrs:
            step, ep, sess = evaluate_multi(rule, N, thr, k=k)
            rows_all.append((thr, step, ep, sess))
        # Pick best-F1 per-step, best-F1 episode, and best-precision-at-reasonable-recall
        best_step_row = max(rows_all, key=lambda x: x[1]["f1"])
        best_ep_row = max(rows_all, key=lambda x: x[2]["f1"])
        # High-precision point: step P >= 0.50 with R max
        hp = [r for r in rows_all if r[1]["p"] >= 0.50 and r[3]["tp"] >= 5]
        best_hp = max(hp, key=lambda r: r[1]["r"]) if hp else None

        tag = f"{rule:<8} {N:>2}"
        alpha_str = f"α={k/100:.2f}" if rule == "ewma" and k else ""
        if alpha_str:
            tag = f"{rule:<5}{alpha_str:>3} {N:>2}"

        thr, s, e, se = best_step_row
        print(f"{tag} {thr:>6.3f} | {fmt_trip(s):^26} | {fmt_trip(e):^26} | "
              f"{fmt_trip(se):^24}   ← best step-F1")
        if best_hp:
            thr, s, e, se = best_hp
            print(f"{'':8} {'':2} {thr:>6.3f} | {fmt_trip(s):^26} | "
                  f"{fmt_trip(e):^26} | {fmt_trip(se):^24}   ← P≥.50 max-R")
        best_rules[(rule, N, k)] = rows_all
    print()

    # ── Sweep 6: median-of-N — best F1 and best P per N ────────────────────
    print(f"\n{'='*100}")
    print("=== Sweep 6: median-of-N — best F1 and best P per N ===")
    print(f"(best-P constrained to step recall >= 0.20 so P→1.0 degeneracies are excluded)")
    print(f"{'='*100}")
    print(f"{'N':>2} {'pick':<6} {'thr':>6}  "
          f"{'step TP/FP/FN':>14}  {'step P':>7} {'step R':>7} {'step F1':>8}  "
          f"{'ep TP/FP/FN':>13}  {'ep F1':>7}  "
          f"{'sess TP/FP/FN':>14}  {'sess F1':>8}")
    print("-" * 120)

    fine_med = [round(0.10 + 0.005 * i, 4) for i in range(181)]  # 0.10..1.00
    R_FLOOR_P = 0.20

    def fmt_row(N, pick, thr, step, ep, sess):
        a = f"{step['tp']:>3}/{step['fp']:>3}/{step['fn']:>3}"
        b = f"{ep['tp']:>2}/{ep['fp']:>3}/{ep['fn']:>3}"
        c = f"{sess['tp']:>2}/{sess['fp']:>2}/{sess['fn']:>2}"
        return (f"{N:>2} {pick:<6} {thr:>6.3f}  "
                f"{a:>14}  {step['p']:>7.3f} {step['r']:>7.3f} "
                f"{step['f1']:>8.3f}  "
                f"{b:>13}  {ep['f1']:>7.3f}  "
                f"{c:>14}  {sess['f1']:>8.3f}")

    for N in range(2, 11):
        rows_n = []
        for thr in fine_med:
            step, ep, sess = evaluate_multi("median", N, thr)
            rows_n.append((thr, step, ep, sess))
        best_f1 = max(rows_n, key=lambda x: x[1]["f1"])
        cand_p = [r for r in rows_n if r[1]["r"] >= R_FLOOR_P]
        best_p = max(cand_p, key=lambda x: (x[1]["p"], x[1]["r"])) if cand_p else None

        thr, step, ep, sess = best_f1
        print(fmt_row(N, "F1", thr, step, ep, sess))
        if best_p is not None:
            thr, step, ep, sess = best_p
            print(fmt_row(N, "P", thr, step, ep, sess))
        else:
            print(f"{N:>2} {'P':<6} (no config hits R ≥ {R_FLOOR_P})")
    print()

    # ── Focused view: mean-of-2, thr 0.30 → 0.60 with ASCII bars ────────────
    print(f"\n{'='*90}")
    print("=== Focused chart: mean-of-2, thr 0.30 → 0.60 ===")
    print(f"{'='*90}")
    print("  step TP/FP/FN  |  ep TP/FP/FN  |  sess TP/FP/FN  | step F1 / ep F1 / sess F1")
    print("-" * 90)
    BAR_W = 20

    def bar(v, scale=1.0):
        n = int(round(BAR_W * min(v / scale, 1.0)))
        return "█" * n + "·" * (BAR_W - n)

    focus_rows = []
    for i in range(31):
        thr = round(0.30 + 0.01 * i, 3)
        step, ep, sess = evaluate_multi("mean", 2, thr)
        focus_rows.append((thr, step, ep, sess))

    print(f"{'thr':>5}  {'step':>11}   {'ep':>10}   {'sess':>10}  "
          f"{'step-F1':^22} {'ep-F1':^22} {'sess-F1':^22}")
    for thr, step, ep, sess in focus_rows:
        a = f"{step['tp']:>3}/{step['fp']:>3}/{step['fn']:>3}"
        b = f"{ep['tp']:>2}/{ep['fp']:>2}/{ep['fn']:>2}"
        c = f"{sess['tp']:>2}/{sess['fp']:>2}/{sess['fn']:>2}"
        # Bars scaled: step F1 in [0, 0.5], ep/sess F1 in [0, 1.0]
        b1 = bar(step['f1'], 0.5)
        b2 = bar(ep['f1'], 0.5)
        b3 = bar(sess['f1'], 1.0)
        print(f"{thr:>5.2f}  {a:>11}   {b:>10}   {c:>10}  "
              f"{b1}{step['f1']:>5.3f} {b2}{ep['f1']:>5.3f} "
              f"{b3}{sess['f1']:>5.3f}")
    print(f"  (step/ep F1 bars scaled to 0.5 max; session F1 bar to 1.0 max)")

    print("\n=== Detail: episode and session counts ===")
    for name, rule, N, thr, k in candidates:
        step, ep, sess = evaluate_multi(rule, N, thr, k=k)
        print(f"\n{name}")
        print(f"  per-step: TP={step['tp']} FP={step['fp']} FN={step['fn']}  "
              f"P={step['p']:.3f} R={step['r']:.3f} F1={step['f1']:.3f}")
        print(f"  episode:  TP={ep['tp']} FP={ep['fp']} FN={ep['fn']}  "
              f"P={ep['p']:.3f} R={ep['r']:.3f} F1={ep['f1']:.3f}")
        print(f"  session:  TP={sess['tp']} FP={sess['fp']} FN={sess['fn']} "
              f"TN={sess['tn']}  "
              f"P={sess['p']:.3f} R={sess['r']:.3f} F1={sess['f1']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
