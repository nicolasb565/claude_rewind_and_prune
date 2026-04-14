#!/usr/bin/env python3
"""
v9 Tier 1 feature batch — extend v9_trimmed_bash (11 dims) with:

  is_search            binary — current step is a search tool (Grep/Glob)
  action_diversity_5   unique action tuples in last 5 steps / 5, range [0.2, 1.0]
  max_consecutive_match  longest run of act_match=1 starting from head, range [0, 5]
  repeat_no_error      binary — p1_act_match=1 AND cur_is_err=0

Feature set: 15 dims (10 v9_trimmed + 1 is_bash + 4 new Tier 1)
Architecture: Linear(15, 20) → ReLU → Linear(20, 10) → ReLU → Linear(10, 1)

All features are computed on the fly from existing per-step jsonl rows
using cmd_hash as an action identifier (cmd_hash is CRC32 of the v5
semantic key, so two steps with the same base_cmd:target_file get the
same value — close enough to action identity for diversity counting).

Before training, run --validate to compute LR correlation on both
distributions for each new feature. Drop anything that flips.

Usage:
  .venv/bin/python benchmarks/v9_tier1_train.py --validate
  .venv/bin/python benchmarks/v9_tier1_train.py --train
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.pipeline.extract_features import V9_FEATURE_NAMES, TOOL_TO_IDX  # noqa: E402

BASE_V9_TRIMMED = (
    [f"v9_p{i+1}_act_match" for i in range(5)]
    + [f"v9_p{i+1}_self_sim" for i in range(5)]
)

BASH_IDX = TOOL_TO_IDX["bash"]
SEARCH_IDX = TOOL_TO_IDX["search"]  # Grep / Glob
VIEW_IDX = TOOL_TO_IDX["view"]      # Read — sometimes treated as search too
EDIT_IDX = TOOL_TO_IDX["edit"]

TIER1_NAMES = [
    "is_bash",
    "is_search",
    "action_diversity_5",
    "max_consecutive_match",
    "match_ratio_5",       # alternative aggregation — replaces max_consecutive_match
    "max_run_anywhere",    # alternative — longest match run anywhere in history
    "repeat_no_error",
    "error_ratio_6",       # marginal — tested for completeness
    "no_error_streak_5",   # marginal — tested for completeness
    # PID-style derived features over the 5-prior window:
    # P = current/most-recent slot (already in BASE_V9_TRIMMED)
    # I = mean of 5 slots (match_ratio_5 for act_match; self_sim_mean_5 new)
    # D = p1 - p5 (trend: +ve means "getting more similar/matching recently")
    "act_match_deriv",     # v9_p1_act_match - v9_p5_act_match
    "self_sim_mean_5",     # mean of v9_p{1..5}_self_sim (I term)
    "self_sim_deriv",      # v9_p1_self_sim - v9_p5_self_sim (D term)
    "self_sim_std_5",      # stddev of v9_p{1..5}_self_sim (variability)
    "is_err_deriv",        # v9_p1_is_err - v9_p5_is_err (error trend)
    # Tier 2 features:
    "min_match_gap",            # smallest gap between matches in 5 slots (1.0 if none)
    "self_sim_max",             # max of v9_p{1..5}_self_sim (any very-similar slot)
    "match_ratio_recent_3",     # match fraction over p1..p3 only (recency-weighted)
    "cur_bash_and_consec_high", # is_bash AND max_consecutive_match >= 0.6 (interaction)
    "action_bigram_repeat",     # 1 if (p1==p3 AND p2==p4) → AB-AB oscillation
    # Tier 3 — more interaction features. Built on the Tier 2 finding that
    # cur_bash_and_consec_high transferred best (interactions beat marginals).
    "cur_edit_and_consec_high",     # edit in a deep run (stuck edit loop)
    "cur_search_and_consec_high",   # grep/glob in a deep run (searching blindly)
    "cur_err_and_consec_high",      # error + repetition (failing same cmd)
    "cur_bash_and_match_ratio",     # continuous version: is_bash * match_ratio_5
    "cur_err_and_match_ratio",      # continuous error × repetition
    "cur_err_and_bash",             # building repeatedly → bash that errors
    # Tier 4 — error-analysis driven:
    "ssmax_and_no_match",           # high self_sim_max AND low match_ratio_5
    "match_and_no_ssmax",            # high match_ratio_5 AND low self_sim_max
]
FULL_FEATURE_NAMES = BASE_V9_TRIMMED + TIER1_NAMES
INPUT_DIM_FULL = len(FULL_FEATURE_NAMES)

DEFAULT_SEED = 42


def compute_tier1_features(session_rows: list[dict]) -> list[dict]:
    """
    Given session rows sorted by step, annotate each row with all Tier 1
    features plus the alternative aggregations. Mutates in place.
    """
    action_history: list[int] = []  # ring of last 5 action tuples (cmd_hash ints)
    err_history: list[float] = []   # ring of last 5 is_err values

    for r in session_rows:
        tool_idx = int(r.get("tool_idx", 6))
        r["is_bash"] = 1.0 if tool_idx == BASH_IDX else 0.0
        r["is_search"] = 1.0 if tool_idx == SEARCH_IDX else 0.0

        if action_history:
            r["action_diversity_5"] = len(set(action_history)) / 5.0
        else:
            r["action_diversity_5"] = 0.0

        # Collect per-slot match flags once for all aggregations
        matches = [r.get(f"v9_p{slot+1}_act_match", 0.0) >= 0.5 for slot in range(5)]

        # max_consecutive_match: longest run from p1 going back (stops at first miss)
        run = 0
        for m in matches:
            if m: run += 1
            else: break
        r["max_consecutive_match"] = float(run) / 5.0

        # match_ratio_5: fraction of 5 slots that match
        r["match_ratio_5"] = sum(matches) / 5.0

        # max_run_anywhere: longest run of matches anywhere in the 5 slots
        best = cur = 0
        for m in matches:
            if m:
                cur += 1
                if cur > best: best = cur
            else:
                cur = 0
        r["max_run_anywhere"] = float(best) / 5.0

        # repeat_no_error: p1 matches AND current step has no error
        p1_match = matches[0]
        cur_err = float(r.get("is_error", 0.0))
        no_err = cur_err < 0.5
        r["repeat_no_error"] = 1.0 if (p1_match and no_err) else 0.0

        # error_ratio_6: fraction of errors in current + last 5 prior
        window = err_history + [cur_err]
        r["error_ratio_6"] = sum(window) / 6.0

        # no_error_streak_5: consecutive no-error steps ending at current (normalized)
        streak = 0 if cur_err > 0.5 else 1
        for e in reversed(err_history):
            if e < 0.5: streak += 1
            else: break
        r["no_error_streak_5"] = min(streak, 5) / 5.0

        # ── PID-style features over the 5-prior window ───────────────────
        # act_match D: slope from oldest to newest slot
        p1_am = float(r.get("v9_p1_act_match", 0.0))
        p5_am = float(r.get("v9_p5_act_match", 0.0))
        r["act_match_deriv"] = p1_am - p5_am

        # self_sim P is p1; I is mean of 5; D is p1-p5; std captures spikiness
        ss = [float(r.get(f"v9_p{i+1}_self_sim", 0.0)) for i in range(5)]
        r["self_sim_mean_5"] = sum(ss) / 5.0
        r["self_sim_deriv"] = ss[0] - ss[4]
        mean_ss = r["self_sim_mean_5"]
        var_ss = sum((v - mean_ss) ** 2 for v in ss) / 5.0
        r["self_sim_std_5"] = math.sqrt(var_ss)

        # is_err derivative (trend in error signal over window)
        p1_err = float(r.get("v9_p1_is_err", 0.0))
        p5_err = float(r.get("v9_p5_is_err", 0.0))
        r["is_err_deriv"] = p1_err - p5_err

        # ── Tier 2 features ──────────────────────────────────────────────
        # min_match_gap: smallest spacing between matches in p1..p5.
        # Encodes "how regular is the repetition". If <2 matches, set to
        # 1.0 (sentinel "no spacing observed" — distinguishes from gap=1).
        match_idx = [i for i, m in enumerate(matches) if m]
        if len(match_idx) >= 2:
            gaps = [match_idx[i+1] - match_idx[i] for i in range(len(match_idx)-1)]
            r["min_match_gap"] = float(min(gaps)) / 5.0
        else:
            r["min_match_gap"] = 1.0

        # self_sim_max: max of 5 slots — at least one very-similar prior?
        r["self_sim_max"] = max(ss)

        # match_ratio_recent_3: fraction of p1..p3 that match (recency-weighted)
        r["match_ratio_recent_3"] = sum(matches[:3]) / 3.0

        # cur_bash_and_consec_high: interaction — bash command in a deep run
        r["cur_bash_and_consec_high"] = (
            r["is_bash"] * (1.0 if r["max_consecutive_match"] >= 0.6 else 0.0)
        )

        # ── Tier 3 interaction features ───────────────────────────────────
        consec_high = 1.0 if r["max_consecutive_match"] >= 0.6 else 0.0
        is_edit = 1.0 if tool_idx == EDIT_IDX else 0.0
        r["cur_edit_and_consec_high"] = is_edit * consec_high
        r["cur_search_and_consec_high"] = r["is_search"] * consec_high
        r["cur_err_and_consec_high"] = cur_err * consec_high
        r["cur_bash_and_match_ratio"] = r["is_bash"] * r["match_ratio_5"]
        r["cur_err_and_match_ratio"] = cur_err * r["match_ratio_5"]
        r["cur_err_and_bash"] = cur_err * r["is_bash"]

        # Tier 4 — diagnosed from FN/FP analysis on 03_llvm:
        # output repeats without matching action (stuck on output, changing approach)
        r["ssmax_and_no_match"] = r["self_sim_max"] * (1.0 - r["match_ratio_5"])
        # action repeats without matching output (same cmd, different results)
        r["match_and_no_ssmax"] = r["match_ratio_5"] * (1.0 - r["self_sim_max"])

        # action_bigram_repeat: does the prior history show ABAB?
        # p1==p3 AND p2==p4 in cmd_hash terms — needs ≥4 prior actions.
        if len(action_history) >= 4:
            a1 = action_history[-1]  # p1
            a2 = action_history[-2]  # p2
            a3 = action_history[-3]  # p3
            a4 = action_history[-4]  # p4
            r["action_bigram_repeat"] = 1.0 if (a1 == a3 and a2 == a4 and a1 != a2) else 0.0
        else:
            r["action_bigram_repeat"] = 0.0

        # Update histories AFTER recording
        cmd_hash_int = int(round(r.get("cmd_hash", 0.0) * 1_000_000))
        action_history.append(cmd_hash_int)
        if len(action_history) > 5: action_history.pop(0)
        err_history.append(cur_err)
        if len(err_history) > 5: err_history.pop(0)


def load_and_annotate(manifest_path: str) -> list[dict]:
    """Load all rows from the manifest, group by session, annotate Tier 1
    features, return flat list with all original + new fields."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    datasets_cfg = manifest.get("datasets", manifest)

    all_rows: list[dict] = []
    for entry in datasets_cfg:
        path = entry["path"]
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_rows.append(json.loads(line))

    # Group by session, sort by step, annotate
    by_sess = defaultdict(list)
    for r in all_rows:
        by_sess[r["session_id"]].append(r)
    for sid, rs in by_sess.items():
        rs.sort(key=lambda r: r.get("step", 0))
        compute_tier1_features(rs)

    return all_rows


# ── LR validation — are the new features correlation-safe? ────────────────

def validate_features(manifest_path: str):
    """Check each Tier 1 feature's correlation direction on both datasets."""
    from scipy.stats import pointbiserialr
    from src.pipeline.parsers.nlile import parse_session
    from src.pipeline.extract_features import compute_step_features

    print("Loading + annotating training data...")
    train_rows = load_and_annotate(manifest_path)
    # Balanced 50k sample
    random.seed(42)
    random.shuffle(train_rows)
    stuck_rows = [r for r in train_rows if r.get("label", 0.0) >= 0.9][:25000]
    prod_rows = [r for r in train_rows if r.get("label", 0.0) <= 0.1][:25000]
    indist = stuck_rows + prod_rows
    random.shuffle(indist)
    print(f"  in-dist: {len(indist)} balanced rows")

    print("Loading benchmark transcripts + computing Tier 1 features OOD...")
    ood_rows = []
    for td in sorted(Path("benchmarks/results/comparison_off").iterdir()):
        if not td.is_dir(): continue
        t = td / "transcript_1.jsonl"
        lp = td / "sonnet_labels.json"
        if not (t.exists() and lp.exists()): continue
        messages = []
        for line in t.read_text().splitlines():
            if not line.strip(): continue
            try: ev = json.loads(line)
            except: continue
            if ev.get("type") in ("user", "assistant"):
                msg = ev.get("message", {})
                if isinstance(msg, dict): messages.append(msg)
        steps = parse_session(messages)
        feats = compute_step_features(steps)
        labels = json.loads(lp.read_text())["labels"]
        n = min(len(feats), len(labels))
        # Add session_id + step fields that compute_tier1_features expects
        session_id = f"bench_{td.name}"
        for i in range(n):
            feats[i]["session_id"] = session_id
            feats[i]["step"] = i
            feats[i]["label"] = (1.0 if labels[i] == "STUCK" else
                                 0.5 if labels[i] == "UNSURE" else 0.0)
        compute_tier1_features(feats[:n])
        for i in range(n):
            if labels[i] != "UNSURE":
                ood_rows.append(feats[i])
    print(f"  OOD: {len(ood_rows)} rows (non-UNSURE)")

    def extract(rows, feat_name):
        X = np.array([float(r[feat_name]) for r in rows], dtype=np.float64)
        y = np.array([1 if r.get("label", 0) >= 0.9 else 0 for r in rows], dtype=np.int32)
        return X, y

    print(f"\n{'feature':<28}{'ind_r':>10}{'ood_r':>10}{'verdict':>15}")
    print("-" * 63)
    for name in TIER1_NAMES:
        X_i, y_i = extract(indist, name)
        X_o, y_o = extract(ood_rows, name)
        r_i = pointbiserialr(y_i, X_i).statistic if X_i.std() > 1e-9 else 0.0
        r_o = pointbiserialr(y_o, X_o).statistic if X_o.std() > 1e-9 else 0.0
        i_sign = "+" if r_i > 0.02 else ("-" if r_i < -0.02 else "≈0")
        o_sign = "+" if r_o > 0.02 else ("-" if r_o < -0.02 else "≈0")
        if i_sign == o_sign and i_sign != "≈0":
            v = "KEEP"
        elif (i_sign == "+" and o_sign == "-") or (i_sign == "-" and o_sign == "+"):
            v = "DROP (flip)"
        elif i_sign == "≈0" and o_sign == "≈0":
            v = "DROP (weak)"
        elif o_sign == "≈0":
            v = "MARGINAL"
        else:
            v = "?"
        print(f"{name:<28}{r_i:>+10.4f}{r_o:>+10.4f}{v:>15}")


# ── Training ──────────────────────────────────────────────────────────────

class V9Tier1MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class Dataset_(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def build_inputs(rows, feature_names, drop_unsure=True):
    X, y = [], []
    for r in rows:
        lbl = r.get("label", 0.0)
        if drop_unsure and 0.3 < lbl < 0.7:
            continue
        X.append([float(r[k]) for k in feature_names])
        y.append(1.0 if lbl >= 0.9 else 0.0)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def session_split(rows, test_fraction=0.1, seed=DEFAULT_SEED):
    by = defaultdict(list)
    for r in rows: by[r["session_id"]].append(r)
    ids = sorted(by.keys())
    rng = random.Random(seed); rng.shuffle(ids)
    n_test = max(1, int(len(ids) * test_fraction))
    test_ids = set(ids[:n_test])
    tr, te = [], []
    for sid, rs in by.items():
        (te if sid in test_ids else tr).extend(rs)
    return tr, te


def metrics_at(preds, labels, threshold):
    pred = (preds >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    return prec, rec, f1, tp, fp, fn


def train(manifest_path, output_dir, feature_names, seed=DEFAULT_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    input_dim = len(feature_names)
    print(f"\nv9_tier1 training — {input_dim} features")
    print(f"Features: {feature_names}")
    print(f"Output: {output_dir}")

    print("Loading + annotating training data...")
    all_rows = load_and_annotate(manifest_path)
    print(f"  total rows: {len(all_rows)}")

    train_rows, test_rows = session_split(all_rows, seed=seed)
    X_train, y_train = build_inputs(train_rows, feature_names)
    X_test, y_test = build_inputs(test_rows, feature_names)
    print(f"  train={len(X_train)}  stuck={int(y_train.sum())}")
    print(f"  test={len(X_test)}  stuck={int(y_test.sum())}")

    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]; y_train = y_train[perm]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0).clip(min=1e-6)
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    train_loader = DataLoader(Dataset_(X_train_n, y_train), batch_size=512, shuffle=True)
    test_loader = DataLoader(Dataset_(X_test_n, y_test), batch_size=1024)

    num_pos = int(y_train.sum())
    num_neg = len(y_train) - num_pos
    base_pw = num_neg / max(num_pos, 1)
    pw_mult = float(os.environ.get("POS_WEIGHT_MULT", "1.0"))
    pos_weight = torch.tensor([base_pw * pw_mult])
    print(f"  pos_weight={pos_weight.item():.1f}")

    model = V9Tier1MLP(input_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {total_params}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = -1.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve = 0
    for epoch in range(40):
        model.train()
        tot, nb = 0.0, 0
        for inp, lab in train_loader:
            opt.zero_grad()
            loss = crit(model(inp), lab)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        model.eval()
        all_s, all_l = [], []
        with torch.no_grad():
            for inp, lab in test_loader:
                s = torch.sigmoid(model(inp))
                all_s.extend(s.numpy()); all_l.extend(lab.numpy())
        scores = np.array(all_s)
        binary_labels = (np.array(all_l) >= 0.5).astype(int)
        prec, rec, f1, tp, fp, fn = metrics_at(scores, binary_labels, 0.5)
        print(f"  epoch {epoch:2d}: loss={tot/nb:.4f}  P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 5:
                print(f"  early stop at {epoch}"); break
    model.load_state_dict(best_state)

    model.eval()
    all_s, all_l = [], []
    with torch.no_grad():
        for inp, lab in test_loader:
            s = torch.sigmoid(model(inp))
            all_s.extend(s.numpy()); all_l.extend(lab.numpy())
    scores = np.array(all_s)
    binary_labels = (np.array(all_l) >= 0.5).astype(int)
    prec, rec, f1, tp, fp, fn = metrics_at(scores, binary_labels, 0.5)
    print(f"\n=== Final in-dist: P={prec:.3f} R={rec:.3f} F1={f1:.3f} ===")

    os.makedirs(output_dir, exist_ok=True)
    final = {"precision": float(prec), "recall": float(rec), "f1": float(f1),
             "tp": tp, "fp": fp, "fn": fn, "threshold": 0.5}
    torch.save({
        "model_state": model.state_dict(),
        "norm_mean": mean.tolist(),
        "norm_std": std.tolist(),
        "threshold": 0.5,
        "metrics": final,
        "total_params": total_params,
        "architecture": "v9_tier1",
        "input_dim": input_dim,
        "feature_names": feature_names,
    }, os.path.join(output_dir, "stuck_checkpoint.pt"))
    w = {k: v.detach().cpu().numpy().tolist() for k, v in model.named_parameters()}
    w["norm_mean"] = mean.tolist(); w["norm_std"] = std.tolist()
    with open(os.path.join(output_dir, "stuck_weights.json"), "w") as f:
        json.dump(w, f)
    with open(os.path.join(output_dir, "stuck_config.json"), "w") as f:
        json.dump({
            "threshold": 0.5,
            "architecture": "v9_tier1",
            "input_dim": input_dim,
            "num_features": input_dim,
            "feature_names": feature_names,
            "total_params": total_params,
            "metrics": final,
        }, f, indent=2)
    print(f"  saved: {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="training_manifest_v6.json")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--output-dir", default="proxy/experiments/v9_tier1")
    ap.add_argument("--features", nargs="+", default=None,
                    help="Override feature list (default: full 15-dim batch)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    if args.validate:
        validate_features(args.manifest)
    if args.train:
        feats = args.features or FULL_FEATURE_NAMES
        train(args.manifest, args.output_dir, feats, args.seed)
    if not args.validate and not args.train:
        print("nothing to do — pass --validate or --train")


if __name__ == "__main__":
    main()
