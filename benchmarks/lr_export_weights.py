#!/usr/bin/env python3
"""
Train the production LR classifier on the in-distribution content features,
dump its weights + normalization stats to proxy/lr_weights.json, and write
per-step OOD scores (indexed by session + step) to
benchmarks/results/lr_scores_ood.json for the JS benchmark replay.

The JSON weight file is the shippable artifact — the JS proxy can load it
directly once the feature extraction is ported.
"""

from __future__ import annotations

import json
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


def main() -> int:
    d = json.loads((REPO / "data/generated/content_prototype.json").read_text())
    indist = d["indist"]
    ood = d["ood"]

    X_tr, y_tr, _ = build_xy(indist)
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0).clip(min=1e-6)
    X_tr_n = (X_tr - mean) / std

    pos = int(y_tr.sum())
    neg = len(y_tr) - pos
    lr = LogisticRegression(
        C=1.0,
        class_weight={0: 1.0, 1: neg / max(pos, 1)},
        max_iter=2000,
        solver="lbfgs",
    )
    lr.fit(X_tr_n, y_tr)
    print(f"trained: {pos} stuck / {len(y_tr)} rows "
          f"({100*pos/len(y_tr):.1f}% pos)")
    for name, w in zip(FEATS, lr.coef_[0]):
        print(f"  {name:<32} = {w:+.4f}")
    print(f"  intercept                     = {lr.intercept_[0]:+.4f}")

    weights_path = REPO / "proxy" / "lr_weights.json"
    payload = {
        "model": "lr_content_v1",
        "features": FEATS,
        "weights": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "training": {
            "n_rows": int(len(y_tr)),
            "n_stuck": int(pos),
            "n_productive": int(neg),
            "class_weight": {"0": 1.0, "1": float(neg / max(pos, 1))},
            "solver": "lbfgs",
            "C": 1.0,
        },
    }
    weights_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {weights_path}")

    # Compute per-step scores for the full OOD set (including UNSURE rows so
    # the simulator can walk sessions end-to-end).
    all_X = np.array([[r[k] for k in FEATS] for r in ood], dtype=np.float64)
    all_X_n = (all_X - mean) / std
    probs = lr.predict_proba(all_X_n)[:, 1].tolist()

    sessions: dict[str, dict] = {}
    for r, p in zip(ood, probs):
        sid = r["session_id"]
        sess = sessions.setdefault(sid, {"session_id": sid, "steps": []})
        sess["steps"].append({
            "step": int(r["step"]),
            "score": float(p),
            "label": float(r["label"]),
        })

    for sess in sessions.values():
        sess["steps"].sort(key=lambda s: s["step"])

    scores_path = REPO / "benchmarks" / "results" / "lr_scores_ood.json"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(
        {"sessions": list(sessions.values())}, indent=2
    ))
    total_steps = sum(len(s["steps"]) for s in sessions.values())
    print(f"wrote {scores_path}  ({len(sessions)} sessions, {total_steps} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
