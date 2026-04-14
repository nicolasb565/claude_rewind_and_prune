#!/usr/bin/env python3
"""
OOD evaluation for v9_tier1 checkpoints.

Parses each benchmark transcript, computes v9 features + Tier 1
annotations via compute_tier1_features, applies each checkpoint's
norm + feature_names, scores with V9Tier1MLP, and compares to
Sonnet labels.

Usage:
  .venv/bin/python benchmarks/v9_tier1_eval.py \
      --models v9_tier1 v9_tier1_pw3 v9_tier1_ratio v9_tier1_ratio_pw3 \
               v9_tier1_run v9_tier1_run_pw3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.pipeline.parsers.nlile import parse_session  # noqa: E402
from src.pipeline.extract_features import compute_step_features  # noqa: E402
from benchmarks.v9_tier1_train import V9Tier1MLP, compute_tier1_features  # noqa: E402

EXP_ROOT = REPO / "proxy" / "experiments"
RUN_DIR = REPO / "benchmarks" / "results" / "comparison_off"


def load_model(ckpt_dir: Path):
    ckpt = torch.load(ckpt_dir / "stuck_checkpoint.pt", weights_only=False)
    config = json.loads((ckpt_dir / "stuck_config.json").read_text())
    input_dim = config["input_dim"]
    model = V9Tier1MLP(input_dim)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = np.array(ckpt["norm_mean"], dtype=np.float32)
    std = np.array(ckpt["norm_std"], dtype=np.float32)
    feature_names = config["feature_names"]
    return model, mean, std, feature_names, config


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


def build_ood_dataset():
    """Return list of (task_name, features, labels) with tier1 annotations."""
    tasks = []
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
        feats = feats[:n]
        labels = labels[:n]
        compute_tier1_features(feats)
        tasks.append((td.name, feats, labels))
    return tasks


def evaluate(model_name: str, tasks):
    model, mean, std, feature_names, config = load_model(EXP_ROOT / model_name)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n========== {model_name} "
          f"(dim={config['input_dim']}, params={params}) ==========")

    all_s, all_l = [], []
    per_task = []
    for task_name, feats, labels in tasks:
        X = np.array(
            [[float(r[k]) for k in feature_names] for r in feats],
            dtype=np.float32,
        )
        Xn = (X - mean) / std
        with torch.no_grad():
            scores = torch.sigmoid(model(torch.tensor(Xn))).numpy()

        threshold = config.get("threshold", 0.5)
        sonnet_stuck = sum(1 for lbl in labels if lbl == "STUCK")
        mlp_hot = int((scores >= threshold).sum())
        agree = sum(1 for s, lbl in zip(scores, labels)
                    if s >= threshold and lbl == "STUCK")
        per_task.append({
            "task": task_name, "n": len(feats),
            "son_stk": sonnet_stuck, "mlp_hot": mlp_hot,
            "agree": agree, "miss": sonnet_stuck - agree,
            "fp": mlp_hot - agree,
            "max": float(scores.max()) if len(scores) else 0.0,
        })
        for s, lbl in zip(scores, labels):
            if lbl == "UNSURE":
                continue
            all_s.append(float(s))
            all_l.append(1 if lbl == "STUCK" else 0)

    print(f"{'task':<22}{'n':>5}{'stk':>5}{'hot':>5}{'ag':>4}"
          f"{'mi':>4}{'fp':>4}{'max':>8}")
    for r in per_task:
        print(f"{r['task']:<22}{r['n']:>5}{r['son_stk']:>5}{r['mlp_hot']:>5}"
              f"{r['agree']:>4}{r['miss']:>4}{r['fp']:>4}{r['max']:>8.3f}")

    arr_s = np.array(all_s); arr_l = np.array(all_l)
    auc = roc_auc_score(arr_l, arr_s) if 0 < arr_l.sum() < len(arr_l) else float("nan")
    th = config.get("threshold", 0.5)
    pred = (arr_s >= th).astype(int)
    tp = int(((pred == 1) & (arr_l == 1)).sum())
    fp = int(((pred == 1) & (arr_l == 0)).sum())
    fn = int(((pred == 0) & (arr_l == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    llvm = sum(1 for r in per_task if r["task"].startswith("03_llvm") and r["agree"] > 0)
    llvm_total = sum(r["son_stk"] for r in per_task if r["task"].startswith("03_llvm"))
    print(f"\nPOOLED  AUC={auc:.4f}  P={prec:.3f}  R={rec:.3f}  "
          f"F1={f1:.3f}  TP={tp} FP={fp} FN={fn}")

    return {
        "model": model_name, "auc": float(auc),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "tp": tp, "fp": fp, "fn": fn,
        "llvm_agree": llvm, "llvm_total": llvm_total,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()

    print("Loading OOD benchmark transcripts...")
    tasks = build_ood_dataset()
    print(f"  {len(tasks)} tasks loaded")

    results = []
    for name in args.models:
        if not (EXP_ROOT / name / "stuck_checkpoint.pt").exists():
            print(f"SKIP {name}: checkpoint not found")
            continue
        results.append(evaluate(name, tasks))

    print("\n" + "=" * 85)
    print("HEAD-TO-HEAD (OOD)")
    print("=" * 85)
    print(f"{'model':<24}{'AUC':>8}{'P':>8}{'R':>8}{'F1':>8}"
          f"{'TP':>6}{'FP':>6}{'FN':>6}")
    for r in results:
        print(f"{r['model']:<24}{r['auc']:>8.4f}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['f1']:>8.3f}"
              f"{r['tp']:>6}{r['fp']:>6}{r['fn']:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
