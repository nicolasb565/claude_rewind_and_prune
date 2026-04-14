#!/usr/bin/env python3
"""
Evaluate the v9 MLP (34 relational features) on the benchmark transcripts
head-to-head against the current production proxy.

For each checkpoint:
  1. Parse each task's stream-json transcript into normalized step dicts
  2. Compute per-step features via extract_features.compute_step_features
     (which returns both v5 and v9 fields in schema 6)
  3. For v9: feed 34-dim flat input directly (no ring buffer)
     For v5: ring-buffer the 7 kept features over 5 history slots (42-dim)
  4. Score every step with the loaded MLP
  5. Compare to Sonnet labels

Reports pooled AUC, precision, recall, F1, plus per-task breakdowns.

Usage:
  .venv/bin/python benchmarks/v9_eval.py
  .venv/bin/python benchmarks/v9_eval.py --models v5_baseline v9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.pipeline.parsers.nlile import parse_session  # noqa: E402
from src.pipeline.extract_features import (  # noqa: E402
    compute_step_features,
    STEP_FEATURES as V5_STEP_FEATURES,
    V9_FEATURE_NAMES,
)


KNOWN_MODELS = {
    "v5_baseline": REPO / "proxy",
    "v9": REPO / "proxy" / "experiments" / "v9",
    "v9_pw3": REPO / "proxy" / "experiments" / "v9_pw3",
    "v9_pw5": REPO / "proxy" / "experiments" / "v9_pw5",
}


class V5MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class V9MLP(nn.Module):
    def __init__(self, input_dim: int = 34):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


def load_model(checkpoint_dir: Path):
    ckpt = torch.load(checkpoint_dir / "stuck_checkpoint.pt", weights_only=False)
    config = json.loads((checkpoint_dir / "stuck_config.json").read_text())
    architecture = config.get("architecture", "v5")
    input_dim = config["input_dim"]
    if architecture == "v9":
        model = V9MLP(input_dim)
    else:
        model = V5MLP(input_dim)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = np.array(ckpt["norm_mean"], dtype=np.float32)
    std = np.array(ckpt["norm_std"], dtype=np.float32)
    return model, mean, std, config, architecture


def parse_transcript_to_steps(path: Path) -> list[dict]:
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


def build_v5_inputs(features: list[dict], config: dict) -> np.ndarray:
    """Ring-buffer build for v5 architecture (same as src/training/train.py)."""
    excluded = set(config.get("excluded_features", []))
    kept_features = [f for f in V5_STEP_FEATURES if f not in excluded]
    n_kept = len(kept_features)
    n_history = config.get("n_history", 5)
    use_score_history = config.get("use_score_history", False)
    input_dim = config["input_dim"]

    inputs = np.zeros((len(features), input_dim), dtype=np.float32)
    feat_buf = np.zeros((n_history, n_kept), dtype=np.float32)
    for i, row in enumerate(features):
        curr = np.array([float(row[f]) for f in kept_features], dtype=np.float32)
        if use_score_history:
            inp = np.concatenate([curr, feat_buf.flatten(), np.zeros(n_history, dtype=np.float32)])
        else:
            inp = np.concatenate([curr, feat_buf.flatten()])
        inputs[i] = inp
        feat_buf = np.roll(feat_buf, 1, axis=0)
        feat_buf[0] = curr
    return inputs


def build_v9_inputs(features: list[dict]) -> np.ndarray:
    """Flat 34-dim per step — no ring buffer (history already embedded)."""
    return np.array(
        [[float(r[k]) for k in V9_FEATURE_NAMES] for r in features],
        dtype=np.float32,
    )


def evaluate(model_name: str, model_dir: Path, run_dir: Path, verbose: bool):
    model, mean, std, config, architecture = load_model(model_dir)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n========== {model_name} ({architecture}, "
          f"input_dim={config['input_dim']}, params={params}) ==========")

    all_scores, all_labels = [], []
    per_task = []

    for task_dir in sorted(run_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        labels_path = task_dir / "sonnet_labels.json"
        transcript = task_dir / "transcript_1.jsonl"
        if not (labels_path.exists() and transcript.exists()):
            continue

        steps = parse_transcript_to_steps(transcript)
        if not steps:
            continue
        features = compute_step_features(steps)
        sonnet = json.loads(labels_path.read_text())
        labels = sonnet["labels"]
        n = min(len(features), len(labels))
        features = features[:n]
        labels = labels[:n]

        if architecture == "v9":
            inputs = build_v9_inputs(features)
        else:
            inputs = build_v5_inputs(features, config)
        inputs = (inputs - mean) / std
        with torch.no_grad():
            scores = torch.sigmoid(model(torch.tensor(inputs, dtype=torch.float32))).numpy()

        threshold = config.get("threshold", 0.5)
        sonnet_stuck = sum(1 for lbl in labels if lbl == "STUCK")
        mlp_hot = int((scores >= threshold).sum())
        agree_stuck = sum(
            1 for s, lbl in zip(scores, labels)
            if s >= threshold and lbl == "STUCK"
        )
        miss = sonnet_stuck - agree_stuck
        fp = mlp_hot - agree_stuck

        per_task.append({
            "task": task_dir.name,
            "n": n,
            "sonnet_stuck": sonnet_stuck,
            "mlp_hot": mlp_hot,
            "agree_stuck": agree_stuck,
            "miss": miss,
            "fp": fp,
            "max_score": float(scores.max()) if len(scores) else 0.0,
        })
        for s, lbl in zip(scores, labels):
            if lbl == "UNSURE":
                continue
            all_scores.append(float(s))
            all_labels.append(1 if lbl == "STUCK" else 0)

    print(f"{'task':<22}{'n':>5}{'son_stk':>8}{'mlp_hot':>9}{'agree':>7}"
          f"{'miss':>6}{'fp':>5}{'max':>8}")
    for r in per_task:
        print(f"{r['task']:<22}{r['n']:>5}{r['sonnet_stuck']:>8}{r['mlp_hot']:>9}"
              f"{r['agree_stuck']:>7}{r['miss']:>6}{r['fp']:>5}{r['max_score']:>8.3f}")

    arr_s = np.array(all_scores)
    arr_l = np.array(all_labels)
    auc = roc_auc_score(arr_l, arr_s) if 0 < arr_l.sum() < len(arr_l) else float("nan")
    threshold = config.get("threshold", 0.5)
    pred = (arr_s >= threshold).astype(int)
    tp = int(((pred == 1) & (arr_l == 1)).sum())
    fp = int(((pred == 1) & (arr_l == 0)).sum())
    fn = int(((pred == 0) & (arr_l == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\nPOOLED  AUC={auc:.4f}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
          f"TP={tp} FP={fp} FN={fn}  (n={len(arr_l)})")

    return {"model": model_name, "architecture": architecture, "auc": float(auc),
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn, "per_task": per_task}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["v5_baseline", "v9"])
    ap.add_argument("--run-dir", default="benchmarks/results/comparison_off")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    results = []
    for name in args.models:
        if name not in KNOWN_MODELS:
            print(f"unknown model: {name}", file=sys.stderr)
            return 1
        if not (KNOWN_MODELS[name] / "stuck_checkpoint.pt").exists():
            print(f"SKIP {name}: checkpoint not found at {KNOWN_MODELS[name]}")
            continue
        results.append(evaluate(name, KNOWN_MODELS[name], run_dir, args.verbose))

    print("\n" + "=" * 90)
    print("HEAD-TO-HEAD")
    print("=" * 90)
    print(f"{'model':<22}{'arch':<6}{'AUC':>8}{'P':>8}{'R':>8}{'F1':>8}{'TP':>6}{'FP':>6}{'FN':>6}")
    for r in results:
        print(f"{r['model']:<22}{r['architecture']:<6}"
              f"{r['auc']:>8.4f}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['f1']:>8.3f}"
              f"{r['tp']:>6}{r['fp']:>6}{r['fn']:>6}")

    out_path = run_dir / "v9_eval.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nfull results: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
