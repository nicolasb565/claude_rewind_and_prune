#!/usr/bin/env python3
"""
v9 MLP trainer — standalone, uses the 34 relational features from
extract_features.py schema 6.

v9 features already embed 5 steps of history in the feature vector, so
this trainer does NOT use a ring buffer. Each training example is just a
34-dim flat vector per labeled step.

Architecture:
  Linear(34, 32) → ReLU → Linear(32, 16) → ReLU → Linear(16, 1) → sigmoid
  ≈ 1,650 parameters. Deliberately small because the features carry more
  signal per dim than the schema-3 v5 features.

Usage:
  .venv/bin/python benchmarks/v9_train.py
  .venv/bin/python benchmarks/v9_train.py --manifest training_manifest_v6.json
  POS_WEIGHT_MULT=3 .venv/bin/python benchmarks/v9_train.py  # pos-class reweight
"""

from __future__ import annotations

import argparse
import json
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

from src.pipeline.extract_features import V9_FEATURE_NAMES  # noqa: E402

DEFAULT_SEED = 42
INPUT_DIM = 34
assert len(V9_FEATURE_NAMES) == INPUT_DIM


class V9Dataset(Dataset):
    def __init__(self, inputs: np.ndarray, labels: np.ndarray):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return self.inputs[i], self.labels[i]


class V9MLP(nn.Module):
    """Small MLP on flat 34-dim v9 features."""

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


def load_rows(jsonl_path: str) -> list[dict]:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_inputs(rows: list[dict], drop_unsure: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y, sids = [], [], []
    for r in rows:
        label = r.get("label", 0.0)
        if drop_unsure and 0.3 < label < 0.7:  # exclude UNSURE (label=0.5)
            continue
        vec = [float(r[k]) for k in V9_FEATURE_NAMES]
        X.append(vec)
        # Binarize: STUCK=1 (label>=0.9), everything else=0
        y.append(1.0 if label >= 0.9 else 0.0)
        sids.append(r["session_id"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), sids


def session_split(rows: list[dict], test_fraction: float = 0.1, seed: int = DEFAULT_SEED):
    by_session = defaultdict(list)
    for r in rows:
        by_session[r["session_id"]].append(r)
    session_ids = sorted(by_session.keys())
    rng = random.Random(seed)
    rng.shuffle(session_ids)
    n_test = max(1, int(len(session_ids) * test_fraction))
    test_ids = set(session_ids[:n_test])
    train_rows, test_rows = [], []
    for sid, rs in by_session.items():
        if sid in test_ids:
            test_rows.extend(rs)
        else:
            train_rows.extend(rs)
    return train_rows, test_rows


def metrics_at(preds, labels, threshold):
    pred = (preds >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    return prec, rec, f1, tp, fp, fn, tn


def train_v9(
    manifest_path: str,
    output_dir: str,
    seed: int = DEFAULT_SEED,
    drop_unsure: bool = True,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    with open(manifest_path) as f:
        manifest = json.load(f)
    datasets_cfg = manifest.get("datasets", manifest)

    print(f"\nv9 training — input_dim={INPUT_DIM}, 1,650 params, no ring buffer")
    print(f"Output directory: {output_dir}")
    print(f"Drop UNSURE labels: {drop_unsure}")
    print()

    all_rows: list[dict] = []
    for entry in datasets_cfg:
        path = entry["path"]
        weight = float(entry.get("weight", 1.0))
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        rows = load_rows(path)
        copies = max(1, int(round(weight)))
        all_rows.extend(rows * copies)
        print(f"  {path}: {len(rows)} rows × {copies}")

    if not all_rows:
        print("ERROR: no data loaded", file=sys.stderr)
        sys.exit(1)

    train_rows, test_rows = session_split(all_rows, seed=seed)
    print(f"\nTrain/test: {len({r['session_id'] for r in train_rows})} sess / "
          f"{len({r['session_id'] for r in test_rows})} sess")

    X_train, y_train, _ = build_inputs(train_rows, drop_unsure=drop_unsure)
    X_test, y_test, _ = build_inputs(test_rows, drop_unsure=drop_unsure)
    print(f"  Train rows: {len(X_train)}  stuck={int(y_train.sum())} ({y_train.mean()*100:.1f}%)")
    print(f"  Test rows:  {len(X_test)}  stuck={int(y_test.sum())} ({y_test.mean()*100:.1f}%)")

    # Shuffle
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    # Normalize per-column (mean/std)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0).clip(min=1e-6)
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    train_ds = V9Dataset(X_train_n, y_train)
    test_ds = V9Dataset(X_test_n, y_test)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024)

    num_pos = int(y_train.sum())
    num_neg = len(y_train) - num_pos
    base_pw = num_neg / max(num_pos, 1)
    pw_mult = float(os.environ.get("POS_WEIGHT_MULT", "1.0"))
    pos_weight = torch.tensor([base_pw * pw_mult])
    print(f"  pos_weight={pos_weight.item():.1f} (base={base_pw:.1f} × mult={pw_mult})")

    model = V9MLP(INPUT_DIM)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    threshold = 0.5
    best_f1 = -1.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve = 0
    patience = 5

    print("\nTraining...")
    for epoch in range(40):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for inp, lab in train_loader:
            optimizer.zero_grad()
            logits = model(inp)
            loss = criterion(logits, lab)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        all_s, all_l = [], []
        with torch.no_grad():
            for inp, lab in test_loader:
                s = torch.sigmoid(model(inp))
                all_s.extend(s.numpy())
                all_l.extend(lab.numpy())
        scores = np.array(all_s)
        binary_labels = (np.array(all_l) >= 0.5).astype(int)
        prec, rec, f1, tp, fp, fn, _ = metrics_at(scores, binary_labels, threshold)
        print(f"  Epoch {epoch:2d}: loss={total_loss/n_batches:.4f} "
              f"t={threshold} P={prec:.3f} R={rec:.3f} F1={f1:.3f} FP={fp} FN={fn}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    # Final eval
    model.eval()
    all_s, all_l = [], []
    with torch.no_grad():
        for inp, lab in test_loader:
            s = torch.sigmoid(model(inp))
            all_s.extend(s.numpy())
            all_l.extend(lab.numpy())
    scores = np.array(all_s)
    binary_labels = (np.array(all_l) >= 0.5).astype(int)
    prec, rec, f1, tp, fp, fn, tn = metrics_at(scores, binary_labels, threshold)
    print(f"\n=== Final test metrics at threshold={threshold} ===")
    print(f"  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  TP={tp} FP={fp} FN={fn} TN={tn}")

    stuck_scores = scores[binary_labels == 1]
    neg_scores = scores[binary_labels == 0]
    print(f"\n  STUCK (n={len(stuck_scores)}): median={np.median(stuck_scores):.3f}  "
          f"p95={np.percentile(stuck_scores, 95):.3f}")
    print(f"  PROD  (n={len(neg_scores)}): median={np.median(neg_scores):.3f}  "
          f"p95={np.percentile(neg_scores, 95):.3f}")

    print("\n=== Threshold sweep ===")
    print(f"  {'t':>5}  {'P':>6}  {'R':>6}  {'F1':>6}  {'FP':>6}  {'FN':>6}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p_t, r_t, f_t, _, fp_t, fn_t, _ = metrics_at(scores, binary_labels, t)
        print(f"  {t:>5.2f}  {p_t:>6.3f}  {r_t:>6.3f}  {f_t:>6.3f}  {fp_t:>6}  {fn_t:>6}")

    os.makedirs(output_dir, exist_ok=True)
    final_metrics = {"precision": float(prec), "recall": float(rec), "f1": float(f1),
                     "tp": tp, "fp": fp, "fn": fn, "tn": tn, "threshold": threshold}
    torch.save({
        "model_state": model.state_dict(),
        "norm_mean": mean.tolist(),
        "norm_std": std.tolist(),
        "threshold": threshold,
        "metrics": final_metrics,
        "total_params": total_params,
        "architecture": "v9",
        "input_dim": INPUT_DIM,
    }, os.path.join(output_dir, "stuck_checkpoint.pt"))

    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy().tolist()
    weights["norm_mean"] = mean.tolist()
    weights["norm_std"] = std.tolist()
    with open(os.path.join(output_dir, "stuck_weights.json"), "w") as f:
        json.dump(weights, f)

    config = {
        "threshold": threshold,
        "model_stage": 9,
        "architecture": "v9",
        "input_dim": INPUT_DIM,
        "num_features": INPUT_DIM,
        "use_score_history": False,
        "excluded_features": [],
        "step_features": list(V9_FEATURE_NAMES),
        "total_params": total_params,
        "metrics": final_metrics,
    }
    with open(os.path.join(output_dir, "stuck_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved:")
    print(f"  {output_dir}/stuck_checkpoint.pt")
    print(f"  {output_dir}/stuck_weights.json")
    print(f"  {output_dir}/stuck_config.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="training_manifest_v6.json")
    ap.add_argument("--output-dir", default="proxy/experiments/v9")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--keep-unsure", action="store_true",
                    help="Don't drop UNSURE-labeled rows (default: drop them)")
    args = ap.parse_args()
    train_v9(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        seed=args.seed,
        drop_unsure=not args.keep_unsure,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
