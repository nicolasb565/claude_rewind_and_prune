"""Train per-step MLP v5 for stuck detection using N-step history ring buffer.

Input: training_manifest.json (list of JSONL files with weights)
Data format: per-step JSONL rows (each row has all STEP_FEATURES + label + session_id)
Architecture: v5 — MLP over [current_features(8) + prev_features×5(40) + prev_scores×5(5)]
              Linear(53,64) → ReLU → Linear(64,32) → ReLU → Linear(32,1) → sigmoid

Usage:
  python train.py [--manifest training_manifest.json]
"""

import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
MODEL_DIR = "proxy"

N_HISTORY = 5

STEP_FEATURES = [
    "tool_idx",
    "cmd_hash",
    "file_hash",
    "output_similarity",
    "has_prior_output",
    "output_length",
    "is_error",
    "step_index_norm",
]

NUM_FEATURES = len(STEP_FEATURES)  # 8
INPUT_DIM = NUM_FEATURES * (1 + N_HISTORY) + N_HISTORY  # 53


class StuckDetectorV5(nn.Module):
    """Per-step MLP with N-step history ring buffer.

    Input: [current_features(8), prev×5_features(40), prev×5_scores(5)] = 53 floats
    Architecture: Linear(53,64) → ReLU → Linear(64,32) → ReLU → Linear(32,1)
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class StepDataset(Dataset):
    def __init__(self, inputs: np.ndarray, labels: np.ndarray):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[i], self.labels[i]


def build_sequences(
    rows_by_session: dict[str, list[dict]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (input_53, label) pairs for all steps with ring-buffer history.

    For each step T the input is:
      [features_T, features_T-1, ..., features_T-5, score_T-1, ..., score_T-5]

    History positions are zero-padded at the start of each session.
    Previous scores use the soft label (0.0/0.5/1.0) as a proxy for the model's
    own ring-buffer score at inference time.

    Returns:
        inputs: float32 array of shape (n_steps, INPUT_DIM)
        labels: float32 array of shape (n_steps,)
        session_ids: list of session_id strings, one per step
    """
    all_inputs: list[np.ndarray] = []
    all_labels: list[float] = []
    all_session_ids: list[str] = []

    for sid in sorted(rows_by_session.keys()):
        rows = sorted(rows_by_session[sid], key=lambda r: r["step"])

        feat_buf = np.zeros((N_HISTORY, NUM_FEATURES), dtype=np.float32)
        score_buf = np.zeros(N_HISTORY, dtype=np.float32)

        for row in rows:
            curr = np.array([float(row[f]) for f in STEP_FEATURES], dtype=np.float32)

            # [current(8), prev_T-1(8), ..., prev_T-5(8), score_T-1, ..., score_T-5]
            inp = np.concatenate([curr, feat_buf.flatten(), score_buf])
            all_inputs.append(inp)
            all_labels.append(float(row["label"]))
            all_session_ids.append(sid)

            # Shift ring buffer: roll down, overwrite position 0 with most recent
            feat_buf = np.roll(feat_buf, 1, axis=0)
            feat_buf[0] = curr
            score_buf = np.roll(score_buf, 1)
            score_buf[0] = float(row["label"])

    return (
        np.array(all_inputs, dtype=np.float32),
        np.array(all_labels, dtype=np.float32),
        all_session_ids,
    )


def load_manifest(manifest_path: str) -> list[dict]:
    """Load training manifest."""
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def load_rows_from_jsonl(path: str) -> list[dict]:
    """Load per-step JSONL rows."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def session_split(
    rows: list[dict],
    test_fraction: float = 0.1,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Split rows by session_id (90/10 split).

    Returns:
        (train_by_session, test_by_session) dicts mapping session_id → rows
    """
    rows_by_session: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        rows_by_session[r["session_id"]].append(r)

    session_ids = sorted(rows_by_session.keys())
    rng = random.Random(SEED)
    rng.shuffle(session_ids)

    n_test = max(1, int(len(session_ids) * test_fraction))
    test_ids = set(session_ids[:n_test])
    train_ids = set(session_ids[n_test:])

    train_by_session = {sid: rows_by_session[sid] for sid in train_ids}
    test_by_session = {sid: rows_by_session[sid] for sid in test_ids}

    # Print split stats
    train_rows = [r for rows in train_by_session.values() for r in rows]
    test_rows = [r for rows in test_by_session.values() for r in rows]
    train_stuck = sum(1 for r in train_rows if r["label"] >= 0.9) / max(len(train_rows), 1)
    test_stuck = sum(1 for r in test_rows if r["label"] >= 0.9) / max(len(test_rows), 1)

    print(f"Train/test split: {len(train_ids)} sessions train, {len(test_ids)} sessions test")
    print(f"  Train STUCK prevalence: {train_stuck * 100:.1f}%")
    print(f"  Test  STUCK prevalence: {test_stuck * 100:.1f}%")

    return train_by_session, test_by_session


def metrics_at(
    preds: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[float, float, float, int, int, int, int]:
    pred = (preds >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    return prec, rec, f1, tp, fp, fn, tn


def train(  # pylint: disable=too-many-statements,too-many-locals,too-many-branches
    manifest_path: str = "training_manifest.json",
) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    manifest = load_manifest(manifest_path)
    datasets_cfg = (
        manifest.get("datasets", manifest) if isinstance(manifest, dict) else manifest
    )

    print("Loading data from manifest...")
    all_rows: list[dict] = []
    for entry in datasets_cfg:
        path = entry["path"]
        weight = float(entry.get("weight", 1.0))
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        rows = load_rows_from_jsonl(path)
        copies = max(1, int(round(weight)))
        all_rows.extend(rows * copies)
        print(f"  {path}: {len(rows)} rows × {copies} = {len(rows) * copies}")

    if not all_rows:
        print("ERROR: no data loaded", file=sys.stderr)
        sys.exit(1)

    train_by_session, test_by_session = session_split(all_rows)

    train_inputs, train_labels, _ = build_sequences(train_by_session)
    test_inputs, test_labels, _ = build_sequences(test_by_session)

    # Shuffle training sequences (across sessions — ring buffer already baked in)
    perm = np.random.permutation(len(train_inputs))
    train_inputs = train_inputs[perm]
    train_labels = train_labels[perm]

    # Normalize feature dims only — score dims (last N_HISTORY positions) are left
    # in [0, 1] so inference values (continuous sigmoid outputs) are not shifted
    # by training-set stats derived from trimodal {0, 0.5, 1} labels.
    feat_dims = INPUT_DIM - N_HISTORY
    mean = np.zeros(INPUT_DIM, dtype=np.float32)
    std = np.ones(INPUT_DIM, dtype=np.float32)
    mean[:feat_dims] = train_inputs[:, :feat_dims].mean(axis=0)
    std[:feat_dims] = train_inputs[:, :feat_dims].std(axis=0).clip(min=1e-6)
    train_inputs = (train_inputs - mean) / std
    test_inputs = (test_inputs - mean) / std

    train_ds = StepDataset(train_inputs, train_labels)
    test_ds = StepDataset(test_inputs, test_labels)

    num_pos = int((train_labels >= 0.9).sum())
    num_neg = len(train_labels) - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)])
    print(
        f"  Class balance: pos={num_pos} neg={num_neg} pos_weight={pos_weight.item():.1f}"
    )

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024)

    model = StuckDetectorV5()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = -1.0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_improve = 0
    patience = 5
    threshold = 0.5

    print("\nTraining...")
    for epoch in range(30):
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
        binary_labels = (np.array(all_l) >= 0.9).astype(int)
        prec, rec, f1, tp, fp, fn, _ = metrics_at(scores, binary_labels, threshold)
        print(
            f"  Epoch {epoch:2d}: loss={total_loss/n_batches:.4f}  "
            f"t={threshold} P={prec:.3f} R={rec:.3f} F1={f1:.3f} FP={fp} FN={fn}"
        )

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

    # Final evaluation
    model.eval()
    all_s, all_l = [], []
    with torch.no_grad():
        for inp, lab in test_loader:
            s = torch.sigmoid(model(inp))
            all_s.extend(s.numpy())
            all_l.extend(lab.numpy())
    scores = np.array(all_s)
    binary_labels = (np.array(all_l) >= 0.9).astype(int)

    prec, rec, f1, tp, fp, fn, tn = metrics_at(scores, binary_labels, threshold)
    print(f"\n=== Final test metrics at threshold={threshold} ===")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")

    stuck_scores = scores[binary_labels == 1]
    neg_scores = scores[binary_labels == 0]
    pcts = [50, 75, 90, 95, 99]
    print("\n=== Score distribution ===")
    print(f"  STUCK  (n={len(stuck_scores)}): " + "  ".join(
        f"p{p}={np.percentile(stuck_scores, p):.3f}" for p in pcts
    ))
    print(f"  PRODUC (n={len(neg_scores)}): " + "  ".join(
        f"p{p}={np.percentile(neg_scores, p):.3f}" for p in pcts
    ))

    print("\n=== Threshold sweep ===")
    print(f"  {'t':>5}  {'P':>6}  {'R':>6}  {'F1':>6}  {'FP':>6}  {'FN':>6}")
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        p, r, f, _, fp, fn, _ = metrics_at(scores, binary_labels, t)
        print(f"  {t:>5.2f}  {p:>6.3f}  {r:>6.3f}  {f:>6.3f}  {fp:>6}  {fn:>6}")

    final_metrics = {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": threshold,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "norm_mean": mean.tolist(),
            "norm_std": std.tolist(),
            "threshold": threshold,
            "metrics": final_metrics,
            "total_params": total_params,
        },
        os.path.join(MODEL_DIR, "stuck_checkpoint.pt"),
    )

    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy().tolist()
    weights["norm_mean"] = mean.tolist()
    weights["norm_std"] = std.tolist()
    with open(os.path.join(MODEL_DIR, "stuck_weights.json"), "w", encoding="utf-8") as f:
        json.dump(weights, f)

    config = {
        "threshold": threshold,
        "model_stage": 5,
        "n_history": N_HISTORY,
        "num_features": NUM_FEATURES,
        "input_dim": INPUT_DIM,
        "total_params": total_params,
        "metrics": final_metrics,
        "step_features": STEP_FEATURES,
    }
    with open(os.path.join(MODEL_DIR, "stuck_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    size = os.path.getsize(os.path.join(MODEL_DIR, "stuck_weights.json"))
    print("\nSaved:")
    print(f"  {MODEL_DIR}/stuck_checkpoint.pt")
    print(f"  {MODEL_DIR}/stuck_weights.json ({size / 1024:.1f} KB)")
    print(f"  {MODEL_DIR}/stuck_config.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="training_manifest.json")
    _args = parser.parse_args()
    train(_args.manifest)
