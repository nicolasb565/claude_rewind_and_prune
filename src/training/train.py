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

DEFAULT_SEED = 42
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
INPUT_DIM_NO_SCORES = NUM_FEATURES * (1 + N_HISTORY)  # 48


class StuckDetectorV5(nn.Module):
    """Per-step MLP with N-step history ring buffer.

    Input: [current_features(8), prev×5_features(40), prev×5_scores(5)] = 53 floats (default)
           Or 48 floats when score history is disabled (no train/inference mismatch).
    Architecture: Linear(input_dim,64) → ReLU → Linear(64,32) → ReLU → Linear(32,1)
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
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
    use_score_history: bool = True,
    excluded_features: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build (input, label) pairs for all steps with ring-buffer history.

    For each step T the input is:
      [features_T, features_T-1, ..., features_T-5, score_T-1, ..., score_T-5]

    History positions are zero-padded at the start of each session.
    Previous scores use the soft label (0.0/0.5/1.0) as a proxy for the model's
    own ring-buffer score at inference time. This creates a train/inference
    mismatch (training sees perfect labels, inference sees the model's own
    sigmoid output) — set use_score_history=False to drop those 5 dims entirely
    and produce a 48-dim input that is identical at train and inference time.

    excluded_features removes the named features (must be in STEP_FEATURES) from
    both the current step and every history slot — used for ablation studies.

    Returns:
        inputs: float32 array of shape (n_steps, kept_features * (1+N_HISTORY) [+ N_HISTORY])
        labels: float32 array of shape (n_steps,)
        session_ids: list of session_id strings, one per step
    """
    excluded = excluded_features or set()
    unknown = excluded - set(STEP_FEATURES)
    if unknown:
        raise ValueError(f"Unknown features in excluded_features: {sorted(unknown)}")

    kept_features = [f for f in STEP_FEATURES if f not in excluded]
    n_kept = len(kept_features)

    all_inputs: list[np.ndarray] = []
    all_labels: list[float] = []
    all_session_ids: list[str] = []

    for sid in sorted(rows_by_session.keys()):
        rows = sorted(rows_by_session[sid], key=lambda r: r["step"])

        feat_buf = np.zeros((N_HISTORY, n_kept), dtype=np.float32)
        score_buf = np.zeros(N_HISTORY, dtype=np.float32)

        for row in rows:
            curr = np.array([float(row[f]) for f in kept_features], dtype=np.float32)

            if use_score_history:
                inp = np.concatenate([curr, feat_buf.flatten(), score_buf])
            else:
                inp = np.concatenate([curr, feat_buf.flatten()])
            all_inputs.append(inp)
            all_labels.append(float(row["label"]))
            all_session_ids.append(sid)

            # Shift ring buffer: roll down, overwrite position 0 with most recent
            feat_buf = np.roll(feat_buf, 1, axis=0)
            feat_buf[0] = curr
            if use_score_history:
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
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Split rows by session_id (90/10 split).

    Returns:
        (train_by_session, test_by_session) dicts mapping session_id → rows
    """
    rows_by_session: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        rows_by_session[r["session_id"]].append(r)

    session_ids = sorted(rows_by_session.keys())
    rng = random.Random(seed)
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
    use_score_history: bool = True,
    excluded_features: set[str] | None = None,
    output_dir: str = MODEL_DIR,
    seed: int = DEFAULT_SEED,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    excluded = excluded_features or set()
    n_kept = NUM_FEATURES - len(excluded)
    feat_block = n_kept * (1 + N_HISTORY)
    input_dim = feat_block + (N_HISTORY if use_score_history else 0)

    print(
        f"\nVariant: {'with score history' if use_score_history else 'NO score history'} "
        f"({input_dim}-dim, {n_kept} features"
        f"{', excluded: ' + ','.join(sorted(excluded)) if excluded else ''})"
    )
    print(f"Output directory: {output_dir}")

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

    train_by_session, test_by_session = session_split(all_rows, seed=seed)

    train_inputs, train_labels, _ = build_sequences(
        train_by_session,
        use_score_history=use_score_history,
        excluded_features=excluded,
    )
    test_inputs, test_labels, _ = build_sequences(
        test_by_session,
        use_score_history=use_score_history,
        excluded_features=excluded,
    )

    # Shuffle training sequences (across sessions — ring buffer already baked in)
    perm = np.random.permutation(len(train_inputs))
    train_inputs = train_inputs[perm]
    train_labels = train_labels[perm]

    # Normalize feature dims only. When score history is enabled the last
    # N_HISTORY positions are left in [0, 1] so inference values (continuous
    # sigmoid outputs) are not shifted by training-set stats derived from
    # trimodal {0, 0.5, 1} labels. When disabled, every dim is a feature.
    feat_dims = input_dim - N_HISTORY if use_score_history else input_dim
    mean = np.zeros(input_dim, dtype=np.float32)
    std = np.ones(input_dim, dtype=np.float32)
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

    model = StuckDetectorV5(input_dim=input_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params} params (input_dim={input_dim})")

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
        # Use distinct names to avoid clobbering tp/fp/fn/tn used in final_metrics
        p_t, r_t, f_t, _, fp_t, fn_t, _ = metrics_at(scores, binary_labels, t)
        print(f"  {t:>5.2f}  {p_t:>6.3f}  {r_t:>6.3f}  {f_t:>6.3f}  {fp_t:>6}  {fn_t:>6}")

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

    os.makedirs(output_dir, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "norm_mean": mean.tolist(),
            "norm_std": std.tolist(),
            "threshold": threshold,
            "metrics": final_metrics,
            "total_params": total_params,
        },
        os.path.join(output_dir, "stuck_checkpoint.pt"),
    )

    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy().tolist()
    weights["norm_mean"] = mean.tolist()
    weights["norm_std"] = std.tolist()
    with open(os.path.join(output_dir, "stuck_weights.json"), "w", encoding="utf-8") as f:
        json.dump(weights, f)

    kept_features = [f for f in STEP_FEATURES if f not in excluded]
    config = {
        "threshold": threshold,
        "model_stage": 5,
        "n_history": N_HISTORY,
        "num_features": n_kept,
        "input_dim": input_dim,
        "use_score_history": use_score_history,
        "excluded_features": sorted(excluded),
        "total_params": total_params,
        "metrics": final_metrics,
        "step_features": kept_features,
    }
    with open(os.path.join(output_dir, "stuck_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    size = os.path.getsize(os.path.join(output_dir, "stuck_weights.json"))
    print("\nSaved:")
    print(f"  {output_dir}/stuck_checkpoint.pt")
    print(f"  {output_dir}/stuck_weights.json ({size / 1024:.1f} KB)")
    print(f"  {output_dir}/stuck_config.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="training_manifest.json")
    parser.add_argument(
        "--no-score-history",
        action="store_true",
        help="Drop the 5 score history dims (input becomes 48-dim, no train/inference mismatch)",
    )
    parser.add_argument(
        "--exclude-feature",
        action="append",
        default=[],
        choices=STEP_FEATURES,
        metavar="NAME",
        help="Drop a feature from the input (repeatable). Used for ablation studies.",
    )
    parser.add_argument(
        "--variant-name",
        default=None,
        help="Subdirectory name under proxy/experiments/ablation/ for outputs. "
        "Auto-generated from --exclude-feature when not set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for split, init, and shuffling (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write checkpoint/weights/config. Overrides --variant-name. "
        "Defaults to proxy/, or an experiments subdir when --no-score-history or "
        "--exclude-feature are set.",
    )
    _args = parser.parse_args()
    excluded_set = set(_args.exclude_feature)

    out = _args.output_dir
    if out is None:
        if excluded_set:
            variant = _args.variant_name or "no_" + "_".join(
                sorted(f.replace("_", "") for f in excluded_set)
            )
            out = os.path.join(MODEL_DIR, "experiments", "ablation", variant)
        elif _args.no_score_history:
            out = os.path.join(MODEL_DIR, "experiments", "no_score_history")
        else:
            out = MODEL_DIR

    train(
        _args.manifest,
        use_score_history=not _args.no_score_history,
        excluded_features=excluded_set,
        output_dir=out,
        seed=_args.seed,
    )
