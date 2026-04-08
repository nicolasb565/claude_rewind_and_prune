"""Step 19: Feature ablation study.

Train 16 models, each with one feature removed. Compare precision@recall>=70%.
"""

import json
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from train_cnn import (
    WindowDataset, CONTINUOUS_FEATURES, WINDOW_FEATURES,
    TOOL_EMBED_DIM, NUM_TOOLS, WINDOW_SIZE, WINDOW_FEAT_DIM,
    TRAIN_FILE, TEST_FILE
)


class StuckDetectorAblation(nn.Module):
    """Flexible model for ablation — accepts variable step dim."""
    def __init__(self, step_dim, use_tool_embed=True):
        super().__init__()
        self.use_tool_embed = use_tool_embed
        if use_tool_embed:
            self.tool_embed = nn.Embedding(NUM_TOOLS, TOOL_EMBED_DIM)
            total_step_dim = TOOL_EMBED_DIM + step_dim
        else:
            total_step_dim = step_dim

        self.conv3 = nn.Conv1d(total_step_dim, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(total_step_dim, 16, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 + WINDOW_FEAT_DIM, 16)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, cat, cont, win_feats):
        if self.use_tool_embed:
            tool_emb = self.tool_embed(cat)
            x = torch.cat([tool_emb, cont], dim=-1)
        else:
            x = cont
        x = x.permute(0, 2, 1)

        c3 = torch.relu(self.conv3(x)).max(dim=2).values
        c5 = torch.relu(self.conv5(x)).max(dim=2).values
        pooled = torch.cat([c3, c5], dim=1)
        combined = torch.cat([pooled, win_feats], dim=1)

        out = torch.relu(self.fc1(combined))
        out = self.dropout(out)
        return self.fc2(out).squeeze(1)


def train_with_features(train_ds, test_ds, feature_mask, use_tool_embed=True,
                        epochs=20, lr=1e-3):
    """Train a model with only selected continuous features."""
    # Apply feature mask to continuous data
    train_cont = train_ds.cont_data[:, :, feature_mask]
    test_cont = test_ds.cont_data[:, :, feature_mask]
    step_dim = feature_mask.sum().item()

    # Normalize
    mean = train_cont.mean(dim=(0, 1))
    std = train_cont.std(dim=(0, 1)).clamp(min=1e-6)
    train_cont = (train_cont - mean) / std
    test_cont = (test_cont - mean) / std

    # Create loaders with masked data
    train_data = list(zip(
        train_ds.cat_data, train_cont, train_ds.win_data, train_ds.labels))
    test_data = list(zip(
        test_ds.cat_data, test_cont, test_ds.win_data, test_ds.labels))

    class SimpleDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    train_loader = DataLoader(SimpleDataset(train_data), batch_size=256, shuffle=True)
    test_loader = DataLoader(SimpleDataset(test_data), batch_size=512)

    # Model
    model = StuckDetectorAblation(step_dim, use_tool_embed)
    num_positive = train_ds.labels.sum().item()
    num_negative = len(train_ds) - num_positive
    pos_weight = torch.tensor([num_negative / max(num_positive, 1)])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for cat, cont, win, labels in train_loader:
            logits = model(cat, cont, win)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Quick eval
        model.eval()
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for cat, cont, win, labels in test_loader:
                scores = torch.sigmoid(model(cat, cont, win))
                all_scores.extend(scores.numpy())
                all_labels.extend(labels.numpy())

        scores = np.array(all_scores)
        labels = np.array(all_labels)

        # Find best threshold at recall >= 0.7
        best_prec = 0
        best_t = 0.5
        for t in np.arange(0.1, 0.95, 0.02):
            preds = (scores >= t).astype(int)
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-6)

            if rec >= 0.7 and prec > best_prec:
                best_prec = prec
                best_t = t

        # F1 at default threshold for early stopping
        preds = (scores >= 0.5).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 4:
                break

    return best_prec, best_t, best_f1


def run_ablation():
    print("Loading datasets...")
    train_ds = WindowDataset(TRAIN_FILE)
    test_ds = WindowDataset(TEST_FILE)

    num_features = len(CONTINUOUS_FEATURES)
    all_mask = torch.ones(num_features, dtype=torch.bool)

    # Baseline: all features
    print("\nBaseline (all features)...")
    base_prec, base_t, base_f1 = train_with_features(
        train_ds, test_ds, all_mask, use_tool_embed=True)
    print(f"  Baseline: precision@recall>=70% = {base_prec:.3f}, "
          f"threshold={base_t:.2f}, f1={base_f1:.3f}")

    # Ablation: remove one feature at a time
    results = [('all_features', base_prec, base_f1, 0.0)]

    for i, feat in enumerate(CONTINUOUS_FEATURES):
        mask = all_mask.clone()
        mask[i] = False
        prec, t, f1 = train_with_features(
            train_ds, test_ds, mask, use_tool_embed=True)
        delta = prec - base_prec
        results.append((f'without_{feat}', prec, f1, delta))
        status = "WORSE" if delta < -0.02 else "BETTER" if delta > 0.02 else "same"
        print(f"  Without {feat:25s}: prec={prec:.3f} (delta={delta:+.3f}) {status}")

    # Also test without tool embedding
    print("\n  Without tool_name (no embedding)...")
    prec, t, f1 = train_with_features(
        train_ds, test_ds, all_mask, use_tool_embed=False)
    delta = prec - base_prec
    results.append(('without_tool_embed', prec, f1, delta))
    print(f"  Without tool_embed:               prec={prec:.3f} (delta={delta:+.3f})")

    # Summary
    print("\n=== Feature Importance Ranking ===")
    print(f"{'Feature':30s} {'Precision':>10s} {'Delta':>8s} {'Impact':>8s}")
    print("-" * 58)
    for name, prec, f1, delta in sorted(results, key=lambda x: x[3]):
        impact = "CRITICAL" if delta < -0.03 else "HELPFUL" if delta < -0.01 else "NOISE" if delta > 0.01 else "neutral"
        print(f"{name:30s} {prec:10.3f} {delta:+8.3f} {impact:>8s}")

    # Identify features to remove (NOISE — removing them improves precision)
    noise_features = [name for name, prec, f1, delta in results
                      if delta > 0.02 and name != 'all_features']
    if noise_features:
        print(f"\nRecommend removing: {noise_features}")
    else:
        print(f"\nNo features clearly harmful — keep all.")

    # Save results
    with open('data/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to data/ablation_results.json")


if __name__ == '__main__':
    run_ablation()
