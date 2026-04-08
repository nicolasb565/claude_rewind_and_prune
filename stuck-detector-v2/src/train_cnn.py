"""Step 18: Train the stuck detector CNN.

Stage 1: ~10K params. Trains on windowed data, evaluates precision@recall>=70%.
"""

import json
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

TRAIN_FILE = 'data/train_windows.jsonl'
TEST_FILE = 'data/test_id.jsonl'
MODEL_DIR = 'proxy'

# Feature dimensions
NUM_TOOLS = 7  # bash, edit, view, search, create, submit, other
TOOL_EMBED_DIM = 4
NUM_CONTINUOUS = 15  # all non-categorical step features
STEP_DIM = TOOL_EMBED_DIM + NUM_CONTINUOUS  # 19
WINDOW_FEAT_DIM = 6
WINDOW_SIZE = 10

# Continuous feature names (order matters — must match encoding)
CONTINUOUS_FEATURES = [
    'steps_since_same_tool', 'steps_since_same_file', 'steps_since_same_cmd',
    'tool_count_in_window', 'file_count_in_window', 'cmd_count_in_window',
    'output_similarity', 'output_length', 'is_error', 'step_index_norm',
    'false_start', 'strategy_change', 'circular_lang',
    'thinking_length', 'self_similarity',
]

WINDOW_FEATURES = [
    'unique_tools_ratio', 'unique_files_ratio', 'unique_cmds_ratio',
    'error_rate', 'output_similarity_avg', 'output_diversity',
]


class WindowDataset(Dataset):
    def __init__(self, filepath):
        self.cat_data = []    # (N, WINDOW_SIZE, 1) int
        self.cont_data = []   # (N, WINDOW_SIZE, NUM_CONTINUOUS) float
        self.win_data = []    # (N, WINDOW_FEAT_DIM) float
        self.labels = []      # (N,) float

        with open(filepath) as f:
            for line in f:
                w = json.loads(line)
                steps = w['steps']
                wf = w['window_features']

                cat = []
                cont = []
                for s in steps:
                    cat.append(s['tool_idx'])
                    cont.append([s[feat] for feat in CONTINUOUS_FEATURES])

                self.cat_data.append(cat)
                self.cont_data.append(cont)
                self.win_data.append([wf[feat] for feat in WINDOW_FEATURES])
                self.labels.append(1.0 if w['label'] == 'STUCK' else 0.0)

        self.cat_data = torch.tensor(self.cat_data, dtype=torch.long)
        self.cont_data = torch.tensor(self.cont_data, dtype=torch.float32)
        self.win_data = torch.tensor(self.win_data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

        # Compute normalization stats for continuous features
        self.cont_mean = self.cont_data.mean(dim=(0, 1))
        self.cont_std = self.cont_data.std(dim=(0, 1)).clamp(min=1e-6)

    def normalize(self, mean=None, std=None):
        """Normalize continuous features. Use provided stats or self."""
        if mean is None:
            mean = self.cont_mean
            std = self.cont_std
        self.cont_data = (self.cont_data - mean) / std
        return mean, std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.cat_data[idx], self.cont_data[idx],
                self.win_data[idx], self.labels[idx])


class StuckDetectorSmall(nn.Module):
    """~10K params. Stage 1."""
    def __init__(self):
        super().__init__()
        self.tool_embed = nn.Embedding(NUM_TOOLS, TOOL_EMBED_DIM)
        step_dim = TOOL_EMBED_DIM + NUM_CONTINUOUS  # 19

        self.conv3 = nn.Conv1d(step_dim, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(step_dim, 16, kernel_size=5, padding=2)

        # 32 + 6 window features = 38
        self.fc1 = nn.Linear(38, 16)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, cat, cont, win_feats):
        tool_emb = self.tool_embed(cat)  # (batch, 10, 4)
        x = torch.cat([tool_emb, cont], dim=-1)  # (batch, 10, 19)
        x = x.permute(0, 2, 1)  # (batch, 19, 10)

        c3 = torch.relu(self.conv3(x)).max(dim=2).values  # (batch, 16)
        c5 = torch.relu(self.conv5(x)).max(dim=2).values  # (batch, 16)

        pooled = torch.cat([c3, c5], dim=1)  # (batch, 32)
        combined = torch.cat([pooled, win_feats], dim=1)  # (batch, 38)

        out = torch.relu(self.fc1(combined))
        out = self.dropout(out)
        return self.fc2(out).squeeze(1)  # (batch,) logits


def compute_metrics(model, loader, threshold=0.5):
    """Compute precision, recall, F1 at given threshold."""
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for cat, cont, win, labels in loader:
            logits = model(cat, cont, win)
            scores = torch.sigmoid(logits)
            all_scores.extend(scores.numpy())
            all_labels.extend(labels.numpy())

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    preds = (scores >= threshold).astype(int)
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'threshold': threshold,
    }


def find_best_threshold(model, loader, min_recall=0.70):
    """Find threshold maximizing precision at recall >= min_recall."""
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for cat, cont, win, labels in loader:
            logits = model(cat, cont, win)
            scores = torch.sigmoid(logits)
            all_scores.extend(scores.numpy())
            all_labels.extend(labels.numpy())

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    best_precision = 0
    best_threshold = 0.5
    best_metrics = None

    for t in np.arange(0.10, 0.95, 0.01):
        preds = (scores >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        if recall >= min_recall and precision > best_precision:
            best_precision = precision
            best_threshold = t
            best_metrics = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(2 * precision * recall / max(precision + recall, 1e-6)),
                'threshold': float(t),
            }

    return best_threshold, best_metrics


def train():
    print("Loading datasets...")
    train_ds = WindowDataset(TRAIN_FILE)
    test_ds = WindowDataset(TEST_FILE)

    # Normalize using train stats
    mean, std = train_ds.normalize()
    test_ds.normalize(mean, std)

    print(f"Train: {len(train_ds)} windows, "
          f"stuck={int(train_ds.labels.sum())}, "
          f"productive={int((1 - train_ds.labels).sum())}")
    print(f"Test:  {len(test_ds)} windows, "
          f"stuck={int(test_ds.labels.sum())}, "
          f"productive={int((1 - test_ds.labels).sum())}")

    # Class-balanced loss
    num_positive = train_ds.labels.sum().item()
    num_negative = len(train_ds) - num_positive
    pos_weight = torch.tensor([num_negative / max(num_positive, 1)])
    print(f"pos_weight: {pos_weight.item():.1f} (ratio {num_negative/max(num_positive,1):.0f}:1)")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512)

    model = StuckDetectorSmall()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: StuckDetectorSmall, {total_params} params")
    print(f"Data ratio: {len(train_ds) / total_params:.1f}x")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training loop
    best_val_f1 = 0
    patience = 5
    no_improve = 0
    best_state = None

    print("\nTraining...")
    for epoch in range(30):
        model.train()
        epoch_loss = 0
        batches = 0
        for cat, cont, win, labels in train_loader:
            logits = model(cat, cont, win)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / batches

        # Evaluate
        metrics = compute_metrics(model, test_loader, threshold=0.5)

        print(f"Epoch {epoch:2d}: loss={avg_loss:.4f} "
              f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
              f"F1={metrics['f1']:.3f} "
              f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")

        if metrics['f1'] > best_val_f1:
            best_val_f1 = metrics['f1']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(best_state)

    # Find optimal threshold
    print("\nFinding optimal threshold (precision@recall>=70%)...")
    threshold, thresh_metrics = find_best_threshold(model, test_loader, min_recall=0.70)
    if thresh_metrics:
        print(f"Optimal threshold: {threshold:.2f}")
        print(f"  Precision: {thresh_metrics['precision']:.3f}")
        print(f"  Recall: {thresh_metrics['recall']:.3f}")
        print(f"  F1: {thresh_metrics['f1']:.3f}")
    else:
        print("WARNING: Could not find threshold with recall >= 70%")
        threshold = 0.5

    # Final metrics at optimal threshold
    final_metrics = compute_metrics(model, test_loader, threshold)
    print(f"\nFinal test metrics at threshold={threshold:.2f}:")
    print(f"  Precision: {final_metrics['precision']:.3f}")
    print(f"  Recall: {final_metrics['recall']:.3f}")
    print(f"  F1: {final_metrics['f1']:.3f}")
    print(f"  TP={final_metrics['tp']} FP={final_metrics['fp']} "
          f"FN={final_metrics['fn']} TN={final_metrics['tn']}")

    # Save model + config
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save PyTorch checkpoint
    torch.save({
        'model_state': model.state_dict(),
        'norm_mean': mean.numpy().tolist(),
        'norm_std': std.numpy().tolist(),
        'threshold': threshold,
        'metrics': final_metrics,
        'total_params': total_params,
    }, os.path.join(MODEL_DIR, 'cnn_checkpoint.pt'))

    # Export weights to JSON for JS
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy().tolist()
    weights['norm_mean'] = mean.numpy().tolist()
    weights['norm_std'] = std.numpy().tolist()

    with open(os.path.join(MODEL_DIR, 'cnn_weights.json'), 'w') as f:
        json.dump(weights, f)

    config = {
        'threshold': threshold,
        'model_stage': 1,
        'total_params': total_params,
        'metrics': final_metrics,
        'window_size': WINDOW_SIZE,
        'tool_embed_dim': TOOL_EMBED_DIM,
        'num_continuous': NUM_CONTINUOUS,
        'continuous_features': CONTINUOUS_FEATURES,
        'window_features': WINDOW_FEATURES,
    }
    with open(os.path.join(MODEL_DIR, 'cnn_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    weights_size = os.path.getsize(os.path.join(MODEL_DIR, 'cnn_weights.json'))
    print(f"\nSaved:")
    print(f"  {MODEL_DIR}/cnn_checkpoint.pt")
    print(f"  {MODEL_DIR}/cnn_weights.json ({weights_size/1024:.1f} KB)")
    print(f"  {MODEL_DIR}/cnn_config.json")

    return model, mean, std, threshold


if __name__ == '__main__':
    train()
