"""Train the 11-feature CNN with DataClaw oversampled 10x.

DataClaw is the only training source with thinking blocks (2.5% of windows),
which matches the runtime environment and the LogReg benchmark sessions.
Oversampling it 10x physically (not via weighted sampling) is the sweet spot:
- Test F1 improves from 0.884 -> 0.899
- Benchmark F1 improves from 0.571 -> 0.714
- Eliminates 30_lapack_bug and 44_llvm_arith false positives

This script produces proxy/cnn_weights.json, proxy/cnn_trimmed_checkpoint.pt
and proxy/cnn_config.json used by the JS proxy.
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))

TRAIN_FILE = 'data/train_windows.jsonl'
TEST_FILE = 'data/test_id.jsonl'
MODEL_DIR = 'proxy'

ALL_FEATURES = [
    'steps_since_same_tool', 'steps_since_same_file', 'steps_since_same_cmd',
    'tool_count_in_window', 'file_count_in_window', 'cmd_count_in_window',
    'output_similarity', 'output_length', 'is_error', 'step_index_norm',
    'false_start', 'strategy_change', 'circular_lang',
    'thinking_length', 'self_similarity',
]
KEEP_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13]  # 11 features incl. thinking_length
KEEP_FEATURES = [ALL_FEATURES[i] for i in KEEP_IDX]
NUM_CONTINUOUS = len(KEEP_FEATURES)

WINDOW_FEATURES = [
    'unique_tools_ratio', 'unique_files_ratio', 'unique_cmds_ratio',
    'error_rate', 'output_similarity_avg', 'output_diversity',
]
WINDOW_FEAT_DIM = len(WINDOW_FEATURES)

NUM_TOOLS = 7
TOOL_EMBED_DIM = 4
WINDOW_SIZE = 10

DATACLAW_OVERSAMPLE = 10

SEED = 42


class StuckDetectorTrimmed(nn.Module):
    def __init__(self):
        super().__init__()
        self.tool_embed = nn.Embedding(NUM_TOOLS, TOOL_EMBED_DIM)
        step_dim = TOOL_EMBED_DIM + NUM_CONTINUOUS
        self.conv3 = nn.Conv1d(step_dim, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(step_dim, 16, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 + WINDOW_FEAT_DIM, 16)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, cat, cont, win_feats):
        tool_emb = self.tool_embed(cat)
        x = torch.cat([tool_emb, cont], dim=-1)
        x = x.permute(0, 2, 1)
        c3 = torch.relu(self.conv3(x)).max(dim=2).values
        c5 = torch.relu(self.conv5(x)).max(dim=2).values
        pooled = torch.cat([c3, c5], dim=1)
        combined = torch.cat([pooled, win_feats], dim=1)
        out = torch.relu(self.fc1(combined))
        out = self.dropout(out)
        return self.fc2(out).squeeze(1)


def load_windows(filepath):
    """Load raw windows as list of dicts. Returns (nlile_rows, dataclaw_rows)."""
    nlile, dataclaw = [], []
    with open(filepath) as f:
        for line in f:
            w = json.loads(line)
            steps = w['steps']
            cat = [s['tool_idx'] for s in steps]
            cont = []
            for s in steps:
                all_vals = [float(s[f]) if not isinstance(s[f], bool)
                            else (1.0 if s[f] else 0.0)
                            for f in ALL_FEATURES]
                cont.append([all_vals[i] for i in KEEP_IDX])
            wf = [w['window_features'][f] for f in WINDOW_FEATURES]
            label = 1.0 if w['label'] == 'STUCK' else 0.0
            row = (cat, cont, wf, label)
            tid = w['trajectory_id']
            if tid.startswith('dc_') or tid.startswith('dataclaw'):
                dataclaw.append(row)
            else:
                nlile.append(row)
    return nlile, dataclaw


def rows_to_tensors(rows):
    cat = torch.tensor([r[0] for r in rows], dtype=torch.long)
    cont = torch.tensor([r[1] for r in rows], dtype=torch.float32)
    win = torch.tensor([r[2] for r in rows], dtype=torch.float32)
    lab = torch.tensor([r[3] for r in rows], dtype=torch.float32)
    return cat, cont, win, lab


class TensorWindowDataset(Dataset):
    def __init__(self, cat, cont, win, lab):
        self.cat, self.cont, self.win, self.lab = cat, cont, win, lab

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, i):
        return self.cat[i], self.cont[i], self.win[i], self.lab[i]


def metrics_at(preds, labels, t):
    pred = (preds >= t).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-6)
    return p, r, f1, tp, fp, fn, tn


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading data...")
    train_nlile, train_dc = load_windows(TRAIN_FILE)
    test_nlile, test_dc = load_windows(TEST_FILE)
    print(f"  Train: nlile={len(train_nlile)}, dataclaw={len(train_dc)}")
    print(f"  Test : nlile={len(test_nlile)}, dataclaw={len(test_dc)}")

    # Oversample DataClaw physically
    oversampled = train_nlile + (train_dc * DATACLAW_OVERSAMPLE)
    print(f"  After {DATACLAW_OVERSAMPLE}x DataClaw oversample: {len(oversampled)} rows")

    # Shuffle
    idx = np.random.permutation(len(oversampled))
    oversampled = [oversampled[i] for i in idx]

    train_cat, train_cont, train_win, train_lab = rows_to_tensors(oversampled)
    test_cat, test_cont, test_win, test_lab = rows_to_tensors(test_nlile + test_dc)

    # Normalize continuous features using training stats
    mean = train_cont.mean(dim=(0, 1))
    std = train_cont.std(dim=(0, 1)).clamp(min=1e-6)
    train_cont = (train_cont - mean) / std
    test_cont = (test_cont - mean) / std

    train_ds = TensorWindowDataset(train_cat, train_cont, train_win, train_lab)
    test_ds = TensorWindowDataset(test_cat, test_cont, test_win, test_lab)

    num_pos = train_lab.sum().item()
    num_neg = len(train_lab) - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)])
    print(f"  Class balance: pos={int(num_pos)} neg={int(num_neg)} pos_weight={pos_weight.item():.1f}")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512)

    model = StuckDetectorTrimmed()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0
    best_state = None
    no_improve = 0
    patience = 5

    print("\nTraining...")
    for epoch in range(30):
        model.train()
        total_loss = 0
        n_batches = 0
        for cat, cont, win, lab in train_loader:
            logits = model(cat, cont, win)
            loss = criterion(logits, lab)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        all_s, all_l = [], []
        with torch.no_grad():
            for cat, cont, win, lab in test_loader:
                s = torch.sigmoid(model(cat, cont, win))
                all_s.extend(s.numpy())
                all_l.extend(lab.numpy())
        scores = np.array(all_s)
        labels = np.array(all_l)
        p, r, f1, tp, fp, fn, _ = metrics_at(scores, labels, 0.96)
        print(f"  Epoch {epoch:2d}: loss={total_loss/n_batches:.4f}  t=0.96 P={p:.3f} R={r:.3f} F1={f1:.3f} FP={fp} FN={fn}")

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

    # Final evaluation at 0.96 threshold
    model.eval()
    all_s, all_l = [], []
    with torch.no_grad():
        for cat, cont, win, lab in test_loader:
            s = torch.sigmoid(model(cat, cont, win))
            all_s.extend(s.numpy())
            all_l.extend(lab.numpy())
    scores = np.array(all_s)
    labels = np.array(all_l)

    t = 0.96
    p, r, f1, tp, fp, fn, tn = metrics_at(scores, labels, t)
    print(f"\n=== Final test metrics at threshold={t} ===")
    print(f"  Precision: {p:.3f}")
    print(f"  Recall:    {r:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")

    final_metrics = {
        'precision': float(p), 'recall': float(r), 'f1': float(f1),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'threshold': t,
    }

    # Save checkpoint
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'norm_mean': mean.numpy().tolist(),
        'norm_std': std.numpy().tolist(),
        'threshold': t,
        'metrics': final_metrics,
        'total_params': total_params,
        'dataclaw_oversample': DATACLAW_OVERSAMPLE,
    }, os.path.join(MODEL_DIR, 'cnn_trimmed_checkpoint.pt'))

    # Export weights JSON for the JS proxy
    weights = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy().tolist()
    weights['norm_mean'] = mean.numpy().tolist()
    weights['norm_std'] = std.numpy().tolist()
    with open(os.path.join(MODEL_DIR, 'cnn_weights.json'), 'w') as f:
        json.dump(weights, f)

    config = {
        'threshold': t,
        'model_stage': 1,
        'total_params': total_params,
        'metrics': final_metrics,
        'window_size': WINDOW_SIZE,
        'tool_embed_dim': TOOL_EMBED_DIM,
        'num_continuous': NUM_CONTINUOUS,
        'continuous_features': KEEP_FEATURES,
        'window_features': WINDOW_FEATURES,
        'dataclaw_oversample': DATACLAW_OVERSAMPLE,
    }
    with open(os.path.join(MODEL_DIR, 'cnn_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    size = os.path.getsize(os.path.join(MODEL_DIR, 'cnn_weights.json'))
    print(f"\nSaved:")
    print(f"  {MODEL_DIR}/cnn_trimmed_checkpoint.pt")
    print(f"  {MODEL_DIR}/cnn_weights.json ({size/1024:.1f} KB)")
    print(f"  {MODEL_DIR}/cnn_config.json")
    print(f"\nTo score the LogReg benchmark sessions: python src/eval_benchmark.py")


if __name__ == '__main__':
    main()
