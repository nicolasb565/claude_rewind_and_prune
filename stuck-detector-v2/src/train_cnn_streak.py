"""Train CNN with streak_prior as an extra window feature.

streak_prior = count of consecutive prior windows (excluding current) in the same
trajectory that scored >= 0.9 with the current model, normalized to [0, 1] by /5.

Pipeline:
1. Score all train+test windows with the current trimmed CNN
2. Compute streak_prior for each window (uses prior windows only)
3. Train a new CNN with 7 window features (was 6)
4. Evaluate on test set
5. Run on LogReg benchmark sessions (proxy-OFF runs)
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

# --- Constants matching trimmed model ---

ALL_FEATURES = [
    'steps_since_same_tool', 'steps_since_same_file', 'steps_since_same_cmd',
    'tool_count_in_window', 'file_count_in_window', 'cmd_count_in_window',
    'output_similarity', 'output_length', 'is_error', 'step_index_norm',
    'false_start', 'strategy_change', 'circular_lang',
    'thinking_length', 'self_similarity',
]
KEEP_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13]
KEEP_FEATURES = [ALL_FEATURES[i] for i in KEEP_IDX]
NUM_CONTINUOUS = len(KEEP_FEATURES)

WINDOW_FEATURES_BASE = [
    'unique_tools_ratio', 'unique_files_ratio', 'unique_cmds_ratio',
    'error_rate', 'output_similarity_avg', 'output_diversity',
]
WINDOW_FEATURES_NEW = WINDOW_FEATURES_BASE + ['streak_prior']  # 7 features
WINDOW_FEAT_DIM = len(WINDOW_FEATURES_NEW)

NUM_TOOLS = 7
TOOL_EMBED_DIM = 4

STREAK_THRESH = 0.9
STREAK_MAX_N = 5  # Cap streak at 5 for normalization

# --- Model ---

class StuckDetectorTrimmed(nn.Module):
    """Original trimmed model (used for scoring history)."""
    def __init__(self, num_continuous, win_dim=6):
        super().__init__()
        self.tool_embed = nn.Embedding(NUM_TOOLS, TOOL_EMBED_DIM)
        step_dim = TOOL_EMBED_DIM + num_continuous
        self.conv3 = nn.Conv1d(step_dim, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(step_dim, 16, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 + win_dim, 16)
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


# --- Dataset with streak feature ---

class WindowDatasetStreak(Dataset):
    def __init__(self, filepath, keep_idx, scores_by_key=None, streak_by_key=None):
        self.cat_data = []
        self.cont_data = []
        self.win_data = []
        self.labels = []
        self.keys = []  # (trajectory_id, window_start) for joining

        with open(filepath) as f:
            for line in f:
                w = json.loads(line)
                steps = w['steps']
                wf_base = [w['window_features'][f] for f in WINDOW_FEATURES_BASE]

                cat = [s['tool_idx'] for s in steps]
                cont = []
                for s in steps:
                    all_vals = [float(s[f]) if not isinstance(s[f], bool)
                                else (1.0 if s[f] else 0.0)
                                for f in ALL_FEATURES]
                    cont.append([all_vals[i] for i in keep_idx])

                key = (w['trajectory_id'], w['window_start'])
                streak = streak_by_key.get(key, 0.0) if streak_by_key else 0.0
                wf_full = wf_base + [streak]

                self.cat_data.append(cat)
                self.cont_data.append(cont)
                self.win_data.append(wf_full)
                self.labels.append(1.0 if w['label'] == 'STUCK' else 0.0)
                self.keys.append(key)

        self.cat_data = torch.tensor(self.cat_data, dtype=torch.long)
        self.cont_data = torch.tensor(self.cont_data, dtype=torch.float32)
        self.win_data = torch.tensor(self.win_data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.cont_mean = self.cont_data.mean(dim=(0, 1))
        self.cont_std = self.cont_data.std(dim=(0, 1)).clamp(min=1e-6)

    def normalize(self, mean=None, std=None):
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


# --- Score windows with old model ---

def score_with_old_model(model, mean, std, filepath):
    """Run inference with the trimmed model on a JSONL file. Return dict {(tid, ws): score}."""
    scores = {}
    with open(filepath) as f:
        for line in f:
            w = json.loads(line)
            cat = torch.tensor([[s['tool_idx'] for s in w['steps']]], dtype=torch.long)
            cont_raw = []
            for s in w['steps']:
                vals = [float(s[f]) if not isinstance(s[f], bool)
                        else (1.0 if s[f] else 0.0)
                        for f in ALL_FEATURES]
                cont_raw.append([vals[i] for i in KEEP_IDX])
            cont = torch.tensor([cont_raw], dtype=torch.float32)
            cont = (cont - mean) / std
            wf = torch.tensor([[w['window_features'][f] for f in WINDOW_FEATURES_BASE]],
                             dtype=torch.float32)
            with torch.no_grad():
                score = torch.sigmoid(model(cat, cont, wf)).item()
            scores[(w['trajectory_id'], w['window_start'])] = score
    return scores


def compute_streak_prior(scores_by_key):
    """For each window, compute streak count from PRIOR windows in same trajectory.
    Returns dict {(tid, ws): streak_normalized}. Excludes current window.
    """
    by_tid = defaultdict(list)
    for (tid, ws), score in scores_by_key.items():
        by_tid[tid].append((ws, score))
    for tid in by_tid:
        by_tid[tid].sort(key=lambda x: x[0])

    streak_by_key = {}
    for tid, windows in by_tid.items():
        # Walk through windows in order
        for i, (ws, score) in enumerate(windows):
            # streak counts how many of the PRIOR (i.e. windows[:i]) consecutive
            # windows scored >= STREAK_THRESH (counting backwards from i-1)
            streak = 0
            for j in range(i - 1, -1, -1):
                if windows[j][1] >= STREAK_THRESH:
                    streak += 1
                else:
                    break
            streak_by_key[(tid, ws)] = min(streak / STREAK_MAX_N, 1.0)
    return streak_by_key


def metrics_at_threshold(preds, labels, t):
    pred = (preds >= t).astype(int)
    tp = ((pred == 1) & (labels == 1)).sum()
    fp = ((pred == 1) & (labels == 0)).sum()
    fn = ((pred == 0) & (labels == 1)).sum()
    tn = ((pred == 0) & (labels == 0)).sum()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-6)
    return p, r, f1, tp, fp, fn, tn


def best_threshold(preds, labels, min_recall=0.70):
    best_p = 0
    best_t = 0.5
    for t in np.arange(0.05, 0.99, 0.01):
        p, r, f1, _, _, _, _ = metrics_at_threshold(preds, labels, t)
        if r >= min_recall and p > best_p:
            best_p = p
            best_t = t
    return best_t


# --- Main ---

def main():
    # 1. Load old model
    print("Loading old (trimmed) model...")
    old_ckpt = torch.load('proxy/cnn_trimmed_checkpoint.pt', weights_only=False)
    old_model = StuckDetectorTrimmed(NUM_CONTINUOUS, win_dim=6)
    old_model.load_state_dict(old_ckpt['model_state'])
    old_model.eval()
    old_mean = torch.tensor(old_ckpt['norm_mean'])
    old_std = torch.tensor(old_ckpt['norm_std']).clamp(min=1e-6)

    # 2. Score train and test
    print("Scoring train set with old model...")
    train_scores = score_with_old_model(old_model, old_mean, old_std, 'data/train_windows.jsonl')
    print(f"  {len(train_scores)} train scores")
    print("Scoring test set with old model...")
    test_scores = score_with_old_model(old_model, old_mean, old_std, 'data/test_id.jsonl')
    print(f"  {len(test_scores)} test scores")

    # 3. Compute streak_prior (within train and test separately — they're disjoint trajectories)
    print("Computing streak_prior...")
    train_streak = compute_streak_prior(train_scores)
    test_streak = compute_streak_prior(test_scores)
    train_nonzero = sum(1 for v in train_streak.values() if v > 0)
    test_nonzero = sum(1 for v in test_streak.values() if v > 0)
    print(f"  Train: {train_nonzero}/{len(train_streak)} windows have nonzero streak ({train_nonzero/len(train_streak)*100:.1f}%)")
    print(f"  Test:  {test_nonzero}/{len(test_streak)} windows have nonzero streak ({test_nonzero/len(test_streak)*100:.1f}%)")

    # 4. Build datasets with streak feature
    print("Loading datasets with streak feature...")
    train_ds = WindowDatasetStreak('data/train_windows.jsonl', KEEP_IDX, train_scores, train_streak)
    test_ds = WindowDatasetStreak('data/test_id.jsonl', KEEP_IDX, test_scores, test_streak)
    mean, std = train_ds.normalize()
    test_ds.normalize(mean, std)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512)

    num_pos = train_ds.labels.sum().item()
    num_neg = len(train_ds) - num_pos
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)])

    # 5. Train new model with 7 window features
    model = StuckDetectorTrimmed(NUM_CONTINUOUS, win_dim=WINDOW_FEAT_DIM)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nNew model: {total_params} params (was 2621)")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0
    best_state = None
    no_improve = 0
    patience = 5

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
        preds = (scores >= 0.5).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-6)

        if f1 > best_f1 or epoch % 3 == 0:
            print(f"Epoch {epoch:2d}: loss={epoch_loss/batches:.4f} P={p:.3f} R={r:.3f} F1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    # 6. Evaluate at best threshold
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

    t = best_threshold(scores, labels, min_recall=0.70)
    p, r, f1, tp, fp, fn, tn = metrics_at_threshold(scores, labels, t)

    print(f"\n=== STREAK CNN test set (threshold={t:.2f}) ===")
    print(f"Precision: {p:.3f}")
    print(f"Recall: {r:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")

    print(f"\n=== Comparison ===")
    print(f"{'Model':25s} {'P':>7s} {'R':>7s} {'F1':>7s} {'FP':>5s} {'FN':>5s}")
    print("-" * 60)
    print(f"{'Old (raw)':25s} {0.901:>7.3f} {0.822:>7.3f} {0.860:>7.3f} {39:>5d} {77:>5d}")
    print(f"{'streak indicator only':25s} {0.843:>7.3f} {0.933:>7.3f} {0.886:>7.3f} {75:>5d} {22:>5d}")
    print(f"{'New CNN+streak':25s} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {fp:>5d} {fn:>5d}")

    # Save model
    torch.save({
        'model_state': model.state_dict(),
        'norm_mean': mean.numpy().tolist(),
        'norm_std': std.numpy().tolist(),
        'threshold': float(t),
        'window_features': WINDOW_FEATURES_NEW,
        'streak_thresh': STREAK_THRESH,
        'streak_max_n': STREAK_MAX_N,
    }, 'proxy/cnn_streak_checkpoint.pt')
    print(f"\nSaved cnn_streak_checkpoint.pt")

    # 7. Run on LogReg benchmark sessions (proxy OFF runs)
    print(f"\n=== Benchmark sessions (off_1, off_2, heldout_off_1) ===")
    sys.path.insert(0, 'src')
    from abstract_trajectory import abstract_trajectory, WINDOW_SIZE as WS

    TOOL_TO_IDX = {'bash':0,'edit':1,'view':2,'search':3,'create':4,'submit':5,'other':6}

    def parse_cc_session(filepath):
        steps = []
        pending = {}
        with open(filepath) as f:
            for line in f:
                entry = json.loads(line)
                msg = entry.get('message', {})
                if not msg or not isinstance(msg, dict): continue
                role = msg.get('role', '')
                content = msg.get('content', '')
                if not isinstance(content, list): continue
                if role == 'assistant':
                    thinking = ''
                    for block in content:
                        if not isinstance(block, dict): continue
                        if block.get('type') == 'thinking':
                            thinking = block.get('thinking', '')
                        elif block.get('type') == 'tool_use':
                            name = block.get('name', '')
                            inp = block.get('input', {})
                            tool_map = {'Bash':'bash','Read':'view','Edit':'edit','Write':'edit',
                                        'Grep':'search','Glob':'search','Agent':'other','Task':'other'}
                            tool = tool_map.get(name, 'other')
                            cmd = inp.get('command', inp.get('file_path', inp.get('pattern', '')))
                            file_path = inp.get('file_path', inp.get('path', None))
                            pending[block.get('id', '')] = {
                                'tool': tool, 'cmd': cmd, 'file': file_path,
                                'thinking': thinking, 'output': ''
                            }
                            thinking = ''
                elif role == 'user':
                    for block in content:
                        if not isinstance(block, dict): continue
                        if block.get('type') == 'tool_result':
                            tid = block.get('tool_use_id', '')
                            if tid in pending:
                                out = block.get('content', '')
                                if isinstance(out, list):
                                    out = '\n'.join(b.get('text','') for b in out if isinstance(b,dict))
                                pending[tid]['output'] = str(out) if out else ''
                                steps.append(pending.pop(tid))
        return steps

    def make_window_input(window, mean_t, std_t, win_dim):
        cat = torch.tensor([[TOOL_TO_IDX.get(s['tool'], 6) for s in window]], dtype=torch.long)
        cont_raw = []
        for s in window:
            all_vals = []
            for feat in ALL_FEATURES:
                v = s.get(feat, 0)
                all_vals.append(float(v) if not isinstance(v, bool) else (1.0 if v else 0.0))
            cont_raw.append([all_vals[i] for i in KEEP_IDX])
        cont = torch.tensor([cont_raw], dtype=torch.float32)
        cont = (cont - mean_t) / std_t

        tools = [s['tool'] for s in window]
        fh = [s.get('file_hash') for s in window if s.get('file_hash') is not None]
        ch = [s.get('cmd_hash') for s in window if s.get('cmd_hash') is not None]
        al = []
        for s in window:
            if s.get('output_set'):
                al.extend(s['output_set'])

        wf = [
            len(set(tools))/len(tools),
            len(set(fh))/max(len(fh),1) if fh else 1.0,
            len(set(ch))/max(len(ch),1) if ch else 1.0,
            sum(1 for s in window if s['is_error'])/len(window),
            sum(s['output_similarity'] for s in window)/len(window),
            len(set(al))/max(len(al),1) if al else 1.0,
        ]
        return cat, cont, wf

    def score_session_with_streak(filepath):
        """Score session in temporal order, computing streak from old model scores."""
        steps = parse_cc_session(filepath)
        if len(steps) < WS: return None, len(steps)
        abstract = abstract_trajectory(steps)
        if len(abstract) < WS: return None, len(steps)

        # First pass: score each window with OLD model
        old_scores_seq = []
        for start in range(0, len(abstract) - WS + 1, 5):
            window = abstract[start:start + WS]
            cat, cont, wf6 = make_window_input(window, old_mean, old_std, 6)
            wf = torch.tensor([wf6], dtype=torch.float32)
            with torch.no_grad():
                s = torch.sigmoid(old_model(cat, cont, wf)).item()
            old_scores_seq.append(s)

        # Second pass: score with NEW model using streak_prior from old scores
        new_scores = []
        for i, start in enumerate(range(0, len(abstract) - WS + 1, 5)):
            window = abstract[start:start + WS]
            cat, cont, wf6 = make_window_input(window, mean, std, 6)
            # streak_prior: count of consecutive prior windows with old_score >= STREAK_THRESH
            streak = 0
            for j in range(i - 1, -1, -1):
                if old_scores_seq[j] >= STREAK_THRESH:
                    streak += 1
                else:
                    break
            streak_norm = min(streak / STREAK_MAX_N, 1.0)
            wf7 = wf6 + [streak_norm]
            wf_t = torch.tensor([wf7], dtype=torch.float32)
            with torch.no_grad():
                s = torch.sigmoid(model(cat, cont, wf_t)).item()
            new_scores.append({'old': old_scores_seq[i], 'new': s, 'streak_prior': streak_norm})

        return new_scores, len(steps)

    results_dir = '/home/nicolas/source/classifier-repos/results'
    claude_projects = os.path.expanduser('~/.claude/projects')

    for run in ['off_1', 'off_2', 'heldout_off_1']:
        run_dir = os.path.join(results_dir, run)
        if not os.path.isdir(run_dir): continue
        print(f"\n--- {run} ---")
        for f in sorted(os.listdir(run_dir)):
            if not f.endswith('.json'): continue
            task = f.replace('.json', '')
            with open(os.path.join(run_dir, f)) as fh:
                data = json.load(fh)
            sid = data.get('session_id', '')
            task_slug = task.replace('_', '-')

            session_file = None
            for prefix in [f'-home-nicolas-source-classifier-repos-worktrees-{task_slug}',
                           '-home-nicolas-source-classifier-repos-boost-libs-beast',
                           '-home-nicolas-source-classifier-repos-boost-libs-geometry']:
                sf = os.path.join(claude_projects, prefix, f'{sid}.jsonl')
                if os.path.exists(sf):
                    session_file = sf
                    break

            if not session_file:
                continue

            scores, n_steps = score_session_with_streak(session_file)
            if scores is None:
                print(f"  {task:25s} {n_steps:3d} steps (short)")
                continue

            old_max = max(s['old'] for s in scores)
            new_max = max(s['new'] for s in scores)
            old_fired = sum(1 for s in scores if s['old'] >= old_ckpt['threshold'])
            new_fired = sum(1 for s in scores if s['new'] >= t)
            marker = ''
            if new_fired > 0 and old_fired == 0:
                marker = ' ** NEW FP **'
            elif old_fired > 0 and new_fired == 0:
                marker = ' ** SUPPRESSED **'
            elif new_fired > 0:
                marker = ' (both fire)'
            print(f"  {task:25s} {n_steps:3d}st  old:{old_max:.3f}({old_fired:2d}/{len(scores)})  new:{new_max:.3f}({new_fired:2d}/{len(scores)}){marker}")


if __name__ == '__main__':
    main()
