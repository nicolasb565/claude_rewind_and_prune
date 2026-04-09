"""Analyze temporal aggregations of CNN scores.

Given the current model, score each test window, group by trajectory, and
evaluate different temporal indicators (moving averages, EMA, max-of-N, etc.)
to find the best confirmation strategy. Compare against the current 2/3 rule.

Output: a table of indicators ranked by F1 / precision-at-recall.
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

# --- Model definition (matches train_cnn.py trimmed model) ---

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
WINDOW_FEATURES = [
    'unique_tools_ratio', 'unique_files_ratio', 'unique_cmds_ratio',
    'error_rate', 'output_similarity_avg', 'output_diversity',
]

NUM_TOOLS = 7
TOOL_EMBED_DIM = 4
WINDOW_FEAT_DIM = 6


class StuckDetectorTrimmed(nn.Module):
    def __init__(self, num_continuous):
        super().__init__()
        self.tool_embed = nn.Embedding(NUM_TOOLS, TOOL_EMBED_DIM)
        step_dim = TOOL_EMBED_DIM + num_continuous
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


def score_test_set(model, mean, std):
    """Run inference on test set, return list of (trajectory_id, window_start, score, label)."""
    results = []
    with open('data/test_id.jsonl') as f:
        for line in f:
            w = json.loads(line)
            cat = torch.tensor([[s['tool_idx'] for s in w['steps']]], dtype=torch.long)
            cont_raw = []
            for s in w['steps']:
                vals = [float(s[f] if not isinstance(s[f], bool) else (1.0 if s[f] else 0.0))
                        for f in ALL_FEATURES]
                cont_raw.append([vals[i] for i in KEEP_IDX])
            cont = torch.tensor([cont_raw], dtype=torch.float32)
            cont = (cont - mean) / std
            wf = torch.tensor([[w['window_features'][f] for f in WINDOW_FEATURES]],
                             dtype=torch.float32)
            with torch.no_grad():
                score = torch.sigmoid(model(cat, cont, wf)).item()
            results.append({
                'tid': w['trajectory_id'],
                'wstart': w['window_start'],
                'score': score,
                'label': 1 if w['label'] == 'STUCK' else 0,
            })
    return results


def group_and_sort(results):
    """Group by trajectory_id, sort each by window_start."""
    by_tid = defaultdict(list)
    for r in results:
        by_tid[r['tid']].append(r)
    for tid in by_tid:
        by_tid[tid].sort(key=lambda x: x['wstart'])
    return by_tid


# --- Temporal indicators ---

def compute_indicator(scores_so_far, kind, **kwargs):
    """Given list of scores up to and including current window, return aggregated score."""
    if kind == 'raw':
        return scores_so_far[-1]
    if kind == 'mean':
        n = kwargs.get('n', 3)
        return float(np.mean(scores_so_far[-n:]))
    if kind == 'max':
        n = kwargs.get('n', 3)
        return float(np.max(scores_so_far[-n:]))
    if kind == 'min':
        n = kwargs.get('n', 3)
        return float(np.min(scores_so_far[-n:]))
    if kind == 'median':
        n = kwargs.get('n', 3)
        return float(np.median(scores_so_far[-n:]))
    if kind == 'trimmed_mean':
        # Drop highest and lowest, average rest
        n = kwargs.get('n', 5)
        recent = sorted(scores_so_far[-n:])
        if len(recent) <= 2:
            return float(np.mean(recent))
        trimmed = recent[1:-1]
        return float(np.mean(trimmed))
    if kind == 'quantile':
        n = kwargs.get('n', 5)
        q = kwargs.get('q', 0.75)
        return float(np.quantile(scores_so_far[-n:], q))
    if kind == 'weighted_mean':
        n = kwargs.get('n', 3)
        recent = scores_so_far[-n:]
        weights = np.linspace(0.5, 1.5, len(recent))
        return float(np.average(recent, weights=weights))
    if kind == 'ema':
        decay = kwargs.get('decay', 0.7)
        ema = scores_so_far[0]
        for s in scores_so_far[1:]:
            ema = decay * s + (1 - decay) * ema
        return float(ema)
    if kind == 'cusum':
        # Cumulative sum of (score - baseline), reset to 0 when negative
        baseline = kwargs.get('baseline', 0.5)
        cusum = 0.0
        for s in scores_so_far:
            cusum = max(0.0, cusum + (s - baseline))
        # Normalize to [0, 1] range using a soft cap
        cap = kwargs.get('cap', 5.0)
        return float(min(cusum / cap, 1.0))
    if kind == 'streak':
        # Count of consecutive recent scores above threshold (normalized)
        thresh = kwargs.get('thresh', 0.5)
        max_n = kwargs.get('max_n', 5)
        streak = 0
        for s in reversed(scores_so_far):
            if s >= thresh:
                streak += 1
            else:
                break
        return float(min(streak / max_n, 1.0))
    if kind == 'frac_above':
        # Fraction of last N windows above threshold (continuous version of M/N rule)
        n = kwargs.get('n', 5)
        thresh = kwargs.get('thresh', 0.5)
        recent = scores_so_far[-n:]
        return sum(1 for s in recent if s >= thresh) / max(len(recent), 1)
    if kind == 'ema_diff':
        # EMA(fast) - EMA(slow): positive means rising, negative means falling
        # Returns abs value mapped to [0, 1] via sigmoid-like scaling
        fast_decay = kwargs.get('fast_decay', 0.7)
        slow_decay = kwargs.get('slow_decay', 0.3)
        fast = scores_so_far[0]
        slow = scores_so_far[0]
        for s in scores_so_far[1:]:
            fast = fast_decay * s + (1 - fast_decay) * fast
            slow = slow_decay * s + (1 - slow_decay) * slow
        # If fast > slow, scores are rising. We just return fast since rising = stuck
        return float(fast)
    if kind == 'slope':
        # Linear regression slope on last N scores, mapped to [0, 1]
        n = kwargs.get('n', 5)
        recent = scores_so_far[-n:]
        if len(recent) < 2:
            return float(recent[0]) if recent else 0.0
        x = np.arange(len(recent))
        slope, intercept = np.polyfit(x, recent, 1)
        # Predicted value at current point + small slope boost
        return float(np.clip(intercept + slope * (len(recent) - 1) + slope * 2, 0, 1))
    if kind == 'max_raw_ema':
        # max(current_raw, EMA_smoothed)
        decay = kwargs.get('decay', 0.7)
        ema = scores_so_far[0]
        for s in scores_so_far[1:]:
            ema = decay * s + (1 - decay) * ema
        return float(max(scores_so_far[-1], ema))
    if kind == 'product_raw_ema':
        # Both raw and EMA must be high (sqrt to keep range)
        decay = kwargs.get('decay', 0.5)
        ema = scores_so_far[0]
        for s in scores_so_far[1:]:
            ema = decay * s + (1 - decay) * ema
        return float(np.sqrt(scores_so_far[-1] * ema))
    if kind == 'count_above':
        n = kwargs.get('n', 3)
        thresh = kwargs.get('thresh', 0.94)
        recent = scores_so_far[-n:]
        if len(recent) < n:
            return 0.0
        return sum(1 for s in recent if s >= thresh) / n
    raise ValueError(f"unknown kind: {kind}")


def sanity_check():
    """Verify each indicator on hand-computed test cases."""
    print("=== SANITY CHECK ===")
    failures = []

    def check(name, actual, expected, tol=1e-6):
        ok = abs(actual - expected) < tol
        marker = "OK" if ok else "FAIL"
        print(f"  [{marker}] {name}: got {actual:.6f}, expected {expected:.6f}")
        if not ok:
            failures.append(name)

    # Raw
    check("raw([0.5])", compute_indicator([0.5], 'raw'), 0.5)
    check("raw([0.1, 0.9])", compute_indicator([0.1, 0.9], 'raw'), 0.9)

    # Mean
    check("mean_3([0.2, 0.4, 0.6])",
          compute_indicator([0.2, 0.4, 0.6], 'mean', n=3),
          (0.2 + 0.4 + 0.6) / 3)
    check("mean_3([0.1, 0.2, 0.3, 0.4, 0.5]) takes last 3",
          compute_indicator([0.1, 0.2, 0.3, 0.4, 0.5], 'mean', n=3),
          (0.3 + 0.4 + 0.5) / 3)

    # Max
    check("max_3([0.2, 0.9, 0.5])",
          compute_indicator([0.2, 0.9, 0.5], 'max', n=3),
          0.9)
    check("max_2 with shorter than n",
          compute_indicator([0.7], 'max', n=2),
          0.7)

    # Median
    check("median_3([0.1, 0.5, 0.9])",
          compute_indicator([0.1, 0.5, 0.9], 'median', n=3),
          0.5)
    check("median_5 with one outlier",
          compute_indicator([0.1, 0.2, 0.3, 0.4, 0.99], 'median', n=5),
          0.3)

    # Trimmed mean
    check("trimmed_mean_5([0.0, 0.4, 0.5, 0.6, 1.0]) drops 0 and 1",
          compute_indicator([0.0, 0.4, 0.5, 0.6, 1.0], 'trimmed_mean', n=5),
          (0.4 + 0.5 + 0.6) / 3)

    # Quantile
    check("quantile_5 q=0.75 of [0.1,0.2,0.3,0.4,0.5]",
          compute_indicator([0.1, 0.2, 0.3, 0.4, 0.5], 'quantile', n=5, q=0.75),
          0.4)  # 75th percentile of these 5 = 0.4

    # Weighted mean (linear 0.5..1.5 weights)
    # For n=3: weights [0.5, 1.0, 1.5], sum=3.0
    # values [0.2, 0.4, 0.6] -> (0.5*0.2 + 1.0*0.4 + 1.5*0.6) / 3 = (0.1+0.4+0.9)/3 = 0.4667
    check("weighted_mean_3([0.2, 0.4, 0.6])",
          compute_indicator([0.2, 0.4, 0.6], 'weighted_mean', n=3),
          (0.5*0.2 + 1.0*0.4 + 1.5*0.6) / (0.5 + 1.0 + 1.5))

    # EMA
    # decay=0.5, [0.2, 0.6, 0.4]
    # ema starts at 0.2
    # ema = 0.5*0.6 + 0.5*0.2 = 0.4
    # ema = 0.5*0.4 + 0.5*0.4 = 0.4
    check("ema decay=0.5 on [0.2, 0.6, 0.4]",
          compute_indicator([0.2, 0.6, 0.4], 'ema', decay=0.5),
          0.4)
    # decay=0.9, [0.2, 0.6, 0.4]
    # ema = 0.2
    # ema = 0.9*0.6 + 0.1*0.2 = 0.54 + 0.02 = 0.56
    # ema = 0.9*0.4 + 0.1*0.56 = 0.36 + 0.056 = 0.416
    check("ema decay=0.9 on [0.2, 0.6, 0.4]",
          compute_indicator([0.2, 0.6, 0.4], 'ema', decay=0.9),
          0.416)

    # CUSUM
    # baseline=0.5, [0.6, 0.7, 0.4, 0.6]
    # cusum: 0+(0.6-0.5)=0.1
    # cusum: 0.1+(0.7-0.5)=0.3
    # cusum: max(0, 0.3+(0.4-0.5))=0.2
    # cusum: max(0, 0.2+(0.6-0.5))=0.3
    # normalized: 0.3 / 5.0 = 0.06
    check("cusum baseline=0.5 on [0.6, 0.7, 0.4, 0.6]",
          compute_indicator([0.6, 0.7, 0.4, 0.6], 'cusum', baseline=0.5, cap=5.0),
          0.3 / 5.0)
    # cusum reset
    # baseline=0.5, [0.6, 0.3, 0.7]
    # cusum: 0+0.1=0.1
    # cusum: max(0, 0.1-0.2)=0
    # cusum: max(0, 0+0.2)=0.2
    check("cusum reset on [0.6, 0.3, 0.7]",
          compute_indicator([0.6, 0.3, 0.7], 'cusum', baseline=0.5, cap=5.0),
          0.2 / 5.0)

    # Streak
    # thresh=0.5, max_n=5, [0.6, 0.4, 0.7, 0.8, 0.9]
    # reversed: 0.9 (>=0.5, streak=1), 0.8 (>=0.5, streak=2), 0.7 (>=0.5, streak=3), 0.4 (break)
    # streak/max_n = 3/5 = 0.6
    check("streak thresh=0.5 on [0.6, 0.4, 0.7, 0.8, 0.9]",
          compute_indicator([0.6, 0.4, 0.7, 0.8, 0.9], 'streak', thresh=0.5, max_n=5),
          3 / 5)
    # All below
    check("streak all below thresh",
          compute_indicator([0.1, 0.2, 0.3], 'streak', thresh=0.5, max_n=5),
          0.0)

    # frac_above
    # thresh=0.5, n=5, [0.1, 0.6, 0.4, 0.8, 0.9]
    # 3 of 5 above 0.5 -> 0.6
    check("frac_above thresh=0.5 on [0.1, 0.6, 0.4, 0.8, 0.9]",
          compute_indicator([0.1, 0.6, 0.4, 0.8, 0.9], 'frac_above', n=5, thresh=0.5),
          3 / 5)

    # max_raw_ema
    # decay=0.5, scores=[0.2, 0.6, 0.4]
    # EMA = 0.4 (from earlier check)
    # raw = 0.4
    # max(0.4, 0.4) = 0.4
    check("max_raw_ema decay=0.5 on [0.2, 0.6, 0.4]",
          compute_indicator([0.2, 0.6, 0.4], 'max_raw_ema', decay=0.5),
          0.4)
    # raw=0.9, EMA after step: 0.5*0.9 + 0.5*0.4 = 0.65; max(0.9, 0.65) = 0.9
    check("max_raw_ema decay=0.5 on [0.2, 0.6, 0.4, 0.9]",
          compute_indicator([0.2, 0.6, 0.4, 0.9], 'max_raw_ema', decay=0.5),
          0.9)

    # product_raw_ema
    # decay=0.5, [0.2, 0.6, 0.4]
    # EMA = 0.4
    # raw = 0.4
    # sqrt(0.4 * 0.4) = 0.4
    check("product_raw_ema decay=0.5 on [0.2, 0.6, 0.4]",
          compute_indicator([0.2, 0.6, 0.4], 'product_raw_ema', decay=0.5),
          0.4)
    # raw=0.9, ema=0.65 -> sqrt(0.585) = 0.7649
    check("product_raw_ema decay=0.5 on [0.2, 0.6, 0.4, 0.9]",
          compute_indicator([0.2, 0.6, 0.4, 0.9], 'product_raw_ema', decay=0.5),
          np.sqrt(0.9 * 0.65))

    # count_above
    # n=3, thresh=0.5, [0.1, 0.6, 0.7] -> 2/3
    check("count_above n=3 thresh=0.5 on [0.1, 0.6, 0.7]",
          compute_indicator([0.1, 0.6, 0.7], 'count_above', n=3, thresh=0.5),
          2 / 3)
    # n=3 with shorter input -> 0.0
    check("count_above n=3 with shorter input",
          compute_indicator([0.6, 0.7], 'count_above', n=3, thresh=0.5),
          0.0)

    if failures:
        print(f"\n{len(failures)} sanity check FAILURES")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"\nAll {26} sanity checks passed.\n")


def evaluate(by_tid, indicator_kind, **kwargs):
    """Apply indicator to each window in order, return all (predicted_score, label) pairs."""
    preds = []
    labels = []
    for tid, windows in by_tid.items():
        scores_so_far = []
        for w in windows:
            scores_so_far.append(w['score'])
            agg = compute_indicator(scores_so_far, indicator_kind, **kwargs)
            preds.append(agg)
            labels.append(w['label'])
    return np.array(preds), np.array(labels)


def metrics_at_threshold(preds, labels, t):
    pred_binary = (preds >= t).astype(int)
    tp = ((pred_binary == 1) & (labels == 1)).sum()
    fp = ((pred_binary == 1) & (labels == 0)).sum()
    fn = ((pred_binary == 0) & (labels == 1)).sum()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-6)
    return p, r, f1, tp, fp, fn


def best_threshold(preds, labels, min_recall=0.70):
    """Find threshold with highest precision at recall >= min_recall."""
    best_p = 0
    best_t = 0.5
    best_f1 = 0
    for t in np.arange(0.05, 0.99, 0.01):
        p, r, f1, _, _, _ = metrics_at_threshold(preds, labels, t)
        if r >= min_recall and p > best_p:
            best_p = p
            best_t = t
            best_f1 = f1
    return best_t, best_p, best_f1


def main():
    sanity_check()

    print("Loading model...")
    ckpt = torch.load('proxy/cnn_trimmed_checkpoint.pt', weights_only=False)
    model = StuckDetectorTrimmed(NUM_CONTINUOUS)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    mean = torch.tensor(ckpt['norm_mean'])
    std = torch.tensor(ckpt['norm_std']).clamp(min=1e-6)

    print("Scoring test set...")
    results = score_test_set(model, mean, std)
    by_tid = group_and_sort(results)
    print(f"  {len(results)} windows across {len(by_tid)} trajectories")

    # All indicators to test
    indicators = [
        ('raw (current)',           'raw',           {}),
        ('mean_2',                  'mean',          {'n': 2}),
        ('mean_3',                  'mean',          {'n': 3}),
        ('mean_5',                  'mean',          {'n': 5}),
        ('max_2',                   'max',           {'n': 2}),
        ('max_3',                   'max',           {'n': 3}),
        ('max_5',                   'max',           {'n': 5}),
        ('min_2',                   'min',           {'n': 2}),
        ('min_3',                   'min',           {'n': 3}),
        ('median_3',                'median',        {'n': 3}),
        ('median_5',                'median',        {'n': 5}),
        ('trimmed_mean_5',          'trimmed_mean',  {'n': 5}),
        ('quantile_5_q75',          'quantile',      {'n': 5, 'q': 0.75}),
        ('quantile_5_q90',          'quantile',      {'n': 5, 'q': 0.90}),
        ('weighted_mean_3',         'weighted_mean', {'n': 3}),
        ('weighted_mean_5',         'weighted_mean', {'n': 5}),
        ('ema_decay_0.3',           'ema',           {'decay': 0.3}),
        ('ema_decay_0.5',           'ema',           {'decay': 0.5}),
        ('ema_decay_0.7',           'ema',           {'decay': 0.7}),
        ('ema_decay_0.9',           'ema',           {'decay': 0.9}),
        ('cusum_baseline_0.5',      'cusum',         {'baseline': 0.5, 'cap': 5.0}),
        ('cusum_baseline_0.7',      'cusum',         {'baseline': 0.7, 'cap': 3.0}),
        ('cusum_baseline_0.9',      'cusum',         {'baseline': 0.9, 'cap': 2.0}),
        ('streak_thresh_0.5',       'streak',        {'thresh': 0.5, 'max_n': 5}),
        ('streak_thresh_0.7',       'streak',        {'thresh': 0.7, 'max_n': 5}),
        ('streak_thresh_0.9',       'streak',        {'thresh': 0.9, 'max_n': 5}),
        ('frac_above_5_t0.5',       'frac_above',    {'n': 5, 'thresh': 0.5}),
        ('frac_above_5_t0.7',       'frac_above',    {'n': 5, 'thresh': 0.7}),
        ('frac_above_5_t0.9',       'frac_above',    {'n': 5, 'thresh': 0.9}),
        ('slope_5',                 'slope',         {'n': 5}),
        ('max_raw_ema_0.7',         'max_raw_ema',   {'decay': 0.7}),
        ('product_raw_ema_0.5',     'product_raw_ema', {'decay': 0.5}),
        ('product_raw_ema_0.7',     'product_raw_ema', {'decay': 0.7}),
    ]

    print("\n=== Continuous indicators (best threshold @ recall>=70%) ===")
    print(f"{'Indicator':25s} {'Thresh':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
    print("-" * 75)
    rows = []
    for name, kind, kwargs in indicators:
        preds, labels = evaluate(by_tid, kind, **kwargs)
        t, p, f1 = best_threshold(preds, labels, min_recall=0.70)
        p_, r_, f1_, tp, fp, fn = metrics_at_threshold(preds, labels, t)
        rows.append((name, t, p_, r_, f1_, tp, fp, fn))
        marker = ' *' if name == 'raw (current)' else ''
        print(f"{name:25s} {t:7.2f} {p_:7.3f} {r_:7.3f} {f1_:7.3f} {tp:5d} {fp:5d} {fn:5d}{marker}")

    # Sort by F1
    rows.sort(key=lambda x: -x[4])
    print(f"\nTop 5 by F1:")
    for name, t, p, r, f1, tp, fp, fn in rows[:5]:
        print(f"  {name}: F1={f1:.3f} P={p:.3f} R={r:.3f}")

    # Discrete count_above indicator (the current 2/3-style rule)
    print("\n=== Discrete confirmation rules (M of N at threshold T) ===")
    print(f"{'Rule':25s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
    print("-" * 65)
    raw_thresh = 0.94
    for n, m in [(2, 1), (2, 2), (3, 2), (3, 3), (5, 3), (5, 4)]:
        preds, labels = evaluate(by_tid, 'count_above',
                                 n=n, thresh=raw_thresh)
        # Predict positive if at least m/n above threshold
        pred_binary = (preds >= m / n - 1e-9).astype(int)
        tp = ((pred_binary == 1) & (labels == 1)).sum()
        fp = ((pred_binary == 1) & (labels == 0)).sum()
        fn = ((pred_binary == 0) & (labels == 1)).sum()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-6)
        marker = '  <- current rule (2/3)' if (n, m) == (3, 2) else ''
        print(f"{m}/{n} >= {raw_thresh}{'':14s} {p:7.3f} {r:7.3f} {f1:7.3f} {tp:5d} {fp:5d} {fn:5d}{marker}")


if __name__ == '__main__':
    main()
