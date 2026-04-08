"""Steps 15-16: Window trajectories into 10-step chunks and split train/test.

Reads curated_pool.jsonl (with parsed_steps) and sonnet_result_*.jsonl (labels).
Outputs train_windows.jsonl and test_id.jsonl split by trajectory_id.
"""

import json
import sys
import os
import random
import math
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from abstract_trajectory import (
    abstract_trajectory, compute_window_features, precompute_review_counts,
    WINDOW_SIZE, STRIDE, TOOL_TO_IDX
)

random.seed(42)

POOL_FILE = 'data/curated_pool.jsonl'
REVIEWS_DIR = 'data/reviews'
TRAIN_FILE = 'data/train_windows.jsonl'
TEST_FILE = 'data/test_id.jsonl'
TEST_SPLIT = 0.2


def load_sonnet_labels():
    """Load all Sonnet labels into a dict by trajectory id."""
    labels = {}
    import glob
    for f in sorted(glob.glob(os.path.join(REVIEWS_DIR, 'sonnet_result_*.jsonl'))):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                labels[r['id']] = r['label']
    return labels


def create_windows(abstract_seq, label, trajectory_id):
    """Slide window across abstract sequence, assign label to each window."""
    windows = []
    if len(abstract_seq) < WINDOW_SIZE:
        return windows

    for start in range(0, len(abstract_seq) - WINDOW_SIZE + 1, STRIDE):
        window_steps = abstract_seq[start:start + WINDOW_SIZE]

        # Per-step features for CNN input
        step_features = []
        for s in window_steps:
            step_features.append({
                'tool_idx': s['tool_idx'],
                'steps_since_same_tool': s['steps_since_same_tool'],
                'steps_since_same_file': s['steps_since_same_file'],
                'steps_since_same_cmd': s['steps_since_same_cmd'],
                'tool_count_in_window': s['tool_count_in_window'],
                'file_count_in_window': s['file_count_in_window'],
                'cmd_count_in_window': s['cmd_count_in_window'],
                'output_similarity': s['output_similarity'],
                'output_length': s['output_length'],
                'is_error': 1.0 if s['is_error'] else 0.0,
                'step_index_norm': s['step_index_norm'],
                'false_start': 1.0 if s['false_start'] else 0.0,
                'strategy_change': 1.0 if s['strategy_change'] else 0.0,
                'circular_lang': 1.0 if s['circular_lang'] else 0.0,
                'thinking_length': s['thinking_length'],
                'self_similarity': s['self_similarity'],
            })

        # Window-level features
        window_feats = compute_window_features(window_steps)

        windows.append({
            'trajectory_id': trajectory_id,
            'window_start': start,
            'label': label,
            'steps': step_features,
            'window_features': window_feats,
        })

    return windows


def process():
    # Load labels
    print("Loading Sonnet labels...")
    labels = load_sonnet_labels()
    print(f"  {len(labels)} labels loaded")
    label_counts = Counter(labels.values())
    print(f"  Distribution: {dict(label_counts)}")

    # Process pool and create windows
    print("\nCreating windows...")
    all_windows = []
    trajectories_by_id = {}
    skipped_no_label = 0
    skipped_unclear = 0
    skipped_short = 0

    with open(POOL_FILE) as f:
        for line_num, line in enumerate(f):
            rec = json.loads(line)
            tid = rec['trajectory_id']
            parsed_steps = rec.get('parsed_steps')

            if not parsed_steps or len(parsed_steps) < WINDOW_SIZE:
                skipped_short += 1
                continue

            if tid not in labels:
                skipped_no_label += 1
                continue

            label = labels[tid]
            if label == 'UNCLEAR':
                skipped_unclear += 1
                continue

            # Abstract
            abstract = abstract_trajectory(parsed_steps)
            if len(abstract) < WINDOW_SIZE:
                skipped_short += 1
                continue

            # Create windows — all get the trajectory-level label
            windows = create_windows(abstract, label, tid)
            all_windows.extend(windows)

            # Track for splitting
            trajectories_by_id[tid] = label

            if (line_num + 1) % 2000 == 0:
                print(f"  Processed {line_num + 1}... ({len(all_windows)} windows)")

    print(f"\nWindowing complete:")
    print(f"  Total windows: {len(all_windows)}")
    print(f"  Skipped (no label): {skipped_no_label}")
    print(f"  Skipped (UNCLEAR): {skipped_unclear}")
    print(f"  Skipped (too short): {skipped_short}")
    print(f"  Trajectories used: {len(trajectories_by_id)}")

    window_labels = Counter(w['label'] for w in all_windows)
    print(f"  Window labels: {dict(window_labels)}")

    # Split by trajectory_id (80/20)
    print("\nSplitting train/test by trajectory_id...")
    traj_ids = list(trajectories_by_id.keys())
    random.shuffle(traj_ids)
    split_point = int(len(traj_ids) * (1 - TEST_SPLIT))
    train_ids = set(traj_ids[:split_point])
    test_ids = set(traj_ids[split_point:])

    assert len(train_ids & test_ids) == 0, "Train/test overlap!"

    train_windows = [w for w in all_windows if w['trajectory_id'] in train_ids]
    test_windows = [w for w in all_windows if w['trajectory_id'] in test_ids]

    train_labels = Counter(w['label'] for w in train_windows)
    test_labels = Counter(w['label'] for w in test_windows)

    print(f"  Train: {len(train_windows)} windows from {len(train_ids)} trajectories")
    print(f"    Labels: {dict(train_labels)}")
    if train_labels.get('STUCK', 0) > 0:
        ratio = train_labels['STUCK'] / len(train_windows)
        print(f"    Stuck ratio: {ratio:.1%}")

    print(f"  Test: {len(test_windows)} windows from {len(test_ids)} trajectories")
    print(f"    Labels: {dict(test_labels)}")
    if test_labels.get('STUCK', 0) > 0:
        ratio = test_labels['STUCK'] / len(test_windows)
        print(f"    Stuck ratio: {ratio:.1%}")

    # Save
    print("\nSaving...")
    with open(TRAIN_FILE, 'w') as f:
        for w in train_windows:
            f.write(json.dumps(w) + '\n')
    print(f"  {TRAIN_FILE}: {len(train_windows)} windows")

    with open(TEST_FILE, 'w') as f:
        for w in test_windows:
            f.write(json.dumps(w) + '\n')
    print(f"  {TEST_FILE}: {len(test_windows)} windows")

    # Sanity checks
    print("\n=== Sanity Checks ===")

    # 1. No empty windows
    empty = sum(1 for w in all_windows if not w['steps'])
    print(f"  Empty windows: {empty} {'OK' if empty == 0 else 'FAIL'}")

    # 2. All windows have correct size
    wrong_size = sum(1 for w in all_windows if len(w['steps']) != WINDOW_SIZE)
    print(f"  Wrong-size windows: {wrong_size} {'OK' if wrong_size == 0 else 'FAIL'}")

    # 3. No train/test trajectory overlap
    print(f"  Train/test overlap: {len(train_ids & test_ids)} {'OK' if len(train_ids & test_ids) == 0 else 'FAIL'}")

    # 4. Class balance preserved across split
    if train_labels.get('STUCK', 0) > 0 and test_labels.get('STUCK', 0) > 0:
        train_ratio = train_labels['STUCK'] / len(train_windows)
        test_ratio = test_labels['STUCK'] / len(test_windows)
        print(f"  Stuck ratio train={train_ratio:.3f} test={test_ratio:.3f} diff={abs(train_ratio-test_ratio):.3f}")
    else:
        print(f"  WARNING: Stuck only in {'train' if train_labels.get('STUCK', 0) > 0 else 'test' if test_labels.get('STUCK', 0) > 0 else 'neither'}")

    # 5. Feature value ranges
    sample = all_windows[:100]
    for feat_name in ['steps_since_same_cmd', 'output_similarity', 'step_index_norm']:
        vals = [s[feat_name] for w in sample for s in w['steps']]
        print(f"  {feat_name}: min={min(vals):.3f} max={max(vals):.3f}")

    print("\nDone!")


if __name__ == '__main__':
    process()
