"""
Merge per-source labeled window files into train/test splits.

Each source in data/sources/*_labeled.jsonl.gz is decompressed, oversampled
per config, shuffled with a fixed seed, and split into:
  data/train_windows.jsonl
  data/test_id.jsonl

Test split is preserved by data/sources/test_trajectories.txt (committed).
Trajectories in that file always go to test. New trajectories from new sources
are assigned to test at TEST_RATIO by trajectory_id hash (deterministic).

Usage:
    python src/merge_sources.py           # skip if outputs already exist
    python src/merge_sources.py --force   # regenerate
"""

import argparse
import gzip
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

SOURCES_DIR   = 'data/sources'
TRAIN_FILE    = 'data/train_windows.jsonl'
TEST_FILE     = 'data/test_id.jsonl'
TEST_TIDS_FILE = os.path.join(SOURCES_DIR, 'test_trajectories.txt')

# Per-source oversampling factors.
# DataClaw is the only source with thinking blocks — oversample to compensate.
# Add new sources here; unknown sources default to 1.
SOURCE_CONFIG = {
    'nlile':           {'oversample': 1},
    'dataclaw':        {'oversample': 5},
    'lelouch':         {'oversample': 3},
    'work_embedded_c': {'oversample': 3},
}

TEST_RATIO = 0.20
SEED       = 42


def load_test_trajectories():
    """Load committed test trajectory IDs (preserves evaluation baseline)."""
    if not os.path.exists(TEST_TIDS_FILE):
        return set()
    with open(TEST_TIDS_FILE) as f:
        return {line.strip() for line in f if line.strip()}


def is_test_by_hash(trajectory_id, ratio=TEST_RATIO):
    """Deterministic test assignment for new trajectories not in test_trajectories.txt."""
    h = int(hashlib.md5(trajectory_id.encode()).hexdigest(), 16)
    return (h % 1000) < int(ratio * 1000)


def source_name_from_path(path):
    """Extract source name from data/sources/NAME_labeled.jsonl.gz."""
    base = os.path.basename(path)
    return base.replace('_labeled.jsonl.gz', '')


def load_source(path):
    """Load all windows from a gzipped labeled JSONL file."""
    windows = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            line = line.strip()
            if line:
                windows.append(json.loads(line))
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if output files exist')
    args = parser.parse_args()

    if not args.force and os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE):
        print(f"{TRAIN_FILE} and {TEST_FILE} already exist. Use --force to regenerate.")
        return

    test_tids = load_test_trajectories()
    print(f"Loaded {len(test_tids)} committed test trajectory IDs")

    # Discover source files
    source_files = sorted(
        os.path.join(SOURCES_DIR, f)
        for f in os.listdir(SOURCES_DIR)
        if f.endswith('_labeled.jsonl.gz')
    )
    if not source_files:
        print(f"No *_labeled.jsonl.gz files found in {SOURCES_DIR}/")
        sys.exit(1)

    all_train = []
    all_test  = []
    stats = []

    for path in source_files:
        name = source_name_from_path(path)
        cfg  = SOURCE_CONFIG.get(name, {'oversample': 1})
        oversample = cfg['oversample']

        windows = load_source(path)
        labels  = Counter(w['label'] for w in windows)

        # Split by trajectory
        by_traj = defaultdict(list)
        for w in windows:
            by_traj[w['trajectory_id']].append(w)

        train_windows = []
        test_windows  = []
        for tid, traj_windows in by_traj.items():
            if tid in test_tids or is_test_by_hash(tid):
                test_windows.extend(traj_windows)
            else:
                train_windows.extend(traj_windows)

        # Apply oversampling to train only
        train_oversampled = train_windows * oversample

        all_train.extend(train_oversampled)
        all_test.extend(test_windows)

        stats.append({
            'source':    name,
            'windows':   len(windows),
            'labels':    dict(labels),
            'train_raw': len(train_windows),
            'test':      len(test_windows),
            'oversample': oversample,
            'train_eff': len(train_oversampled),
        })
        print(f"  {name}: {len(windows)} windows "
              f"(STUCK={labels.get('STUCK',0)}, PROD={labels.get('PRODUCTIVE',0)}) "
              f"→ train×{oversample}={len(train_oversampled)}, test={len(test_windows)}")

    # Shuffle with fixed seed
    rng = random.Random(SEED)
    rng.shuffle(all_train)
    rng.shuffle(all_test)

    # Write outputs
    with open(TRAIN_FILE, 'w') as f:
        for w in all_train:
            f.write(json.dumps(w) + '\n')

    with open(TEST_FILE, 'w') as f:
        for w in all_test:
            f.write(json.dumps(w) + '\n')

    # Summary
    train_labels = Counter(w['label'] for w in all_train)
    test_labels  = Counter(w['label'] for w in all_test)
    print(f"\n{'='*60}")
    print(f"train_windows.jsonl: {len(all_train)} windows  "
          f"STUCK={train_labels.get('STUCK',0)}  PROD={train_labels.get('PRODUCTIVE',0)}")
    print(f"test_id.jsonl:       {len(all_test)} windows  "
          f"STUCK={test_labels.get('STUCK',0)}  PROD={test_labels.get('PRODUCTIVE',0)}")
    print(f"\nTo add a new source:")
    print(f"  1. Place data/sources/<name>_labeled.jsonl.gz")
    print(f"  2. Add oversampling factor to SOURCE_CONFIG in this script")
    print(f"  3. Run: python src/merge_sources.py --force")


if __name__ == '__main__':
    main()
