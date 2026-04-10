"""Merge Sonnet/Opus review results back into training data.

Usage:
    # Check status
    python src/review_unclear.py status

    # After Sonnet results are in, merge and escalate UNCLEAR to Opus
    python src/review_unclear.py merge-sonnet

    # After Opus results are in, final merge
    python src/review_unclear.py merge-opus
"""

import json
import os
import sys
from collections import Counter

UNCLEAR_DIR = 'data/cc_unclear_batches'
UNCLEAR_WINDOWS = 'data/cc_unclear_windows.jsonl'
SONNET_RESULTS_DIR = 'data/cc_sonnet_results'
OPUS_DIR = 'data/cc_opus_batches'
OPUS_RESULTS_DIR = 'data/cc_opus_results'
LABELED_FILE = 'data/cc_labeled.jsonl'


def load_unclear_windows():
    """Load full training windows keyed by unclear_id."""
    windows = {}
    with open(UNCLEAR_WINDOWS) as f:
        for line in f:
            w = json.loads(line)
            uid = w.pop('unclear_id', None)
            if uid:
                windows[uid] = w
    return windows


def status():
    """Show review progress."""
    batches = sorted(f for f in os.listdir(UNCLEAR_DIR)
                     if f.startswith('batch_') and f.endswith('.jsonl'))
    os.makedirs(SONNET_RESULTS_DIR, exist_ok=True)

    done = 0
    pending = 0
    for b in batches:
        num = b.replace('batch_', '').replace('.jsonl', '')
        result = os.path.join(SONNET_RESULTS_DIR, f'result_{num}.jsonl')
        if os.path.exists(result):
            done += 1
        else:
            pending += 1

    print(f"Sonnet review: {done}/{len(batches)} batches done, {pending} pending")

    if os.path.isdir(OPUS_DIR):
        opus_batches = [f for f in os.listdir(OPUS_DIR) if f.endswith('.jsonl')]
        opus_done = 0
        if os.path.isdir(OPUS_RESULTS_DIR):
            opus_done = len([f for f in os.listdir(OPUS_RESULTS_DIR) if f.endswith('.jsonl')])
        print(f"Opus review: {opus_done}/{len(opus_batches)} batches done")


def merge_sonnet():
    """Merge Sonnet results. STUCK/PRODUCTIVE → labeled file, UNCLEAR → Opus batches."""
    os.makedirs(OPUS_DIR, exist_ok=True)

    # Load full training windows for resolved items
    print("Loading unclear windows...")
    windows = load_unclear_windows()
    print(f"  {len(windows)} windows loaded")

    # Load batch items for UNCLEAR→Opus escalation (they have raw text for Opus)
    batch_items = {}
    for b in sorted(os.listdir(UNCLEAR_DIR)):
        if not b.startswith('batch_') or not b.endswith('.jsonl'):
            continue
        with open(os.path.join(UNCLEAR_DIR, b)) as f:
            for line in f:
                item = json.loads(line)
                batch_items[item['id']] = item

    # Read Sonnet results
    labels = Counter()
    resolved = []
    still_unclear = []

    result_files = sorted(f for f in os.listdir(SONNET_RESULTS_DIR) if f.endswith('.jsonl'))
    if not result_files:
        print("No Sonnet results found. Run agents first.")
        return

    for rf in result_files:
        with open(os.path.join(SONNET_RESULTS_DIR, rf)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                label = r.get('label', 'UNCLEAR')
                rid = r.get('id', '')
                labels[label] += 1

                if label in ('STUCK', 'PRODUCTIVE') and rid in windows:
                    w = windows[rid]
                    w['label'] = label
                    resolved.append(w)
                elif label == 'UNCLEAR' and rid in batch_items:
                    item = batch_items[rid]
                    item['sonnet_reason'] = r.get('reason', '')
                    still_unclear.append(item)

    print(f"\nSonnet results: {dict(labels)}")
    print(f"Resolved: {len(resolved)}")
    print(f"Still UNCLEAR (→ Opus): {len(still_unclear)}")

    # Append resolved to labeled file
    with open(LABELED_FILE, 'a') as f:
        for w in resolved:
            f.write(json.dumps(w) + '\n')
    print(f"Appended {len(resolved)} to {LABELED_FILE}")

    # Write Opus batches (with sonnet_reason for context)
    batch_size = 50
    n_batches = 0
    for i in range(0, len(still_unclear), batch_size):
        batch = still_unclear[i:i + batch_size]
        batch_file = os.path.join(OPUS_DIR, f'batch_{n_batches:04d}.jsonl')
        with open(batch_file, 'w') as f:
            for item in batch:
                f.write(json.dumps(item) + '\n')
        n_batches += 1
    print(f"Wrote {n_batches} Opus batches to {OPUS_DIR}/")

    # Print final labeled stats
    _print_labeled_stats()


def merge_opus():
    """Final merge: Opus results → labeled file, drop remaining UNCLEAR."""
    windows = load_unclear_windows()

    labels = Counter()
    resolved = []

    if not os.path.isdir(OPUS_RESULTS_DIR):
        print("No Opus results directory found.")
        return

    for rf in sorted(os.listdir(OPUS_RESULTS_DIR)):
        if not rf.endswith('.jsonl'):
            continue
        with open(os.path.join(OPUS_RESULTS_DIR, rf)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                label = r.get('label', 'UNCLEAR')
                rid = r.get('id', '')
                labels[label] += 1

                if label in ('STUCK', 'PRODUCTIVE') and rid in windows:
                    w = windows[rid]
                    w['label'] = label
                    resolved.append(w)

    print(f"Opus results: {dict(labels)}")
    dropped = labels.get('UNCLEAR', 0)
    print(f"Resolved: {len(resolved)}, Dropped (still UNCLEAR): {dropped}")

    with open(LABELED_FILE, 'a') as f:
        for w in resolved:
            f.write(json.dumps(w) + '\n')
    print(f"Appended {len(resolved)} to {LABELED_FILE}")

    _print_labeled_stats()


def _print_labeled_stats():
    """Print current labeled file stats."""
    labels = Counter()
    with open(LABELED_FILE) as f:
        for line in f:
            w = json.loads(line)
            labels[w['label']] += 1
    total = sum(labels.values())
    print(f"\nCurrent {LABELED_FILE}: {total} total")
    for lbl in ['STUCK', 'PRODUCTIVE']:
        print(f"  {lbl}: {labels.get(lbl, 0)} ({labels.get(lbl, 0)/total*100:.1f}%)")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/review_unclear.py [status|merge-sonnet|merge-opus]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == 'status':
        status()
    elif cmd == 'merge-sonnet':
        merge_sonnet()
    elif cmd == 'merge-opus':
        merge_opus()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
