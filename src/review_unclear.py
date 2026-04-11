"""Merge Sonnet/Opus review results back into per-source labeled files.

Usage:
    # Check status for a source
    python src/review_unclear.py status <source>

    # After Sonnet results are in, merge resolved windows back into source file
    python src/review_unclear.py merge-sonnet <source>

    # After Opus results are in, final merge
    python src/review_unclear.py merge-opus <source>

    # <source> is the source name, e.g. 'nlile', 'dataclaw', 'work_embedded_c'
"""

import json
import os
import sys
from collections import Counter

UNCLEAR_DIR    = 'data/cc_unclear_batches'
SOURCES_DIR    = 'data/sources'
SONNET_RESULTS_DIR = 'data/cc_sonnet_results'
OPUS_DIR       = 'data/cc_opus_batches'
OPUS_RESULTS_DIR   = 'data/cc_opus_results'


def source_paths(source):
    return {
        'labeled':         os.path.join(SOURCES_DIR, f'{source}_labeled.jsonl'),
        'unclear_windows': os.path.join(SOURCES_DIR, f'{source}_unclear_windows.jsonl'),
        'batch_prefix':    f'{source}_batch_',
    }


def load_unclear_windows(unclear_windows_file):
    """Load full training windows keyed by unclear_id."""
    windows = {}
    with open(unclear_windows_file) as f:
        for line in f:
            w = json.loads(line)
            uid = w.pop('unclear_id', None)
            if uid:
                windows[uid] = w
    return windows


def status(source):
    """Show review progress for a source."""
    paths = source_paths(source)
    batch_prefix = paths['batch_prefix']
    batches = sorted(f for f in os.listdir(UNCLEAR_DIR)
                     if f.startswith(batch_prefix) and f.endswith('.jsonl'))
    os.makedirs(SONNET_RESULTS_DIR, exist_ok=True)

    done = pending = 0
    for b in batches:
        num = b.replace(batch_prefix, '').replace('.jsonl', '')
        result = os.path.join(SONNET_RESULTS_DIR, f'result_{source}_{num}.jsonl')
        if os.path.exists(result):
            done += 1
        else:
            pending += 1

    print(f"[{source}] Sonnet review: {done}/{len(batches)} batches done, {pending} pending")


def merge_sonnet(source):
    """Merge Sonnet results into source labeled file. UNCLEAR → Opus batches."""
    os.makedirs(OPUS_DIR, exist_ok=True)
    paths = source_paths(source)
    batch_prefix = paths['batch_prefix']

    print(f"Loading unclear windows for [{source}]...")
    windows = load_unclear_windows(paths['unclear_windows'])
    print(f"  {len(windows)} windows loaded")

    batch_items = {}
    for b in sorted(os.listdir(UNCLEAR_DIR)):
        if not b.startswith(batch_prefix) or not b.endswith('.jsonl'):
            continue
        with open(os.path.join(UNCLEAR_DIR, b)) as f:
            for line in f:
                item = json.loads(line)
                batch_items[item['id']] = item

    labels = Counter()
    resolved = []
    still_unclear = []

    result_files = sorted(f for f in os.listdir(SONNET_RESULTS_DIR)
                          if f.startswith(f'result_{source}_') and f.endswith('.jsonl'))
    if not result_files:
        print(f"No Sonnet results found for [{source}]. Run agents first.")
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
                rid   = r.get('id', '')
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
    print(f"Resolved: {len(resolved)}, Still UNCLEAR (→ Opus): {len(still_unclear)}")

    with open(paths['labeled'], 'a') as f:
        for w in resolved:
            f.write(json.dumps(w) + '\n')
    print(f"Appended {len(resolved)} to {paths['labeled']}")
    print(f"Compress when ready: gzip -k {paths['labeled']}")

    n_batches = 0
    for i in range(0, len(still_unclear), 50):
        batch = still_unclear[i:i + 50]
        batch_file = os.path.join(OPUS_DIR, f'{source}_batch_{n_batches:04d}.jsonl')
        with open(batch_file, 'w') as f:
            for item in batch:
                f.write(json.dumps(item) + '\n')
        n_batches += 1
    if n_batches:
        print(f"Wrote {n_batches} Opus batches to {OPUS_DIR}/")


def merge_opus(source):
    """Final merge: Opus results → source labeled file, drop remaining UNCLEAR."""
    paths = source_paths(source)
    windows = load_unclear_windows(paths['unclear_windows'])

    labels = Counter()
    resolved = []

    if not os.path.isdir(OPUS_RESULTS_DIR):
        print("No Opus results directory found.")
        return

    for rf in sorted(os.listdir(OPUS_RESULTS_DIR)):
        if not rf.startswith(source) or not rf.endswith('.jsonl'):
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
                rid   = r.get('id', '')
                labels[label] += 1
                if label in ('STUCK', 'PRODUCTIVE') and rid in windows:
                    w = windows[rid]
                    w['label'] = label
                    resolved.append(w)

    dropped = labels.get('UNCLEAR', 0)
    print(f"Opus results: {dict(labels)}")
    print(f"Resolved: {len(resolved)}, Dropped (still UNCLEAR): {dropped}")

    with open(paths['labeled'], 'a') as f:
        for w in resolved:
            f.write(json.dumps(w) + '\n')
    print(f"Appended {len(resolved)} to {paths['labeled']}")
    print(f"Compress when ready: gzip -k {paths['labeled']}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python src/review_unclear.py [status|merge-sonnet|merge-opus] <source>")
        print("  source: nlile, dataclaw, work_embedded_c, ...")
        sys.exit(1)

    cmd    = sys.argv[1]
    source = sys.argv[2]
    if cmd == 'status':
        status(source)
    elif cmd == 'merge-sonnet':
        merge_sonnet(source)
    elif cmd == 'merge-opus':
        merge_opus(source)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
