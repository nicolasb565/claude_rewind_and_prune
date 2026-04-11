"""Migrate labels from cc_labeled_backup.jsonl into the new pipeline format.

Run after label_sessions.py has produced fresh CANDIDATE batches with clean v4
features. Resolves each CANDIDATE against existing labels so we don't re-pay
for Sonnet review we already have.

Resolution order per CANDIDATE window:
  1. Found in cc_labeled_backup.jsonl  → apply label, label_source='sonnet'
     (PRODUCTIVE from heuristic never appears as CANDIDATE; anything found here
      was old STUCK or UNCLEAR, both Sonnet-reviewed)
  2. Not found, but Sonnet said UNCLEAR → data/review/escalated/ (→ Opus)
  3. Not found anywhere               → data/review/batches/    (→ new Sonnet)

Usage:
  # First rename the backup, then run label_sessions.py, then this script:
  mv data/cc_labeled.jsonl data/cc_labeled_backup.jsonl
  python src/label_sessions.py nlile
  python src/label_sessions.py dataclaw
  python src/label_sessions.py <source> <path>   # repeat for each source
  python src/migrate_labels.py

  # Then run reviews as normal:
  python src/review_sonnet.py <source>   # for any new Sonnet batches
  python src/review_opus.py <source>     # for escalated UNCLEAR batches
  python src/merge_sources.py --force
"""

import json
import os
import sys
from collections import Counter

BACKUP_FILE  = 'data/cc_labeled_backup.jsonl'
BATCHES_DIR  = 'data/review/batches'
SOURCES_DIR  = 'data/sources'
ESCALATE_DIR = 'data/review/escalated'

# Old Sonnet result directories — scanned to find previously UNCLEAR windows
OLD_RESULT_DIRS = [
    'data/cc_sonnet_results',
    'data/cc_sonnet_results_small',
]

BATCH_SIZE = 50


def load_prior_labels(backup_file):
    """Load cc_labeled_backup.jsonl → {trajectory_id_wN: label}."""
    prior = {}
    if not os.path.exists(backup_file):
        print(f"Warning: {backup_file} not found — no prior labels available")
        return prior
    with open(backup_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            w = json.loads(line)
            wid = f"{w['trajectory_id']}_w{w['window_start']}"
            prior[wid] = w['label']
    print(f"Loaded {len(prior):,} prior labels from {backup_file}")
    return prior


def load_prior_unclear(result_dirs):
    """Scan old Sonnet result dirs → set of IDs where Sonnet said UNCLEAR."""
    unclear = set()
    for d in result_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.endswith('.jsonl'):
                continue
            with open(os.path.join(d, fname)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r.get('label', '').upper() == 'UNCLEAR':
                        unclear.add(r['id'])
    print(f"Loaded {len(unclear):,} previously UNCLEAR IDs from old Sonnet results")
    return unclear


def load_candidates(batches_dir):
    """Load all CANDIDATE batch files → {source: [items]}."""
    by_source = {}
    if not os.path.isdir(batches_dir):
        print(f"No batches directory found: {batches_dir}")
        return by_source
    for fname in sorted(os.listdir(batches_dir)):
        if not fname.endswith('.jsonl'):
            continue
        # fname format: {source}_batch_NNNN.jsonl
        source = fname.rsplit('_batch_', 1)[0]
        by_source.setdefault(source, [])
        with open(os.path.join(batches_dir, fname)) as f:
            for line in f:
                line = line.strip()
                if line:
                    by_source[source].append(json.loads(line))
    return by_source


def migrate(sources=None):
    os.makedirs(ESCALATE_DIR, exist_ok=True)

    prior   = load_prior_labels(BACKUP_FILE)
    unclear = load_prior_unclear(OLD_RESULT_DIRS)

    by_source = load_candidates(BATCHES_DIR)
    if not by_source:
        print("No CANDIDATE batches found. Run label_sessions.py first.")
        return

    if sources:
        by_source = {s: v for s, v in by_source.items() if s in sources}

    total_resolved = total_escalated = total_new_sonnet = 0

    for source, items in sorted(by_source.items()):
        counts    = Counter()
        resolved  = []
        escalated = []
        new_sonnet = []

        for item in items:
            wid = item['id']
            if wid in prior:
                full_window = item['_full_window']
                full_window['label']        = prior[wid]
                full_window['label_source'] = 'sonnet'
                resolved.append(full_window)
                counts['resolved'] += 1
            elif wid in unclear:
                escalated.append(item)
                counts['escalated'] += 1
            else:
                new_sonnet.append(item)
                counts['new_sonnet'] += 1

        print(f"\n[{source}] {len(items)} candidates:")
        print(f"  Resolved from backup:  {counts['resolved']}")
        print(f"  Escalated to Opus:     {counts['escalated']}")
        print(f"  New Sonnet batches:    {counts['new_sonnet']}")

        # Append resolved windows to labeled file
        if resolved:
            labeled_file = os.path.join(SOURCES_DIR, f'{source}_labeled.jsonl')
            with open(labeled_file, 'a') as f:
                for w in resolved:
                    f.write(json.dumps(w) + '\n')
            print(f"  Appended {len(resolved)} windows to {labeled_file}")

        # Write escalated items for Opus
        if escalated:
            n = 0
            for i in range(0, len(escalated), BATCH_SIZE):
                batch = escalated[i:i + BATCH_SIZE]
                out = os.path.join(ESCALATE_DIR, f'{source}_batch_{n:04d}.jsonl')
                with open(out, 'w') as f:
                    for item in batch:
                        f.write(json.dumps(item) + '\n')
                n += 1
            print(f"  Wrote {n} Opus batches to {ESCALATE_DIR}/")

        # Write new Sonnet batches (overwrite existing batch files)
        if new_sonnet:
            # Remove old batch files for this source first
            for fname in os.listdir(BATCHES_DIR):
                if fname.startswith(f'{source}_batch_') and fname.endswith('.jsonl'):
                    os.remove(os.path.join(BATCHES_DIR, fname))
            n = 0
            for i in range(0, len(new_sonnet), BATCH_SIZE):
                batch = new_sonnet[i:i + BATCH_SIZE]
                out = os.path.join(BATCHES_DIR, f'{source}_batch_{n:04d}.jsonl')
                with open(out, 'w') as f:
                    for item in batch:
                        f.write(json.dumps(item) + '\n')
                n += 1
            print(f"  Wrote {n} new Sonnet batches to {BATCHES_DIR}/")
        else:
            # No new Sonnet work — remove the old batch files
            for fname in os.listdir(BATCHES_DIR):
                if fname.startswith(f'{source}_batch_') and fname.endswith('.jsonl'):
                    os.remove(os.path.join(BATCHES_DIR, fname))

        # Correction pass: heuristic-PRODUCTIVE windows that backup says STUCK
        # are false negatives introduced by v4 feature recomputation shifting
        # borderline windows across the PRODUCTIVE threshold.  Sonnet is ground
        # truth so we override them in-place.
        labeled_file = os.path.join(SOURCES_DIR, f'{source}_labeled.jsonl')
        if os.path.exists(labeled_file) and prior:
            corrected = 0
            windows = []
            with open(labeled_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    w = json.loads(line)
                    wid = f"{w['trajectory_id']}_w{w['window_start']}"
                    if (w.get('label_source') == 'heuristic'
                            and w.get('label') == 'PRODUCTIVE'
                            and prior.get(wid) == 'STUCK'):
                        w['label']        = 'STUCK'
                        w['label_source'] = 'sonnet'
                        corrected += 1
                    windows.append(w)
            if corrected:
                with open(labeled_file, 'w') as f:
                    for w in windows:
                        f.write(json.dumps(w) + '\n')
                print(f"  Corrected {corrected} heuristic-PRODUCTIVE → STUCK "
                      f"(backup override)")
                counts['corrected'] = corrected

        total_resolved   += counts['resolved']
        total_escalated  += counts['escalated']
        total_new_sonnet += counts['new_sonnet']

    print(f"\nTotal across all sources:")
    print(f"  Resolved (→ labeled):  {total_resolved}")
    print(f"  Escalated (→ Opus):    {total_escalated}")
    print(f"  New Sonnet batches:    {total_new_sonnet}")

    if total_new_sonnet:
        print(f"\nNext: run Sonnet review agents on {BATCHES_DIR}/<source>_batch_*.jsonl")
        print(f"      python src/review_sonnet.py <source>")
    if total_escalated:
        print(f"      run Opus review agents on {ESCALATE_DIR}/<source>_batch_*.jsonl")
        print(f"      python src/review_opus.py <source>")
    if not total_new_sonnet and not total_escalated:
        print(f"\nAll candidates resolved. Next:")
        print(f"  gzip -k data/sources/<source>_labeled.jsonl")
        print(f"  python src/merge_sources.py --force")


def main():
    sources = [s for s in sys.argv[1:] if not s.startswith('--')] or None
    migrate(sources)


if __name__ == '__main__':
    main()
