"""Sample heuristic-PRODUCTIVE windows and run them through Sonnet review.

Re-parses raw sessions to capture the human-readable step format (cmd, output
snippets) that the label pipeline only builds for CANDIDATE windows.  Randomly
samples N productive windows, writes them as a single CANDIDATE batch, then
runs Sonnet review via run_review.py logic and reports the false-positive rate.

Usage:
  python src/sample_productive.py [--n 300] [--out /tmp/prod_sample]
"""

import gc
import json
import os
import random
import sys

# -- reuse label_sessions internals ------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from label_sessions import (
    abstract_trajectory, compute_window_features, precompute_review_counts,
    classify, _review_step, _step_features, parse_nlile_session,
    WINDOW_SIZE, STRIDE, TOOL_NAMES,
)
try:
    from parse_dataclaw import parse_dataclaw_session, has_outputs
    HAS_DATACLAW = True
except ImportError:
    HAS_DATACLAW = False


def create_productive_review_items(abstract_seq, trajectory_id, parsed_steps):
    """Like create_windows but returns review-format items for PRODUCTIVE windows."""
    items = []
    for start in range(0, len(abstract_seq) - WINDOW_SIZE + 1, STRIDE):
        window    = abstract_seq[start:start + WINDOW_SIZE]
        win_feats = compute_window_features(window)
        precomp   = precompute_review_counts(window)
        label, _  = classify(precomp)
        if label != 'PRODUCTIVE':
            continue
        raw_window = parsed_steps[start:start + WINDOW_SIZE] if parsed_steps else []
        review_steps = [
            _review_step(window[j], raw_window[j] if j < len(raw_window) else None)
            for j in range(len(window))
        ]
        full_window = {
            'trajectory_id': trajectory_id,
            'window_start':  start,
            'label':         'PRODUCTIVE',
            'label_source':  'heuristic',
            'steps':         [_step_features(s) for s in window],
            'window_features': win_feats,
        }
        items.append({
            'id':            f'{trajectory_id}_w{start}',
            'steps':         review_steps,
            'precomputed':   precomp,
            'window_features': win_feats,
            '_full_window':  full_window,
        })
    return items


def collect_nlile(limit=None):
    import pyarrow.parquet as pq
    parquet_dir = 'data/separate/nlile_parquet/data'
    files = sorted(f for f in os.listdir(parquet_dir) if f.endswith('.parquet'))
    items = []
    for fname in files:
        pf = pq.read_table(os.path.join(parquet_dir, fname))
        for i in range(len(pf)):
            msgs_raw = pf.column('messages_json')[i].as_py()
            if not msgs_raw:
                continue
            msgs = json.loads(msgs_raw)
            has_tools = any(
                isinstance(m.get('content'), list) and
                any(b.get('type') == 'tool_use' for b in m['content'] if isinstance(b, dict))
                for m in msgs
            )
            if not has_tools:
                continue
            row_id = pf.column('id')[i].as_py()
            parsed = parse_nlile_session(msgs)
            if len(parsed) < WINDOW_SIZE:
                continue
            abstract = abstract_trajectory(parsed)
            if len(abstract) < WINDOW_SIZE:
                continue
            items.extend(create_productive_review_items(abstract, f'nlile_{row_id}', parsed))
            if limit and len(items) >= limit:
                del pf; gc.collect()
                return items
        del pf; gc.collect()
        print(f'  nlile {fname}: {len(items)} productive so far', flush=True)
    return items


def collect_dataclaw():
    if not HAS_DATACLAW:
        return []
    path = 'data/separate/dataclaw/woctordho/conversations.jsonl'
    items = []
    with open(path) as f:
        for line in f:
            sess = json.loads(line)
            if not has_outputs(sess['messages']):
                continue
            parsed = parse_dataclaw_session(sess['messages'])
            if len(parsed) < WINDOW_SIZE:
                continue
            abstract = abstract_trajectory(parsed)
            if len(abstract) < WINDOW_SIZE:
                continue
            items.extend(create_productive_review_items(abstract, f"dc_{sess['session_id']}", parsed))
    print(f'  dataclaw: {len(items)} productive', flush=True)
    return items


def main():
    n_sample = 300
    out_dir  = '/tmp/prod_sample'

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--n' and i + 1 < len(args):
            n_sample = int(args[i+1]); i += 2
        elif args[i] == '--out' and i + 1 < len(args):
            out_dir = args[i+1]; i += 2
        else:
            i += 1

    os.makedirs(out_dir, exist_ok=True)

    print('Collecting heuristic-PRODUCTIVE windows from raw sessions...')
    # Collect slightly more than needed so random sample is unbiased;
    # stop early once we have 10× the target to avoid full scan
    nlile_items = collect_nlile(limit=n_sample * 10)
    dc_items    = collect_dataclaw()
    all_items   = nlile_items + dc_items
    print(f'Total productive windows scanned: {len(all_items)}')

    if len(all_items) < n_sample:
        print(f'Warning: only {len(all_items)} available, using all')
        sample = all_items
    else:
        sample = random.sample(all_items, n_sample)

    batch_path = os.path.join(out_dir, 'productive_sample_batch_0000.jsonl')
    with open(batch_path, 'w') as f:
        for item in sample:
            f.write(json.dumps(item) + '\n')
    print(f'Wrote {len(sample)} items to {batch_path}')
    print(f'\nNext:')
    print(f'  python src/run_review.py sonnet --batch-dir {out_dir} --out-dir {out_dir}/results')
    print(f'  # then check false-positive rate in {out_dir}/results/')


if __name__ == '__main__':
    main()
