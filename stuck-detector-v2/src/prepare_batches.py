"""Steps 9-11: Abstract curated pool, run heuristic, write batch files.

Reads curated_pool.jsonl, abstracts each trajectory, runs heuristic labeling,
pre-computes review counts, and writes batch files for Haiku review.
"""

import json
import sys
import os
import gc

sys.path.insert(0, os.path.dirname(__file__))
from abstract_trajectory import abstract_trajectory, precompute_review_counts

POOL_FILE = 'data/curated_pool.jsonl'
BATCH_DIR = 'data/reviews'
BATCH_SIZE = 50


def heuristic_suggest(abstract_steps):
    """Per-step heuristic suggestions."""
    suggestions = []
    for i, step in enumerate(abstract_steps):
        stuck_signals = 0
        productive_signals = 0

        if step['steps_since_same_cmd'] < 0.15 and step['output_similarity'] > 0.8:
            stuck_signals += 2
        elif step['steps_since_same_cmd'] < 0.3:
            stuck_signals += 1
        if step['output_similarity'] > 0.8 and step['steps_since_same_cmd'] < 0.3:
            stuck_signals += 1
        if step['cmd_count_in_window'] > 0.4:
            stuck_signals += 1

        if step['steps_since_same_cmd'] > 0.5:
            productive_signals += 1
        if step['output_similarity'] < 0.3 and step['steps_since_same_cmd'] < 0.5:
            productive_signals += 1

        if stuck_signals >= 2 and productive_signals == 0:
            suggestion = 'stuck'
        elif productive_signals >= 1 and stuck_signals == 0:
            suggestion = 'productive'
        else:
            suggestion = 'ambiguous'

        suggestions.append(suggestion)
    return suggestions


def compact_step(step):
    """Create a compact step dict for the batch file."""
    return {
        'tool': step['tool'],
        'since_cmd': round(step['steps_since_same_cmd'], 2),
        'since_file': round(step['steps_since_same_file'], 2),
        'out_sim': round(step['output_similarity'], 2),
        'cmd_count': round(step['cmd_count_in_window'], 2),
        'error': step['is_error'],
    }


def process():
    os.makedirs(BATCH_DIR, exist_ok=True)

    # Process in streaming fashion
    batch = []
    batch_idx = 0
    total = 0
    skipped = 0

    print("Processing curated pool...")
    with open(POOL_FILE) as f:
        for line_num, line in enumerate(f):
            rec = json.loads(line)
            parsed_steps = rec.get('parsed_steps')

            if not parsed_steps or len(parsed_steps) < 3:
                skipped += 1
                continue

            # Abstract
            abstract = abstract_trajectory(parsed_steps)
            if len(abstract) < 3:
                skipped += 1
                continue

            # Heuristic
            heuristic = heuristic_suggest(abstract)
            stuck_count = heuristic.count('stuck')
            prod_count = heuristic.count('productive')
            ambig_count = heuristic.count('ambiguous')

            # Last 10 steps for reviewer
            last_10 = abstract[-10:]
            last_10_compact = [compact_step(s) for s in last_10]
            last_10_heuristic = heuristic[-10:]

            # Precomputed counts for Haiku
            precomputed = precompute_review_counts(last_10)

            # Build batch item
            item = {
                'id': rec['trajectory_id'],
                'source': rec['source_dataset'],
                'model_strength': rec['model_strength'],
                'resolved': rec.get('resolved'),
                'exit_status': rec.get('exit_status'),
                'n_steps': rec['n_steps'],
                'heuristic_summary': {
                    'stuck': stuck_count,
                    'productive': prod_count,
                    'ambiguous': ambig_count,
                    'overall': ('stuck' if stuck_count > prod_count
                                else 'productive' if prod_count > stuck_count
                                else 'ambiguous'),
                },
                'tool_sequence': ' -> '.join(s['tool'] for s in abstract[-15:]),
                'last_10_steps': last_10_compact,
                'per_step_heuristic': last_10_heuristic,
                'precomputed': precomputed,
            }

            batch.append(item)
            total += 1

            # Write batch when full
            if len(batch) >= BATCH_SIZE:
                batch_file = os.path.join(BATCH_DIR, f'batch_{batch_idx:04d}.jsonl')
                with open(batch_file, 'w') as bf:
                    for b in batch:
                        bf.write(json.dumps(b) + '\n')
                batch_idx += 1
                batch = []

            if (line_num + 1) % 2000 == 0:
                print(f"  Processed {line_num + 1}... ({total} kept, {skipped} skipped)")

    # Write remaining
    if batch:
        batch_file = os.path.join(BATCH_DIR, f'batch_{batch_idx:04d}.jsonl')
        with open(batch_file, 'w') as bf:
            for b in batch:
                bf.write(json.dumps(b) + '\n')
        batch_idx += 1

    print(f"\n{'='*50}")
    print(f"Total trajectories processed: {total}")
    print(f"Skipped (too short): {skipped}")
    print(f"Batch files written: {batch_idx} (batch size {BATCH_SIZE})")
    print(f"Batch files in: {BATCH_DIR}/batch_0000.jsonl .. batch_{batch_idx-1:04d}.jsonl")


if __name__ == '__main__':
    process()
