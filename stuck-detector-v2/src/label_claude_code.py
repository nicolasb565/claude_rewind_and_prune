"""Label Claude Code windows (nlile + DataClaw) with deterministic rules.

Produces STUCK/PRODUCTIVE/UNCLEAR labels. UNCLEAR windows are written
to batch files for Sonnet review (and Opus escalation if still unclear).

Usage:
    python src/label_claude_code.py

Reads:  data/nlile_claude_code_windows.jsonl, data/dataclaw_windows.jsonl
Writes: data/cc_labeled.jsonl          (STUCK + PRODUCTIVE)
        data/cc_unclear_batches/       (UNCLEAR batches for Sonnet)
"""

import json
import os
import sys
from collections import Counter

TOOL_NAMES = ['bash', 'edit', 'view', 'search', 'create', 'submit', 'other']


def precompute_from_window(steps):
    """Compute review counts from window step dicts (tool_idx based)."""
    tight_loop_steps = sum(
        1 for s in steps
        if s['steps_since_same_cmd'] < 0.15 and s['output_similarity'] > 0.8
    )
    diverse_steps = sum(
        1 for s in steps
        if s['steps_since_same_cmd'] > 0.5
    )
    error_steps = sum(1 for s in steps if s.get('is_error'))
    tools = [TOOL_NAMES[s['tool_idx']] for s in steps]
    unique_tools = len(set(tools))
    has_submit = any(t == 'submit' for t in tools)

    return {
        'tight_loop_steps': tight_loop_steps,
        'diverse_steps': diverse_steps,
        'error_steps': error_steps,
        'unique_tools': unique_tools,
        'has_submit': has_submit,
    }


def classify_window(steps):
    """Apply deterministic rules to a window's steps.

    Returns (label, confidence, reason, precomputed).
    Matches Haiku agent rules exactly.
    """
    precomputed = precompute_from_window(steps)
    tight = precomputed['tight_loop_steps']
    diverse = precomputed['diverse_steps']
    errors = precomputed['error_steps']
    has_submit = precomputed.get('has_submit', False)

    reason_parts = f"tight_loop={tight} diverse={diverse} errors={errors}"

    # STUCK rules
    if tight >= 3 and tight >= diverse + 2:
        return 'STUCK', 'high', f"{reason_parts} → tight_loop>=3 AND tight_loop>=diverse+2", precomputed
    if errors >= 7 and diverse < 3:
        return 'STUCK', 'high', f"{reason_parts} → error_steps>=7 AND diverse<3", precomputed

    # PRODUCTIVE rules
    if tight == 0:
        return 'PRODUCTIVE', 'high', f"{reason_parts} → tight_loop==0", precomputed
    if diverse >= tight + 3:
        return 'PRODUCTIVE', 'high', f"{reason_parts} → diverse>=tight_loop+3", precomputed
    if diverse >= 6:
        return 'PRODUCTIVE', 'high', f"{reason_parts} → diverse>=6", precomputed
    if has_submit and diverse >= 2:
        return 'PRODUCTIVE', 'high', f"{reason_parts} → has_submit AND diverse>=2", precomputed

    # UNCLEAR — needs Sonnet review
    return 'UNCLEAR', 'low', f"{reason_parts} → no rule matched", precomputed


BATCH_SIZE = 50


def compact_step(step):
    """Create compact step for Sonnet review."""
    tool = TOOL_NAMES[step['tool_idx']] if 'tool_idx' in step else step.get('tool', 'other')
    return {
        'tool': tool,
        'since_cmd': round(step.get('steps_since_same_cmd', 0), 2),
        'since_file': round(step.get('steps_since_same_file', 0), 2),
        'out_sim': round(step.get('output_similarity', 0), 2),
        'cmd_count': round(step.get('cmd_count_in_window', 0), 2),
        'error': 1.0 if step.get('is_error', 0) else 0.0,
    }


def process():
    INPUT_FILES = [
        'data/nlile_claude_code_windows.jsonl',
        'data/dataclaw_windows.jsonl',
    ]
    LABELED_FILE = 'data/cc_labeled.jsonl'
    UNCLEAR_DIR = 'data/cc_unclear_batches'

    os.makedirs(UNCLEAR_DIR, exist_ok=True)

    labels = Counter()
    labeled_count = 0
    unclear_batch = []
    batch_idx = 0
    unclear_total = 0

    with open(LABELED_FILE, 'w') as out_f:
        for input_file in INPUT_FILES:
            source = os.path.basename(input_file).replace('_windows.jsonl', '')
            print(f"\nProcessing {input_file}...")
            line_count = 0

            with open(input_file) as f:
                for line in f:
                    w = json.loads(line)
                    steps = w['steps']
                    line_count += 1

                    label, confidence, reason, precomputed = classify_window(steps)
                    labels[label] += 1

                    if label in ('STUCK', 'PRODUCTIVE'):
                        w['label'] = label
                        out_f.write(json.dumps(w) + '\n')
                        labeled_count += 1
                    else:
                        # UNCLEAR — write to batch for Sonnet review
                        unclear_total += 1
                        batch_item = {
                            'id': w.get('trajectory_id', f'{source}_{line_count}'),
                            'source': source,
                            'window_start': w.get('window_start', 0),
                            'trajectory_id': w.get('trajectory_id', ''),
                            'steps': [compact_step(s) for s in steps],
                            'precomputed': precomputed,
                            'reason': reason,
                            # Keep full window for re-insertion after labeling
                            '_full_window': w,
                        }
                        unclear_batch.append(batch_item)

                        if len(unclear_batch) >= BATCH_SIZE:
                            batch_file = os.path.join(UNCLEAR_DIR, f'batch_{batch_idx:04d}.jsonl')
                            with open(batch_file, 'w') as bf:
                                for item in unclear_batch:
                                    bf.write(json.dumps(item) + '\n')
                            batch_idx += 1
                            unclear_batch = []

            print(f"  {line_count} windows from {source}")

    # Write remaining unclear batch
    if unclear_batch:
        batch_file = os.path.join(UNCLEAR_DIR, f'batch_{batch_idx:04d}.jsonl')
        with open(batch_file, 'w') as bf:
            for item in unclear_batch:
                bf.write(json.dumps(item) + '\n')
        batch_idx += 1

    print(f"\n{'='*50}")
    print(f"Label distribution: {dict(labels)}")
    print(f"Labeled (STUCK+PRODUCTIVE): {labeled_count} → {LABELED_FILE}")
    print(f"UNCLEAR: {unclear_total} → {batch_idx} batches in {UNCLEAR_DIR}/")
    total = sum(labels.values())
    for lbl in ['STUCK', 'PRODUCTIVE', 'UNCLEAR']:
        pct = labels[lbl] / total * 100 if total else 0
        print(f"  {lbl}: {labels[lbl]} ({pct:.1f}%)")


if __name__ == '__main__':
    process()
