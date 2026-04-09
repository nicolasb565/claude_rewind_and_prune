"""Process Claude Code data (nlile + DataClaw) end-to-end.

1. Parse raw sessions into step dicts
2. Abstract with cmd_semantic_key
3. Window into 10-step chunks
4. Label with deterministic rules (STUCK/PRODUCTIVE/UNCLEAR)

Usage:
    python src/process_claude_code.py

Reads:  data/separate/nlile_parquet/data/*.parquet
        data/separate/dataclaw/woctordho/conversations.jsonl
Writes: data/cc_labeled.jsonl      (STUCK + PRODUCTIVE windows)
        data/cc_unclear_batches/   (UNCLEAR for Sonnet review)
"""

import json
import os
import sys
import gc
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from abstract_trajectory import (
    abstract_trajectory, compute_window_features, precompute_review_counts,
    WINDOW_SIZE, STRIDE, TOOL_TO_IDX
)
from parse_dataclaw import parse_dataclaw_session, has_outputs

LABELED_FILE = 'data/cc_labeled.jsonl'
UNCLEAR_DIR = 'data/cc_unclear_batches'
BATCH_SIZE = 50

TOOL_NAMES = ['bash', 'edit', 'view', 'search', 'create', 'submit', 'other']

# --- Deterministic labeling ---

def classify_precomputed(precomputed):
    tight = precomputed['tight_loop_steps']
    diverse = precomputed['diverse_steps']
    errors = precomputed['error_steps']
    has_submit = precomputed.get('has_submit', False)
    reason_parts = f"tight_loop={tight} diverse={diverse} errors={errors}"

    if tight >= 3 and tight >= diverse + 2:
        return 'STUCK', f"{reason_parts} -> tight_loop>=3 AND tight_loop>=diverse+2"
    if errors >= 7 and diverse < 3:
        return 'STUCK', f"{reason_parts} -> error_steps>=7 AND diverse<3"
    if tight == 0:
        return 'PRODUCTIVE', f"{reason_parts} -> tight_loop==0"
    if diverse >= tight + 3:
        return 'PRODUCTIVE', f"{reason_parts} -> diverse>=tight_loop+3"
    if diverse >= 6:
        return 'PRODUCTIVE', f"{reason_parts} -> diverse>=6"
    if has_submit and diverse >= 2:
        return 'PRODUCTIVE', f"{reason_parts} -> has_submit AND diverse>=2"

    return 'UNCLEAR', f"{reason_parts} -> no rule matched"


# --- Window creation ---

def create_windows(abstract_seq, trajectory_id, parsed_steps=None):
    """Slide window, label each, return (labeled, unclear) lists.

    parsed_steps: original step dicts with raw text (cmd, file, output, thinking).
    Carried through to UNCLEAR batches so Sonnet can see actual content.
    """
    labeled = []
    unclear = []

    for start in range(0, len(abstract_seq) - WINDOW_SIZE + 1, STRIDE):
        window_steps = abstract_seq[start:start + WINDOW_SIZE]

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

        window_feats = compute_window_features(window_steps)
        precomputed = precompute_review_counts(window_steps)
        label, reason = classify_precomputed(precomputed)

        window = {
            'trajectory_id': trajectory_id,
            'window_start': start,
            'label': label,
            'steps': step_features,
            'window_features': window_feats,
        }

        if label == 'UNCLEAR':
            # Build rich batch item for Sonnet review with raw text
            raw_window = parsed_steps[start:start + WINDOW_SIZE] if parsed_steps else []
            review_steps = []
            for j, s in enumerate(window_steps):
                tool = TOOL_NAMES[s['tool_idx']] if isinstance(s['tool_idx'], int) else s.get('tool', 'other')
                step_data = {
                    'tool': tool,
                    'since_cmd': round(s['steps_since_same_cmd'], 2),
                    'since_file': round(s['steps_since_same_file'], 2),
                    'out_sim': round(s['output_similarity'], 2),
                    'cmd_count': round(s['cmd_count_in_window'], 2),
                    'error': 1.0 if s['is_error'] else 0.0,
                    'out_len': round(s['output_length'], 2),
                    'step_pos': round(s['step_index_norm'], 2),
                    'tool_repeat': round(s['tool_count_in_window'], 2),
                }
                # Add raw text from parsed steps (truncated)
                if j < len(raw_window):
                    raw = raw_window[j]
                    if raw.get('cmd'):
                        step_data['cmd'] = raw['cmd'][:200]
                    if raw.get('file'):
                        step_data['file'] = raw['file']
                    if raw.get('output'):
                        # First and last 5 lines, max 500 chars
                        lines = raw['output'].strip().split('\n')
                        if len(lines) <= 10:
                            step_data['output_snippet'] = raw['output'][:500]
                        else:
                            head = '\n'.join(lines[:5])
                            tail = '\n'.join(lines[-5:])
                            step_data['output_snippet'] = f"{head}\n... ({len(lines)} lines) ...\n{tail}"[:500]
                    if raw.get('thinking'):
                        step_data['thinking_snippet'] = raw['thinking'][:300]
                review_steps.append(step_data)
            unclear.append({
                'id': f"{trajectory_id}_w{start}",
                'trajectory_id': trajectory_id,
                'window_start': start,
                'steps': review_steps,
                'precomputed': precomputed,
                'window_features': window_feats,
                'reason': reason,
                '_full_window': window,
            })
        else:
            labeled.append(window)

    return labeled, unclear


# --- nlile parsing ---

def parse_nlile_session(messages):
    """Parse Anthropic API format messages into step dicts."""
    steps = []
    # Collect tool_use from assistant, tool_result from user
    pending_tool_uses = {}  # tool_use_id -> {tool, cmd, file}
    last_thinking = ''

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        if not isinstance(content, list):
            continue

        for block in content:
            btype = block.get('type', '')

            if btype == 'thinking':
                last_thinking = block.get('thinking', block.get('text', ''))

            elif btype == 'tool_use':
                name = block.get('name', '')
                inp = block.get('input', {})
                tool_id = block.get('id', '')
                tool = TOOL_TO_IDX_NAME.get(name, 'other')
                cmd = inp.get('command', inp.get('file_path', inp.get('pattern', '')))
                file_path = inp.get('file_path', inp.get('path', None))
                pending_tool_uses[tool_id] = {
                    'tool': tool,
                    'cmd': cmd,
                    'file': file_path,
                    'thinking': last_thinking,
                }
                last_thinking = ''

            elif btype == 'tool_result':
                tool_id = block.get('tool_use_id', '')
                if tool_id in pending_tool_uses:
                    tu = pending_tool_uses.pop(tool_id)
                    output = block.get('content', '')
                    if isinstance(output, list):
                        # Extract text from content blocks
                        output = '\n'.join(
                            b.get('text', '') for b in output
                            if isinstance(b, dict) and b.get('type') == 'text'
                        )
                    tu['output'] = str(output) if output else ''
                    steps.append(tu)

    # Flush pending tool_uses without results
    for tu in pending_tool_uses.values():
        tu['output'] = ''
        steps.append(tu)

    return steps


# Tool name -> abstract category for nlile
TOOL_TO_IDX_NAME = {
    'Bash': 'bash', 'bash': 'bash',
    'Read': 'view', 'read': 'view',
    'Edit': 'edit', 'edit': 'edit', 'Write': 'edit', 'write': 'edit', 'MultiEdit': 'edit',
    'Grep': 'search', 'grep': 'search', 'Glob': 'search', 'glob': 'search',
    'Agent': 'other', 'Task': 'other', 'TodoRead': 'other', 'TodoWrite': 'other',
}


def process_nlile():
    """Process nlile parquet files, return (labeled_windows, unclear_items)."""
    import pyarrow.parquet as pq

    parquet_dir = 'data/separate/nlile_parquet/data'
    files = sorted(f for f in os.listdir(parquet_dir) if f.endswith('.parquet'))

    all_labeled = []
    all_unclear = []
    sessions_processed = 0
    sessions_skipped = 0

    for fname in files:
        pf = pq.read_table(os.path.join(parquet_dir, fname))
        for i in range(len(pf)):
            msgs_raw = pf.column('messages_json')[i].as_py()
            if not msgs_raw:
                continue
            msgs = json.loads(msgs_raw)

            # Must have tool interactions (any source)
            has_tools = any(
                isinstance(m.get('content'), list) and
                any(b.get('type') == 'tool_use' for b in m['content'] if isinstance(b, dict))
                for m in msgs
            )
            if not has_tools:
                sessions_skipped += 1
                continue

            row_id = pf.column('id')[i].as_py()
            parsed = parse_nlile_session(msgs)

            if len(parsed) < WINDOW_SIZE:
                sessions_skipped += 1
                continue

            abstract = abstract_trajectory(parsed)
            if len(abstract) < WINDOW_SIZE:
                sessions_skipped += 1
                continue

            labeled, unclear = create_windows(abstract, f"nlile_{row_id}", parsed)
            all_labeled.extend(labeled)
            all_unclear.extend(unclear)
            sessions_processed += 1

        del pf
        gc.collect()
        print(f"  {fname}: {sessions_processed} sessions so far")

    print(f"nlile: {sessions_processed} sessions, {sessions_skipped} skipped")
    return all_labeled, all_unclear


def process_dataclaw():
    """Process DataClaw woctordho sessions."""
    path = 'data/separate/dataclaw/woctordho/conversations.jsonl'

    all_labeled = []
    all_unclear = []
    sessions_processed = 0
    sessions_skipped = 0

    with open(path) as f:
        for line in f:
            sess = json.loads(line)
            if not has_outputs(sess['messages']):
                sessions_skipped += 1
                continue

            parsed = parse_dataclaw_session(sess['messages'])
            if len(parsed) < WINDOW_SIZE:
                sessions_skipped += 1
                continue

            tid = f"dc_{sess['session_id']}"
            abstract = abstract_trajectory(parsed)
            if len(abstract) < WINDOW_SIZE:
                sessions_skipped += 1
                continue

            labeled, unclear = create_windows(abstract, tid, parsed)
            all_labeled.extend(labeled)
            all_unclear.extend(unclear)
            sessions_processed += 1

    print(f"DataClaw: {sessions_processed} sessions, {sessions_skipped} skipped")
    return all_labeled, all_unclear


def main():
    UNCLEAR_WINDOWS_FILE = 'data/cc_unclear_windows.jsonl'

    os.makedirs(UNCLEAR_DIR, exist_ok=True)

    # Clear old data
    if os.path.exists(LABELED_FILE):
        os.remove(LABELED_FILE)
    if os.path.exists(UNCLEAR_WINDOWS_FILE):
        os.remove(UNCLEAR_WINDOWS_FILE)
    for f in os.listdir(UNCLEAR_DIR):
        os.remove(os.path.join(UNCLEAR_DIR, f))

    print("=== Processing nlile ===")
    nlile_labeled, nlile_unclear = process_nlile()
    print(f"  Labeled: {len(nlile_labeled)}, Unclear: {len(nlile_unclear)}")

    print("\n=== Processing DataClaw ===")
    dc_labeled, dc_unclear = process_dataclaw()
    print(f"  Labeled: {len(dc_labeled)}, Unclear: {len(dc_unclear)}")

    # Combine
    all_labeled = nlile_labeled + dc_labeled
    all_unclear = nlile_unclear + dc_unclear

    # Write labeled windows
    labels = Counter()
    with open(LABELED_FILE, 'w') as f:
        for w in all_labeled:
            labels[w['label']] += 1
            f.write(json.dumps(w) + '\n')

    # Write unclear: full training windows (for merge) and review batches (for Sonnet)
    batch_idx = 0
    with open(UNCLEAR_WINDOWS_FILE, 'w') as wf:
        for i in range(0, len(all_unclear), BATCH_SIZE):
            batch = all_unclear[i:i + BATCH_SIZE]
            batch_file = os.path.join(UNCLEAR_DIR, f'batch_{batch_idx:04d}.jsonl')
            with open(batch_file, 'w') as f:
                for item in batch:
                    # Save full training window separately (keyed by id)
                    full_window = item.pop('_full_window', {})
                    full_window['unclear_id'] = item['id']
                    wf.write(json.dumps(full_window) + '\n')
                    # Batch file for Sonnet: no _full_window
                    f.write(json.dumps(item) + '\n')
            batch_idx += 1
    print(f"Full windows for merge: {UNCLEAR_WINDOWS_FILE}")

    print(f"\n{'='*50}")
    print(f"Label distribution: {dict(labels)}")
    print(f"Labeled (STUCK+PRODUCTIVE): {len(all_labeled)} -> {LABELED_FILE}")
    print(f"UNCLEAR: {len(all_unclear)} -> {batch_idx} batches in {UNCLEAR_DIR}/")
    total = len(all_labeled) + len(all_unclear)
    for lbl in ['STUCK', 'PRODUCTIVE']:
        print(f"  {lbl}: {labels[lbl]} ({labels[lbl]/total*100:.1f}%)")
    print(f"  UNCLEAR: {len(all_unclear)} ({len(all_unclear)/total*100:.1f}%)")

    # Verify features are populated
    print("\n=== Feature sanity check (sample from labeled) ===")
    if all_labeled:
        sample = all_labeled[:100]
        for feat in ['output_similarity', 'output_length', 'is_error', 'thinking_length',
                     'steps_since_same_cmd', 'self_similarity']:
            vals = [s[feat] for w in sample for s in w['steps']]
            nonzero = sum(1 for v in vals if v != 0) / len(vals) * 100
            print(f"  {feat:25s} mean={sum(vals)/len(vals):.4f} nonzero={nonzero:.1f}%")


if __name__ == '__main__':
    main()
