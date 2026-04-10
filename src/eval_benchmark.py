"""Score the LogReg benchmark sessions with the current CNN weights.

Reads session transcripts from Claude's project cache and scores each one.
Paths are machine-specific — configure via environment variables:

    BENCHMARK_RESULTS_DIR   directory with off_1/, off_2/, heldout_off_1/ subdirs,
                            each containing <task>.json files with session_id fields
                            (default: none, script exits if not set)

    CLAUDE_PROJECTS_DIR     directory where Claude Code stores session .jsonl files
                            (default: ~/.claude/projects)

    WORKTREE_BASE           base path used to derive Claude project dir names from task slugs.
                            The Claude project dir is derived by replacing '/' with '-' in
                            the worktree absolute path. Set this to the absolute path of
                            the directory containing the task worktrees.
                            (default: none — script searches common patterns)

Usage:
    BENCHMARK_RESULTS_DIR=/path/to/results python src/eval_benchmark.py
"""

import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

from train_cnn_oversample import (
    StuckDetectorTrimmed, ALL_FEATURES, KEEP_IDX, WINDOW_FEATURES, WINDOW_SIZE
)
from abstract_trajectory import abstract_trajectory

MODEL_DIR = 'proxy'
TOOL_TO_IDX = {'bash': 0, 'edit': 1, 'view': 2, 'search': 3, 'create': 4, 'submit': 5, 'other': 6}


def load_model():
    ckpt = torch.load(os.path.join(MODEL_DIR, 'cnn_trimmed_checkpoint.pt'), weights_only=False)
    model = StuckDetectorTrimmed()
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    mean = torch.tensor(ckpt['norm_mean'])
    std = torch.tensor(ckpt['norm_std']).clamp(min=1e-6)
    threshold = ckpt['threshold']
    return model, mean, std, threshold


def parse_cc_session(filepath):
    steps = []
    pending = {}
    tool_map = {
        'Bash': 'bash', 'Read': 'view', 'Edit': 'edit', 'Write': 'edit',
        'Grep': 'search', 'Glob': 'search', 'Agent': 'other', 'Task': 'other',
    }
    with open(filepath) as f:
        for line in f:
            entry = json.loads(line)
            msg = entry.get('message', {})
            if not msg or not isinstance(msg, dict):
                continue
            role = msg.get('role', '')
            content = msg.get('content', '')
            if not isinstance(content, list):
                continue
            if role == 'assistant':
                thinking = ''
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get('type') == 'thinking':
                        thinking = block.get('thinking', '')
                    elif block.get('type') == 'tool_use':
                        inp = block.get('input', {})
                        pending[block.get('id', '')] = {
                            'tool': tool_map.get(block.get('name', ''), 'other'),
                            'cmd': inp.get('command', inp.get('file_path', inp.get('pattern', ''))),
                            'file': inp.get('file_path', inp.get('path', None)),
                            'thinking': thinking,
                            'output': '',
                        }
                        thinking = ''
            elif role == 'user':
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get('type') == 'tool_result':
                        tid = block.get('tool_use_id', '')
                        if tid in pending:
                            out = block.get('content', '')
                            if isinstance(out, list):
                                out = '\n'.join(b.get('text', '') for b in out if isinstance(b, dict))
                            pending[tid]['output'] = str(out) if out else ''
                            steps.append(pending.pop(tid))
    return steps


def score_session(filepath, model, mean, std):
    steps = parse_cc_session(filepath)
    if len(steps) < WINDOW_SIZE:
        return None, len(steps)
    abstract = abstract_trajectory(steps)
    if len(abstract) < WINDOW_SIZE:
        return None, len(steps)

    scores = []
    for start in range(0, len(abstract) - WINDOW_SIZE + 1, 5):
        window = abstract[start:start + WINDOW_SIZE]

        cat = torch.tensor([[TOOL_TO_IDX.get(s['tool'], 6) for s in window]], dtype=torch.long)
        cont_raw = []
        for s in window:
            all_vals = [float(s.get(f, 0)) if not isinstance(s.get(f, 0), bool)
                        else (1.0 if s.get(f, 0) else 0.0) for f in ALL_FEATURES]
            cont_raw.append([all_vals[i] for i in KEEP_IDX])
        cont = (torch.tensor([cont_raw], dtype=torch.float32) - mean) / std

        tools = [s['tool'] for s in window]
        fh = [s.get('file_hash') for s in window if s.get('file_hash') is not None]
        ch = [s.get('cmd_hash') for s in window if s.get('cmd_hash') is not None]
        al = [line for s in window if s.get('output_set') for line in s['output_set']]
        wf = torch.tensor([[
            len(set(tools)) / len(tools),
            len(set(fh)) / max(len(fh), 1) if fh else 1.0,
            len(set(ch)) / max(len(ch), 1) if ch else 1.0,
            sum(1 for s in window if s['is_error']) / len(window),
            sum(s['output_similarity'] for s in window) / len(window),
            len(set(al)) / max(len(al), 1) if al else 1.0,
        ]], dtype=torch.float32)

        with torch.no_grad():
            scores.append(torch.sigmoid(model(cat, cont, wf)).item())

    return scores, len(steps)


def find_session_file(sid, task_slug, claude_projects, worktree_base):
    """Derive the Claude project dir name from the worktree path convention."""
    candidates = []

    if worktree_base:
        # Claude project dir = worktree absolute path with '/' replaced by '-'
        worktree_path = os.path.join(worktree_base, task_slug)
        prefix = worktree_path.replace('/', '-')
        candidates.append(os.path.join(claude_projects, prefix, f'{sid}.jsonl'))

    # Fallback: scan claude_projects for dirs containing the task slug
    try:
        for d in os.listdir(claude_projects):
            if task_slug in d:
                candidates.append(os.path.join(claude_projects, d, f'{sid}.jsonl'))
    except OSError:
        pass

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main():
    results_dir = os.environ.get('BENCHMARK_RESULTS_DIR', '')
    if not results_dir or not os.path.isdir(results_dir):
        print("Set BENCHMARK_RESULTS_DIR to the directory containing off_1/, off_2/, heldout_off_1/")
        sys.exit(1)

    claude_projects = os.environ.get('CLAUDE_PROJECTS_DIR',
                                     os.path.expanduser('~/.claude/projects'))
    worktree_base = os.environ.get('WORKTREE_BASE', '')

    model, mean, std, threshold = load_model()
    print(f"Model loaded. Threshold: {threshold}")

    summary = {}
    for run in ['off_1', 'off_2', 'heldout_off_1']:
        run_dir = os.path.join(results_dir, run)
        if not os.path.isdir(run_dir):
            continue
        print(f"\n--- {run} ---")
        for fname in sorted(os.listdir(run_dir)):
            if not fname.endswith('.json'):
                continue
            task = fname.replace('.json', '')
            with open(os.path.join(run_dir, fname)) as fh:
                d = json.load(fh)
            sid = d.get('session_id', '')
            task_slug = task.replace('_', '-')

            session_file = find_session_file(sid, task_slug, claude_projects, worktree_base)
            if not session_file:
                print(f"  {task:25s} (session not found)")
                continue

            scores, n_steps = score_session(session_file, model, mean, std)
            if scores is None:
                print(f"  {task:25s} {n_steps:3d} steps (too short)")
                continue

            max_s = max(scores)
            fired = sum(1 for s in scores if s >= threshold)
            tag = ' <fired>' if fired > 0 else ''
            print(f"  {task:25s} {n_steps:3d}st  max={max_s:.3f} ({fired:2d}/{len(scores)}){tag}")
            summary[f'{run}/{task}'] = {'max': max_s, 'fired': fired, 'n_windows': len(scores)}

    out_path = os.path.join(MODEL_DIR, 'benchmark_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
