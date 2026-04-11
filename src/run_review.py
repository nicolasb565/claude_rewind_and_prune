"""Run LLM review agents on pending batch files.

Reads items from data/review/batches/ (Sonnet) or data/review/escalated/ (Opus),
submits them in sub-batches of SUBBATCH_SIZE windows per API call, with up to
MAX_WORKERS concurrent calls. Writes results to data/review/results/{model}/.

Uses `claude -p` (Claude Code CLI) rather than the Anthropic Python SDK, so no
API key or extra dependencies are required beyond a logged-in claude install.

Usage:
  python src/run_review.py sonnet [<source>]   # review pending Sonnet batches
  python src/run_review.py opus   [<source>]   # review pending Opus batches
  python src/run_review.py opus   --sample N   # only process first N items (for sampling)
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

BATCHES_DIR  = 'data/review/batches'
ESCALATE_DIR = 'data/review/escalated'
RESULTS_BASE = 'data/review/results'

SUBBATCH_SIZE = 15
MAX_WORKERS   = 10

SONNET_MODEL  = 'sonnet'
OPUS_MODEL    = 'opus'

SYSTEM_PROMPT = """You are reviewing windows of Claude Code tool-call sequences to determine
whether the agent is stuck in a loop or making genuine progress.

Each window is 10 consecutive tool calls. For each window you will be shown:
- The sequence of tool calls with commands and output snippets
- Heuristic signals (tight_loop_steps, diverse_steps, error_steps)

Classify each window as exactly one of:
  STUCK      — agent is repeating the same actions with no meaningful progress
               (same commands cycling, same errors repeating, no new information)
  PRODUCTIVE — agent is making genuine progress toward the task
               (exploring new areas, iterating toward a solution, error recovery)
  UNCLEAR    — genuinely ambiguous; you cannot confidently assign STUCK or PRODUCTIVE

Respond with a JSON array, one object per window, in the same order:
[
  {"id": "<window_id>", "label": "STUCK|PRODUCTIVE|UNCLEAR", "reason": "<one sentence>"},
  ...
]
Output ONLY the JSON array, no other text."""


def format_window(item):
    lines = [f"Window ID: {item['id']}"]
    p = item.get('precomputed', {})
    lines.append(
        f"Signals: tight_loop={p.get('tight_loop_steps',0)} "
        f"diverse={p.get('diverse_steps',0)} "
        f"errors={p.get('error_steps',0)}"
    )
    lines.append("Steps:")
    for i, s in enumerate(item['steps']):
        parts = [f"  [{i+1}] {s['tool']}"]
        if s.get('cmd'):
            parts.append(f"    cmd: {s['cmd'][:150]}")
        if s.get('file'):
            parts.append(f"    file: {s['file']}")
        if s.get('output_snippet'):
            snippet = s['output_snippet'][:300]
            parts.append(f"    output: {snippet}")
        if s.get('error'):
            parts.append(f"    ERROR")
        lines.extend(parts)
    return '\n'.join(lines)


def parse_response(text, items):
    """Extract JSON array from response, return list of result dicts."""
    # Strip markdown fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text.strip(), flags=re.MULTILINE)
    try:
        results = json.loads(text)
        if isinstance(results, list):
            return results
    except json.JSONDecodeError:
        pass

    # Fallback: try to find JSON array in the text
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Last resort: return UNCLEAR for all items
    print(f"  WARNING: could not parse response, marking {len(items)} items UNCLEAR")
    return [{'id': it['id'], 'label': 'UNCLEAR', 'reason': 'parse error'} for it in items]


async def review_subbatch(items, model, semaphore):
    async with semaphore:
        prompt = '\n\n---\n\n'.join(format_window(it) for it in items)
        try:
            proc = await asyncio.create_subprocess_exec(
                'claude', '-p', prompt,
                '--model', model,
                '--system-prompt', SYSTEM_PROMPT,
                '--output-format', 'json',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(stderr.decode().strip() or stdout.decode()[:200])
            # --output-format json wraps response in {"result": "..."}
            outer = json.loads(stdout.decode())
            text = outer.get('result', stdout.decode())
            results = parse_response(text, items)
            result_map = {r['id']: r for r in results}
            return [result_map.get(it['id'], {'id': it['id'], 'label': 'UNCLEAR', 'reason': 'missing from response'})
                    for it in items]
        except Exception as e:
            print(f"  CLI error: {e}")
            return [{'id': it['id'], 'label': 'UNCLEAR', 'reason': f'cli error: {e}'} for it in items]


async def process_batch_file(batch_path, out_path, model, semaphore):
    items = []
    with open(batch_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if not items:
        return []

    # Split into sub-batches of SUBBATCH_SIZE
    subbatches = [items[i:i+SUBBATCH_SIZE] for i in range(0, len(items), SUBBATCH_SIZE)]
    tasks = [review_subbatch(sb, model, semaphore) for sb in subbatches]
    results_nested = await asyncio.gather(*tasks)
    results = [r for sublist in results_nested for r in sublist]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    return results


async def run(mode, source_filter=None, sample_n=None):
    if mode == 'sonnet':
        in_dir   = BATCHES_DIR
        model    = SONNET_MODEL
        out_dir  = os.path.join(RESULTS_BASE, 'sonnet')
    else:
        in_dir   = ESCALATE_DIR
        model    = OPUS_MODEL
        out_dir  = os.path.join(RESULTS_BASE, 'opus')

    if not os.path.isdir(in_dir):
        print(f"No input directory: {in_dir}")
        return

    batch_files = sorted(
        f for f in os.listdir(in_dir)
        if f.endswith('.jsonl')
        and (not source_filter or f.startswith(f'{source_filter}_'))
    )
    if not batch_files:
        print(f"No batch files found in {in_dir}/")
        return

    # For sampling: collect items up to sample_n total
    if sample_n:
        sampled = []
        for fname in batch_files:
            with open(os.path.join(in_dir, fname)) as f:
                for line in f:
                    if line.strip():
                        sampled.append((fname, json.loads(line)))
                        if len(sampled) >= sample_n:
                            break
            if len(sampled) >= sample_n:
                break
        # Group back by file
        by_file = {}
        for fname, item in sampled:
            by_file.setdefault(fname, []).append(item)
        batch_files = list(by_file.keys())
        # Write temp files for sampled items
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp()
        new_batch_files = []
        for fname, items in by_file.items():
            tmp = os.path.join(tmpdir, fname)
            with open(tmp, 'w') as f:
                for item in items:
                    f.write(json.dumps(item) + '\n')
            new_batch_files.append((fname, tmp))
        print(f"Sampling {len(sampled)} items from {len(new_batch_files)} files")
    else:
        new_batch_files = [(f, os.path.join(in_dir, f)) for f in batch_files]

    total_items = sum(
        sum(1 for l in open(path) if l.strip())
        for _, path in new_batch_files
    )
    print(f"Running {mode} review on {len(new_batch_files)} batch files ({total_items} items)")
    print(f"Model: {model}  sub-batch: {SUBBATCH_SIZE}  workers: {MAX_WORKERS}")

    semaphore = asyncio.Semaphore(MAX_WORKERS)
    tasks = []
    for fname, path in new_batch_files:
        out_path = os.path.join(out_dir, fname)
        tasks.append(process_batch_file(path, out_path, model, semaphore))

    all_results = await asyncio.gather(*tasks)

    # Summary
    from collections import Counter
    counts = Counter()
    for results in all_results:
        for r in results:
            counts[r.get('label', '?').upper()] += 1

    print(f"\nResults: {dict(counts)}")
    print(f"Total: {sum(counts.values())}")
    if sample_n:
        unclear_pct = 100 * counts.get('UNCLEAR', 0) / max(sum(counts.values()), 1)
        print(f"UNCLEAR rate: {unclear_pct:.1f}%")
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('sonnet', 'opus'):
        print("Usage: python src/run_review.py sonnet|opus [<source>] [--sample N]")
        sys.exit(1)

    mode = sys.argv[1]
    source_filter = None
    sample_n = None

    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == '--sample' and i + 1 < len(args):
            sample_n = int(args[i+1])
            i += 2
        else:
            source_filter = args[i]
            i += 1

    asyncio.run(run(mode, source_filter, sample_n))


if __name__ == '__main__':
    main()
