"""Analyze stuck loop lengths across all labeled training data.

Usage:
  python analyze_stuck_loops.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SOURCES = [
    "data/generated/nlile_v2.jsonl",
    "data/generated/dataclaw_claude_v2.jsonl",
    "data/generated/masterclass_v2.jsonl",
    "data/generated/claudeset_v2.jsonl",
]


def load_sessions(paths):
    """Load all rows grouped by session_id, preserving step order."""
    sessions = defaultdict(list)
    for path in paths:
        p = Path(path)
        if not p.exists():
            print(f"  skipping {path} (not found)")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                sessions[row["session_id"]].append(row)
    # Sort each session by step index
    for sid in sessions:
        sessions[sid].sort(key=lambda r: r["step"])
    return sessions


def extract_run_lengths(sessions):
    """Extract lengths of consecutive STUCK runs per session."""
    stuck_runs = []
    productive_runs = []

    for rows in sessions.values():
        labels = [r["label"] for r in rows]
        # Walk through labels counting consecutive runs
        i = 0
        while i < len(labels):
            val = labels[i]
            j = i
            while j < len(labels) and labels[j] == val:
                j += 1
            run_len = j - i
            if val >= 0.9:  # STUCK (1.0) or UNSURE (0.5 filtered out at 0.9)
                stuck_runs.append(run_len)
            else:
                productive_runs.append(run_len)
            i = j

    return stuck_runs, productive_runs


def print_distribution(name, runs):
    arr = np.array(runs)
    pcts = [50, 75, 90, 95, 99, 100]
    print(f"\n{name} run lengths (n={len(arr)} runs, {arr.sum()} steps total):")
    print("  " + "  ".join(f"p{p}={int(np.percentile(arr, p))}" for p in pcts))
    print(f"  mean={arr.mean():.1f}  std={arr.std():.1f}")

    # Histogram of run lengths
    max_show = min(int(np.percentile(arr, 99)), 20)
    counts = np.bincount(arr, minlength=max_show + 2)
    print(f"  Length distribution (1–{max_show}+):")
    for length in range(1, max_show + 1):
        n = counts[length] if length < len(counts) else 0
        bar = "█" * min(int(n / max(counts[1:]) * 40), 40)
        print(f"    {length:3d}: {n:6d}  {bar}")
    gt = arr[arr > max_show].shape[0]
    if gt:
        print(f"    {max_show+1}+: {gt:6d}")


def main():
    print("Loading sessions...")
    sessions = load_sessions(SOURCES)
    print(f"  {len(sessions)} sessions loaded")

    stuck_runs, productive_runs = extract_run_lengths(sessions)

    print_distribution("STUCK", stuck_runs)
    print_distribution("PRODUCTIVE", productive_runs)

    stuck_arr = np.array(stuck_runs)
    print(f"\n=== N recommendation ===")
    for pct in [75, 90, 95]:
        n = int(np.percentile(stuck_arr, pct))
        print(f"  N={n} covers {pct}% of stuck runs")


if __name__ == "__main__":
    main()
