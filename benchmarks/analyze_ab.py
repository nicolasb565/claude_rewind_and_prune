#!/usr/bin/env python3
"""
Auto-discover paired OFF/ON benchmark runs under benchmarks/results/ and
report A/B stats — per-task durations, pass rates, paired t-test on
total/per-task runtime, and a rough power projection.

Discovery:
  - Scans `benchmarks/results/run_*/run.log` for the `proxy=off|on` header.
  - Pairs OFF/ON runs in run-number order. Excess on either side is dropped
    with a warning. (Run number, not chronological time, but in practice
    they're identical because run.sh assigns sequential IDs.)
  - --task filters to a single task ID (default: all tasks combined).

Usage:
  .venv/bin/python benchmarks/analyze_ab.py                     # all tasks
  .venv/bin/python benchmarks/analyze_ab.py --task 03_llvm_loop_vec
  .venv/bin/python benchmarks/analyze_ab.py --task 03_llvm_loop_vec --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "benchmarks" / "results"

# Critical t values at α=0.05 two-sided, df = n_pairs - 1 (paired test).
# Filled for df=1..30 plus wider entries so the 16–24 range doesn't round down
# to the df=20 or df=25 value (which would be anti-conservative).
T_CRIT_05 = {
    1: 12.71, 2: 4.30,  3: 3.18,  4: 2.78,  5: 2.57,
    6: 2.45,  7: 2.36,  8: 2.31,  9: 2.26,  10: 2.23,
    11: 2.20, 12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13,
    16: 2.12, 17: 2.11, 18: 2.10, 19: 2.09, 20: 2.09,
    21: 2.08, 22: 2.07, 23: 2.07, 24: 2.06, 25: 2.06,
    30: 2.04, 40: 2.02, 60: 2.00, 120: 1.98,
}


def t_crit(df: int) -> float:
    if df <= 0:
        return float("inf")
    if df in T_CRIT_05:
        return T_CRIT_05[df]
    # For df between table entries, return the next-larger listed df's
    # crit — which is SMALLER than the true crit (since crit decreases in df).
    # That's anti-conservative. We fix by returning the next-SMALLER key's
    # value, which is LARGER than the true crit (conservative).
    keys = sorted(T_CRIT_05)
    prev = keys[0]
    for k in keys:
        if k > df:
            return T_CRIT_05[prev]
        prev = k
    return T_CRIT_05[keys[-1]]


def discover_runs() -> tuple[list[Path], list[Path]]:
    """Return (off_runs, on_runs), each sorted by run number."""
    off, on = [], []
    if not RESULTS.exists():
        return off, on
    for d in sorted(RESULTS.glob("run_*")):
        log = d / "run.log"
        if not log.exists():
            continue
        # Header has e.g. `run_id=run_003 proxy=off auth=subscription ...`
        header = ""
        with open(log) as f:
            for line in f:
                if "proxy=" in line:
                    header = line
                    break
        if "proxy=off" in header:
            off.append(d)
        elif "proxy=on" in header:
            on.append(d)
    # Sort by run number embedded in the dir name (run_003 → 3)
    def key(p: Path) -> int:
        m = re.search(r"run_(\d+)$", p.name)
        return int(m.group(1)) if m else 0
    off.sort(key=key)
    on.sort(key=key)
    return off, on


def load_task(run_dir: Path, task: str) -> dict | None:
    p = run_dir / task / "summary_1.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_verify(run_dir: Path, task: str) -> int | None:
    """Return verify_exit if verify_1.json exists, else None."""
    p = run_dir / task / "verify_1.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("verify_exit")
    except (json.JSONDecodeError, OSError):
        return None


def discover_tasks(runs: list[Path]) -> list[str]:
    """Tasks present across the union of all runs."""
    tasks: set[str] = set()
    for r in runs:
        for d in r.iterdir():
            if d.is_dir() and (d / "summary_1.json").exists():
                tasks.add(d.name)
    return sorted(tasks)


def fmt_seconds(s: float | int | None) -> str:
    if s is None:
        return "  —  "
    return f"{int(s):>5}"


def pair_runs(off: list[Path], on: list[Path], task: str) -> list[tuple[Path, Path]]:
    """Pair adjacent OFF→ON runs by run number.

    `run_ab.sh` always produces consecutive (OFF, ON) pairs (run_N, run_N+1),
    so the right pairing is: walk all runs in run-number order, and any time
    an OFF is immediately followed by an ON, that's a pair. Anything else is
    dropped (orphan OFF, leading or trailing ON, double-OFF from a smoke
    test, etc.). This is robust against:
      - older runs from prior code versions sitting in results/
      - manual one-off runs that aren't part of an A/B
      - the rbtree-only smoke test that produced an unpaired OFF
    """
    def num(p: Path) -> int:
        m = re.search(r"run_(\d+)$", p.name)
        return int(m.group(1)) if m else 0

    off_set = {r.name for r in off if (r / task / "summary_1.json").exists()}
    on_set = {r.name for r in on if (r / task / "summary_1.json").exists()}
    all_runs = sorted(
        [r for r in off + on if (r / task / "summary_1.json").exists()],
        key=num,
    )

    pairs: list[tuple[Path, Path]] = []
    skipped_off: list[str] = []
    skipped_on: list[str] = []

    i = 0
    while i < len(all_runs):
        cur = all_runs[i]
        is_off = cur.name in off_set
        if is_off and i + 1 < len(all_runs):
            nxt = all_runs[i + 1]
            if nxt.name in on_set and num(nxt) == num(cur) + 1:
                pairs.append((cur, nxt))
                i += 2
                continue
            skipped_off.append(cur.name)
            i += 1
        elif is_off:
            skipped_off.append(cur.name)
            i += 1
        else:
            skipped_on.append(cur.name)
            i += 1

    if skipped_off or skipped_on:
        if skipped_off:
            print(f"  note: unpaired OFF runs skipped: {skipped_off}")
        if skipped_on:
            print(f"  note: unpaired ON runs skipped: {skipped_on}")

    return pairs


def stats_for_task(task: str, pairs: list[tuple[Path, Path]], verbose: bool) -> dict:
    """Compute paired stats for one task."""
    rows = []
    for off_dir, on_dir in pairs:
        off_s = load_task(off_dir, task)
        on_s = load_task(on_dir, task)
        if off_s is None or on_s is None:
            continue
        off_d = off_s.get("duration_seconds")
        on_d = on_s.get("duration_seconds")
        off_e = off_s.get("exit_code")
        on_e = on_s.get("exit_code")
        off_v = load_verify(off_dir, task)
        on_v = load_verify(on_dir, task)
        rows.append({
            "off_dir": off_dir.name,
            "on_dir": on_dir.name,
            "off_dur": off_d,
            "on_dur": on_d,
            "off_exit": off_e,
            "on_exit": on_e,
            "off_verify": off_v,
            "on_verify": on_v,
            "diff": (on_d - off_d) if (off_d is not None and on_d is not None) else None,
        })

    if not rows:
        return {"task": task, "n": 0, "rows": rows}

    diffs = [r["diff"] for r in rows if r["diff"] is not None]
    off_durs = [r["off_dur"] for r in rows if r["off_dur"] is not None]
    on_durs = [r["on_dur"] for r in rows if r["on_dur"] is not None]
    off_pass = sum(1 for r in rows if r["off_exit"] == 0)
    on_pass = sum(1 for r in rows if r["on_exit"] == 0)
    off_verify_pass = sum(1 for r in rows if r["off_verify"] == 0)
    on_verify_pass = sum(1 for r in rows if r["on_verify"] == 0)
    has_verify = any(r["off_verify"] is not None for r in rows)

    n = len(diffs)
    out = {
        "task": task,
        "n": n,
        "rows": rows,
        "off_pass": off_pass,
        "on_pass": on_pass,
        "off_verify_pass": off_verify_pass if has_verify else None,
        "on_verify_pass": on_verify_pass if has_verify else None,
        "off_durs": off_durs,
        "on_durs": on_durs,
        "diffs": diffs,
        "off_total": sum(off_durs),
        "on_total": sum(on_durs),
        "off_median": statistics.median(off_durs) if off_durs else None,
        "on_median": statistics.median(on_durs) if on_durs else None,
    }

    if n >= 2:
        mean_diff = statistics.mean(diffs)
        sd_diff = statistics.stdev(diffs)
        sem = sd_diff / math.sqrt(n)
        t = mean_diff / sem if sem > 0 else 0
        cohen_d = mean_diff / sd_diff if sd_diff > 0 else 0
        crit = t_crit(n - 1)
        out["mean_diff"] = mean_diff
        out["sd_diff"] = sd_diff
        out["t"] = t
        out["cohen_d"] = cohen_d
        out["t_crit_05"] = crit
        out["significant_05"] = abs(t) > crit

        # Power projection: roughly how many pairs to reach 80% power
        # at α=0.05 (two-sided), assuming current effect size persists.
        # Approximation: n_needed ≈ ((z_α + z_β) / d)^2 for paired test
        # where z_α = 1.96, z_β (80% power) = 0.84
        if abs(cohen_d) > 0.05:
            out["n_for_80pct_power"] = max(1, math.ceil(((1.96 + 0.84) / abs(cohen_d)) ** 2))
        else:
            out["n_for_80pct_power"] = None

    return out


def print_task_summary(s: dict, verbose: bool) -> None:
    task = s["task"]
    n = s["n"]
    if n == 0:
        print(f"\n{task}: no paired data")
        return
    print(f"\n{task}  (n={n} pairs)")
    print(f"  pass:    OFF {s['off_pass']}/{n}   ON {s['on_pass']}/{n}")
    if s.get("off_verify_pass") is not None:
        print(f"  verify:  OFF {s['off_verify_pass']}/{n}   ON {s['on_verify_pass']}/{n}")
    print(f"  totals:  OFF {s['off_total']:>6}s   ON {s['on_total']:>6}s   "
          f"Δ {s['on_total']-s['off_total']:+6}s")
    if s["off_median"] is not None and s["on_median"] is not None:
        print(f"  median:  OFF {s['off_median']:>6.0f}s  ON {s['on_median']:>6.0f}s   "
              f"Δ {s['on_median']-s['off_median']:+6.0f}s")
    else:
        print(f"  median:  (one side has no duration data — skipped)")
    if "mean_diff" in s:
        print(f"  paired:  mean Δ = {s['mean_diff']:+.0f}s  sd Δ = {s['sd_diff']:.0f}s  "
              f"t = {s['t']:+.2f}  d = {s['cohen_d']:+.2f}")
        sig = "**SIGNIFICANT (α=0.05)**" if s["significant_05"] else "not significant (α=0.05)"
        print(f"           {sig}    t_crit = {s['t_crit_05']:.2f}")
        if s["n_for_80pct_power"]:
            need = s["n_for_80pct_power"]
            extra = max(0, need - n)
            print(f"           ~n={need} pairs for 80% power at this effect size "
                  f"({extra} more after current {n})")
    if verbose:
        print(f"  per-pair durations:")
        for r in s["rows"]:
            v_off = "" if r["off_verify"] is None else f" v={r['off_verify']}"
            v_on  = "" if r["on_verify"]  is None else f" v={r['on_verify']}"
            print(f"    {r['off_dir']:<10} OFF {fmt_seconds(r['off_dur'])}s "
                  f"e={r['off_exit']}{v_off}  ↔  "
                  f"{r['on_dir']:<10} ON {fmt_seconds(r['on_dur'])}s "
                  f"e={r['on_exit']}{v_on}  Δ {r['diff']:+5}s"
                  if r['diff'] is not None else
                  f"    {r['off_dir']:<10} OFF {fmt_seconds(r['off_dur'])}s  "
                  f"{r['on_dir']:<10} ON {fmt_seconds(r['on_dur'])}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="single task id; default = all tasks")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="print per-pair durations")
    args = ap.parse_args()

    off, on = discover_runs()
    print(f"=== Discovered runs ===")
    print(f"  OFF: {len(off)}  →  {[r.name for r in off]}")
    print(f"  ON:  {len(on)}  →  {[r.name for r in on]}")

    if not off or not on:
        print("\nNothing to compare.")
        return 1

    tasks = [args.task] if args.task else discover_tasks(off + on)

    summaries = []
    for task in tasks:
        pairs = pair_runs(off, on, task)
        if not pairs:
            continue
        s = stats_for_task(task, pairs, args.verbose)
        summaries.append(s)
        print_task_summary(s, args.verbose)

    # Pooled stats across tasks (only meaningful if --task wasn't set, but
    # always show — it's the obvious "headline" number)
    print("\n" + "=" * 70)
    print("=== Pooled across reported tasks ===")
    all_off = sum(s["off_total"] for s in summaries if s["n"] > 0)
    all_on = sum(s["on_total"] for s in summaries if s["n"] > 0)
    all_off_pass = sum(s["off_pass"] for s in summaries if s["n"] > 0)
    all_on_pass = sum(s["on_pass"] for s in summaries if s["n"] > 0)
    all_n = sum(s["n"] for s in summaries if s["n"] > 0)
    print(f"  total seconds   OFF {all_off}s   ON {all_on}s   "
          f"Δ {all_on-all_off:+}s ({100*(all_on-all_off)/max(all_off,1):+.1f}%)")
    print(f"  task-runs pass  OFF {all_off_pass}/{all_n}   ON {all_on_pass}/{all_n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
