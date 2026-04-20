#!/usr/bin/env python3
"""
Analyze cache invalidation cost of the context-hygiene proxy.

Parses `proxy_logs/events-*.jsonl` under benchmarks/results/run_*/ to
correlate `compact` events (history rewrites) with subsequent
`cache_stats` events (upstream usage reports).

Primary questions this answers:
  1. Did total cost go up or down? Compute weighted input cost =
       input_tokens * 1.0 + cache_creation_input_tokens * 1.25
                          + cache_read_input_tokens * 0.1
     per the Anthropic pricing ratios, per run.
  2. Did cache hit rate drop? Compute
       cache_read / (input + cache_creation + cache_read)
     for ON vs OFF runs.
  3. Do `compact` events immediately cause a cache_creation spike on
     the next request? For each compact, look at the delta in
     cache_creation between the request before and the request after.

Usage:
  .venv/bin/python benchmarks/analyze_cache.py              # all paired runs
  .venv/bin/python benchmarks/analyze_cache.py --task 08_express_async
  .venv/bin/python benchmarks/analyze_cache.py --run-dir results/run_026
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "benchmarks" / "results"

# Anthropic pricing ratios (relative to base input). Base = 1.0; cache
# writes are 1.25x, cache reads are 0.1x. We use these to compute a
# single "weighted input cost" number that is comparable across runs.
W_INPUT = 1.0
W_CREATE = 1.25
W_READ = 0.1


@dataclass
class RunStats:
    run_dir: Path
    proxy: str  # "on" | "off"
    input_tokens: int = 0
    cache_creation: int = 0
    cache_read: int = 0
    output_tokens: int = 0
    n_requests: int = 0
    n_compact_events: int = 0
    compact_tokens_saved_est: int = 0
    # For spike analysis: ordered cache_creation values per cache_stats event.
    creation_series: list[int] = field(default_factory=list)
    # Index (into creation_series) of the first cache_stats after each compact.
    compact_followup_idx: list[int] = field(default_factory=list)

    @property
    def weighted_cost(self) -> float:
        return (
            self.input_tokens * W_INPUT
            + self.cache_creation * W_CREATE
            + self.cache_read * W_READ
        )

    @property
    def cache_hit_rate(self) -> float:
        denom = self.input_tokens + self.cache_creation + self.cache_read
        return self.cache_read / denom if denom else 0.0


def read_events(proxy_log_dir: Path) -> list[dict]:
    events: list[dict] = []
    if not proxy_log_dir.exists():
        return events
    for f in sorted(proxy_log_dir.glob("events-*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def analyze_run(run_dir: Path) -> RunStats:
    # Infer proxy state from run.log header.
    proxy = "off"
    log = run_dir / "run.log"
    if log.exists():
        with open(log) as fh:
            for line in fh:
                if "proxy=" in line:
                    if "proxy=on" in line:
                        proxy = "on"
                    break
    stats = RunStats(run_dir=run_dir, proxy=proxy)

    events = read_events(run_dir / "proxy_logs")
    pending_compact = False
    for ev in events:
        t = ev.get("type")
        if t == "compact":
            stats.n_compact_events += 1
            stats.compact_tokens_saved_est += ev.get("tokensSavedEstimate", 0) or 0
            pending_compact = True
        elif t == "cache_stats":
            stats.n_requests += 1
            stats.input_tokens += ev.get("input_tokens") or 0
            stats.cache_creation += ev.get("cache_creation_input_tokens") or 0
            stats.cache_read += ev.get("cache_read_input_tokens") or 0
            stats.output_tokens += ev.get("output_tokens") or 0
            stats.creation_series.append(ev.get("cache_creation_input_tokens") or 0)
            if pending_compact:
                stats.compact_followup_idx.append(len(stats.creation_series) - 1)
                pending_compact = False

    return stats


def discover_runs(filter_task: str | None) -> list[RunStats]:
    runs: list[RunStats] = []
    if not RESULTS.exists():
        return runs
    for d in sorted(RESULTS.glob("run_*")):
        log = d / "run.log"
        if not log.exists():
            continue
        if filter_task and not (d / filter_task).exists():
            continue
        runs.append(analyze_run(d))
    return runs


def pair_runs(runs: list[RunStats]) -> list[tuple[RunStats, RunStats]]:
    """Pair consecutive OFF/ON runs in run-number order."""
    off = [r for r in runs if r.proxy == "off"]
    on = [r for r in runs if r.proxy == "on"]
    n = min(len(off), len(on))
    if len(off) != len(on):
        print(f"warning: unequal runs (off={len(off)} on={len(on)}), using first {n} pairs")
    return list(zip(off[:n], on[:n]))


def fmt_k(n: int | float) -> str:
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:.0f}"


def creation_spike_delta(stats: RunStats) -> float | None:
    """Median cache_creation increase on requests following a compact event,
    relative to the overall median. Returns None when there isn't enough data.
    """
    if not stats.compact_followup_idx or len(stats.creation_series) < 3:
        return None
    baseline = statistics.median(stats.creation_series)
    post = [stats.creation_series[i] for i in stats.compact_followup_idx]
    if not post:
        return None
    return statistics.median(post) - baseline


def report(pairs: list[tuple[RunStats, RunStats]]) -> None:
    if not pairs:
        print("no paired runs found under benchmarks/results/")
        return

    print(f"# Cache analysis — {len(pairs)} pair(s)\n")
    print(
        f"{'pair':<6}{'run':<14}{'proxy':<6}"
        f"{'requests':>10}{'input':>10}{'creation':>12}"
        f"{'read':>12}{'output':>10}{'weighted':>12}{'hit%':>8}"
    )
    print("-" * 100)
    for i, (o, n) in enumerate(pairs, 1):
        for side, s in (("off", o), ("on", n)):
            print(
                f"{i:<6}{s.run_dir.name:<14}{side:<6}"
                f"{s.n_requests:>10}"
                f"{fmt_k(s.input_tokens):>10}"
                f"{fmt_k(s.cache_creation):>12}"
                f"{fmt_k(s.cache_read):>12}"
                f"{fmt_k(s.output_tokens):>10}"
                f"{fmt_k(s.weighted_cost):>12}"
                f"{s.cache_hit_rate * 100:>7.1f}%"
            )
        delta = n.weighted_cost - o.weighted_cost
        pct = (delta / o.weighted_cost * 100) if o.weighted_cost else 0.0
        print(f"     Δ weighted = {fmt_k(delta)} ({pct:+.1f}%)    "
              f"compact events on ON: {n.n_compact_events}    "
              f"est tokens saved: {fmt_k(n.compact_tokens_saved_est)}")
        spike = creation_spike_delta(n)
        if spike is not None:
            print(f"     post-compact cache_creation median Δ vs run median: "
                  f"{fmt_k(spike)}")
        print()

    # Aggregate
    print("-- aggregate --")
    off_cost = sum(o.weighted_cost for o, _ in pairs)
    on_cost = sum(n.weighted_cost for _, n in pairs)
    off_hits = sum(o.cache_read for o, _ in pairs)
    on_hits = sum(n.cache_read for _, n in pairs)
    off_denom = sum(o.cache_read + o.cache_creation + o.input_tokens for o, _ in pairs)
    on_denom = sum(n.cache_read + n.cache_creation + n.input_tokens for _, n in pairs)
    print(f"  weighted cost:  off={fmt_k(off_cost)}  on={fmt_k(on_cost)}  "
          f"Δ={fmt_k(on_cost - off_cost)} "
          f"({(on_cost - off_cost) / off_cost * 100 if off_cost else 0:+.1f}%)")
    print(f"  cache hit rate: off={off_hits / off_denom * 100 if off_denom else 0:.1f}%  "
          f"on={on_hits / on_denom * 100 if on_denom else 0:.1f}%")
    total_saved = sum(n.compact_tokens_saved_est for _, n in pairs)
    total_compacts = sum(n.n_compact_events for _, n in pairs)
    print(f"  compact events: {total_compacts}   est tokens saved (pre-cache): "
          f"{fmt_k(total_saved)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", help="filter to runs that touched this task id")
    ap.add_argument("--run-dir", help="analyze a single run dir (no pairing)")
    args = ap.parse_args()

    if args.run_dir:
        rd = Path(args.run_dir)
        if not rd.is_absolute():
            rd = REPO / rd
        s = analyze_run(rd)
        print(f"{s.run_dir.name} proxy={s.proxy}")
        print(f"  requests: {s.n_requests}")
        print(f"  input_tokens: {fmt_k(s.input_tokens)}")
        print(f"  cache_creation: {fmt_k(s.cache_creation)}")
        print(f"  cache_read: {fmt_k(s.cache_read)}")
        print(f"  output_tokens: {fmt_k(s.output_tokens)}")
        print(f"  weighted cost: {fmt_k(s.weighted_cost)}")
        print(f"  cache hit rate: {s.cache_hit_rate * 100:.1f}%")
        print(f"  compact events: {s.n_compact_events}")
        print(f"  est tokens saved (pre-cache): {fmt_k(s.compact_tokens_saved_est)}")
        spike = creation_spike_delta(s)
        if spike is not None:
            print(f"  post-compact cache_creation median Δ vs run median: {fmt_k(spike)}")
        return

    runs = discover_runs(args.task)
    pairs = pair_runs(runs)
    report(pairs)


if __name__ == "__main__":
    main()
