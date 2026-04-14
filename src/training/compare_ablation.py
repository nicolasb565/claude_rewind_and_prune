"""Aggregate stuck_config.json files across seeds and print a comparison table.

Usage:
  python src/training/compare_ablation.py
"""

import glob
import json
import os
import statistics
from collections import defaultdict


def main() -> None:
    base = "proxy/experiments/ablation_multiseed"
    if not os.path.isdir(base):
        # Fall back to single-seed layout
        single = "proxy/experiments/ablation"
        if os.path.isdir(single):
            print("[single-seed mode — no multiseed runs found]")
            for cfg_path in sorted(glob.glob(os.path.join(single, "*", "stuck_config.json"))):
                variant = os.path.basename(os.path.dirname(cfg_path))
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                m = cfg.get("metrics", {})
                print(
                    f"{variant:<25}  feats={cfg.get('num_features')}  "
                    f"F1={m.get('f1', 0):.4f}  P={m.get('precision', 0):.4f}  "
                    f"R={m.get('recall', 0):.4f}"
                )
        else:
            print("No ablation runs found")
        return

    # Collect: variant → list of metrics dicts (one per seed)
    by_variant: dict[str, list[dict]] = defaultdict(list)
    by_variant_meta: dict[str, dict] = {}

    for cfg_path in sorted(glob.glob(os.path.join(base, "*", "seed_*", "stuck_config.json"))):
        seed_dir = os.path.dirname(cfg_path)
        variant_dir = os.path.dirname(seed_dir)
        variant = os.path.basename(variant_dir)
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        by_variant[variant].append(cfg["metrics"])
        by_variant_meta[variant] = {
            "num_features": cfg.get("num_features"),
            "input_dim": cfg.get("input_dim"),
            "total_params": cfg.get("total_params"),
            "excluded": cfg.get("excluded_features", []),
        }

    if not by_variant:
        print(f"No runs found under {base}")
        return

    def agg(values: list[float]) -> tuple[float, float]:
        if len(values) < 2:
            return values[0] if values else 0.0, 0.0
        return statistics.mean(values), statistics.stdev(values)

    # Determine baseline (for delta column)
    baseline_metrics = by_variant.get("baseline", [])
    baseline_f1_mean, _ = agg([m["f1"] for m in baseline_metrics]) if baseline_metrics else (0, 0)

    # Print
    n_seeds = max(len(v) for v in by_variant.values())
    print(f"\nAblation results across {n_seeds} seeds (mean ± std)")
    print()
    header = (
        f"{'variant':<20}  {'feats':>5}  {'dim':>4}  {'params':>6}  "
        f"{'F1 mean ± std':>17}  {'Δ F1':>8}  "
        f"{'P mean ± std':>17}  {'R mean ± std':>17}  {'n':>3}"
    )
    print(header)
    print("-" * len(header))

    # Sort: baseline first, then by F1 desc
    def sort_key(item: tuple[str, list[dict]]) -> tuple[int, float]:
        name, ms = item
        if name == "baseline":
            return (0, 0)
        return (1, -agg([m["f1"] for m in ms])[0])

    for variant, metrics_list in sorted(by_variant.items(), key=sort_key):
        n = len(metrics_list)
        meta = by_variant_meta[variant]
        f1s = [m["f1"] for m in metrics_list]
        ps = [m["precision"] for m in metrics_list]
        rs = [m["recall"] for m in metrics_list]
        f1_m, f1_s = agg(f1s)
        p_m, p_s = agg(ps)
        r_m, r_s = agg(rs)
        delta = f1_m - baseline_f1_mean if variant != "baseline" else 0.0
        delta_str = f"{delta:+.4f}" if variant != "baseline" else "  —    "
        print(
            f"{variant:<20}  {meta['num_features']:>5}  {meta['input_dim']:>4}  "
            f"{meta['total_params']:>6}  "
            f"{f1_m:>7.4f} ± {f1_s:>6.4f}  {delta_str:>8}  "
            f"{p_m:>7.4f} ± {p_s:>6.4f}  {r_m:>7.4f} ± {r_s:>6.4f}  {n:>3}"
        )


if __name__ == "__main__":
    main()
