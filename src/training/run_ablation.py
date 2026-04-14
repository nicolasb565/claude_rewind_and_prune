"""Run the feature ablation matrix across multiple seeds.

Each (variant, seed) combination trains a fresh model and writes outputs to
proxy/experiments/ablation_multiseed/<variant>/seed_<N>/.

Usage:
  python src/training/run_ablation.py [--seeds 5]
"""

import argparse
import os
import sys

# Make the package importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.training.train import MODEL_DIR, train  # noqa: E402

VARIANTS: list[tuple[str, set[str]]] = [
    ("baseline", set()),
    ("no_file_hash", {"file_hash"}),
    ("no_step_idx", {"step_index_norm"}),
    ("no_has_prior", {"has_prior_output"}),
    ("no_output_length", {"output_length"}),
    ("no_is_error", {"is_error"}),
    ("minimal", {"file_hash", "step_index_norm", "has_prior_output"}),
    ("aggressive_min", {"file_hash", "step_index_norm", "has_prior_output", "output_length"}),
    # Stage 2: confirm dropping has_prior_output on top of no_step_idx
    ("no_step_idx_no_has_prior", {"step_index_norm", "has_prior_output"}),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds per variant (default: 5)")
    parser.add_argument("--start-seed", type=int, default=42, help="First seed value (default: 42)")
    parser.add_argument("--manifest", default="training_manifest.json")
    args = parser.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    total = len(VARIANTS) * len(seeds)
    done = 0

    for variant_name, excluded in VARIANTS:
        for seed in seeds:
            done += 1
            out_dir = os.path.join(
                MODEL_DIR, "experiments", "ablation_multiseed", variant_name, f"seed_{seed}"
            )
            print(f"\n[{done}/{total}] {variant_name} (seed={seed}) → {out_dir}")
            if os.path.exists(os.path.join(out_dir, "stuck_config.json")):
                print("  already exists — skipping")
                continue
            train(
                manifest_path=args.manifest,
                use_score_history=False,
                excluded_features=excluded,
                output_dir=out_dir,
                seed=seed,
            )

    print(f"\nDone — {total} runs written under {MODEL_DIR}/experiments/ablation_multiseed/")


if __name__ == "__main__":
    main()
