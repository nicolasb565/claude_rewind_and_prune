#!/usr/bin/env python3
"""Benchmark runner — load model once, run all fixtures × [baseline, hygiene].

Each fixture is a directory under harness/fixtures/ containing:
- GOAL.md                — task description for the agent
- src/, tests/, etc.     — the working code
- (optional) VALIDATE.sh — shell script that exits 0 if task succeeded, else 1
  If missing, we default to `pytest tests/ -x`.

For each run we copy the fixture to a scratch dir first so repeated runs
start from a clean slate.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def load_goal(fixture_dir: Path) -> str:
    goal_file = fixture_dir / "GOAL.md"
    return goal_file.read_text().strip()


def validate(work_dir: Path) -> tuple[bool, str]:
    """Check whether the task is complete. Default: pytest must pass."""
    vs = work_dir / "VALIDATE.sh"
    if vs.exists():
        cmd = ["bash", str(vs)]
    else:
        cmd = ["python", "-m", "pytest", "tests/", "-x", "-q"]
    try:
        r = subprocess.run(cmd, cwd=str(work_dir), capture_output=True,
                           text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False, "validation timed out"
    return r.returncode == 0, (r.stdout + r.stderr)[-500:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-dir", type=Path,
                    default=Path("harness/fixtures"))
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--adapter", type=Path,
                    default=Path("proxy/experiments/qwen35_2b_hygiene_v13b/final"))
    ap.add_argument("--no-adapter", action="store_true",
                    help="Run with base model only, no LoRA")
    ap.add_argument("--modes", nargs="+", default=["baseline", "hygiene"])
    ap.add_argument("--fixture", default=None,
                    help="Run only this single fixture name")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--max-context-tokens", type=int, default=8000)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy decoding (deterministic). >0 enables sampling.")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--out", type=Path,
                    default=Path("harness/results.jsonl"))
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness.agent import run_agent
    from src.pipeline.verify_gemma_tokenization import TOOLS_SCHEMA

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"loading base model: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to("cuda:0")
    if args.no_adapter:
        print("SKIPPING adapter — running base model only")
    else:
        print(f"loading adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    fixtures = sorted(p for p in args.fixtures_dir.iterdir() if p.is_dir())
    if args.fixture:
        fixtures = [p for p in fixtures if p.name == args.fixture]
        if not fixtures:
            print(f"no fixture named {args.fixture!r}")
            return 1
    print(f"fixtures: {[p.name for p in fixtures]}")
    print(f"modes: {args.modes}\n")

    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for fixture in fixtures:
        goal = load_goal(fixture)
        for mode in args.modes:
            print(f"\n=== {fixture.name} / {mode} ===")
            # Fresh copy of the fixture for each run
            with tempfile.TemporaryDirectory(prefix=f"harness_{fixture.name}_{mode}_") as td:
                work_dir = Path(td) / fixture.name
                shutil.copytree(fixture, work_dir)

                t0 = time.time()
                metrics = run_agent(
                    model=model, tokenizer=tok, tools_schema=TOOLS_SCHEMA,
                    work_dir=work_dir, goal=goal, mode=mode,
                    max_steps=args.max_steps,
                    max_context_tokens=args.max_context_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    verbose=args.verbose,
                )
                success, validation_tail = validate(work_dir)
                metrics["fixture"] = fixture.name
                metrics["success"] = success
                metrics["validation_tail"] = validation_tail
                results.append(metrics)

                print(f"  → success={success} steps={metrics['steps']} "
                      f"tokens={metrics['total_input_tokens']} "
                      f"max_ctx={metrics['max_context_size']} "
                      f"checkpoints={metrics['n_checkpoints']} "
                      f"elided={metrics['n_elided']} "
                      f"time={metrics['wall_time_s']:.1f}s "
                      f"stop={metrics['stop_reason']}")

    with args.out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'fixture':<25} {'mode':<10} {'ok':<4} {'steps':<6} {'tokens':<8} {'ctx':<6} {'ckpt':<5}")
    print("-" * 80)
    for r in results:
        print(f"{r['fixture']:<25} {r['mode']:<10} "
              f"{'✓' if r['success'] else '✗':<4} "
              f"{r['steps']:<6} {r['total_input_tokens']:<8} "
              f"{r['max_context_size']:<6} {r['n_checkpoints']:<5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
