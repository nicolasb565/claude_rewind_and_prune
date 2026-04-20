#!/usr/bin/env python3
"""Runner for the shadow-logged agent on a single fixture.

Loads Qwen 3.5 4B bf16, runs one session on the chosen fixture, writes
per-turn JSONL to harness/results/shadow/<fixture>_<timestamp>.jsonl, and
prints a summary.
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


def validate(work_dir: Path) -> tuple[bool, str]:
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
    ap.add_argument("--fixture", default="bug_01_offbyone",
                    help="name of a directory under harness/fixtures/")
    ap.add_argument("--fixtures-dir", type=Path,
                    default=Path("harness/fixtures"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-context-tokens", type=int, default=16000)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("harness/results/shadow"))
    ap.add_argument("--verbose", action="store_true", default=True)
    ap.add_argument("--act-on-shadow", action="store_true", default=False,
                    help="Phase 2: on first shadow YES, rewind prior turns "
                         "(keeping goal + last tool_result) and splice summary.")
    ap.add_argument("--rewind-cooldown", type=int, default=3,
                    help="Turns to wait after a rewind before another can fire.")
    args = ap.parse_args()

    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    from harness.shadow_agent import run_shadow_agent
    from src.pipeline.verify_gemma_tokenization import TOOLS_SCHEMA

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    fixture_dir = args.fixtures_dir / args.fixture
    if not fixture_dir.exists():
        print(f"fixture not found: {fixture_dir}", file=sys.stderr)
        return 1
    goal = (fixture_dir / "GOAL.md").read_text().strip()

    print(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = args.out_dir / f"{args.fixture}_{stamp}.jsonl"
    meta_path = args.out_dir / f"{args.fixture}_{stamp}.meta.json"
    print(f"log: {log_path}")

    with tempfile.TemporaryDirectory(prefix=f"shadow_{args.fixture}_") as td:
        work_dir = Path(td) / args.fixture
        # Skip build/ artifacts during copy — they may contain absolute paths
        # baked into Makefiles from a prior configure. Fresh configure happens
        # in VALIDATE.sh or via the agent's own rebuild.
        shutil.copytree(
            fixture_dir, work_dir,
            ignore=shutil.ignore_patterns("build", "__pycache__", ".pytest_cache"),
        )

        metrics = run_shadow_agent(
            model=model, tokenizer=tok, tools_schema=TOOLS_SCHEMA,
            work_dir=work_dir, goal=goal,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            max_context_tokens=args.max_context_tokens,
            log_path=log_path, verbose=args.verbose,
            act_on_shadow=args.act_on_shadow,
            rewind_cooldown=args.rewind_cooldown,
        )
        success, tail = validate(work_dir)
        metrics.update({
            "fixture": args.fixture, "model": args.model,
            "success": success, "validation_tail": tail,
            "log_path": str(log_path), "stamp": stamp,
        })

    meta_path.write_text(json.dumps(metrics, indent=2))
    print("\n=== summary ===")
    print(f"  fixture:      {args.fixture}")
    print(f"  act_on_shadow:{metrics.get('act_on_shadow', False)}")
    print(f"  success:      {success}")
    print(f"  steps:        {metrics['steps']}")
    print(f"  tool_calls:   {metrics['n_tool_calls']}")
    print(f"  shadow YES:   {metrics['n_shadow_yes']}")
    print(f"  rewinds:      {metrics.get('n_rewinds', 0)}")
    print(f"  stop_reason:  {metrics['stop_reason']}")
    print(f"  wall time:    {metrics['wall_time_s']:.1f}s")
    print(f"  log:          {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
