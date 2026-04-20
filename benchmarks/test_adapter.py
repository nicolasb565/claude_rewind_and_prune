#!/usr/bin/env python3
"""
Quick manual test of the trained LoRA adapter.

Takes a training chunk, strips the final checkpoint emission (last 2
messages: assistant tool_call + tool ack), and generates with both the
base model and the fine-tuned base+adapter. Prints both completions
side-by-side so we can eyeball whether the trained model emits a
checkpoint_progress call where the base model doesn't.

Usage (inside rocm/pytorch container via test_adapter_docker.sh):
  python -u benchmarks/test_adapter.py [--line N] [--data path]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/generated/hygiene_v1.chunks.jsonl"))
    ap.add_argument("--line", type=int, default=0, help="0-based chunk index (from val partition if possible)")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--adapter", type=Path, default=Path("proxy/experiments/qwen25_coder_1.5b_hygiene_v1/final"))
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log(f"loading tokenizer: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model)

    # Load the chunk and strip the checkpoint emission from the messages.
    with args.data.open() as f:
        for i, line in enumerate(f):
            if i == args.line:
                row = json.loads(line)
                break
        else:
            raise SystemExit(f"line {args.line} not found")

    msgs = row["messages"]
    # Expected tail: assistant tool_call (checkpoint_progress) + tool ack.
    # Strip those two; the model should ideally produce something similar.
    ground_truth = msgs[-2:]
    prompt_msgs = msgs[:-2]
    log(f"chunk {args.line}: {len(msgs)} msgs → stripped to {len(prompt_msgs)} prompt msgs")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.pipeline.verify_gemma_tokenization import TOOLS_SCHEMA

    text = tok.apply_chat_template(
        prompt_msgs, tools=TOOLS_SCHEMA,
        tokenize=False, add_generation_prompt=True,
    )
    enc = tok(text, return_tensors="pt").to("cuda:0")
    log(f"prompt tokens: {int(enc.input_ids.shape[1])}")

    # ── Base model generation ─────────────────────────────────────────────
    log(f"loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    base.eval()
    log("generating with base (no adapter)...")
    with torch.no_grad():
        t0 = time.time()
        out_base = base.generate(
            enc.input_ids,
            max_new_tokens=args.max_new, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        dt_base = time.time() - t0
    base_text = tok.decode(out_base[0, int(enc.input_ids.shape[1]):], skip_special_tokens=False)
    log(f"base generated {int(out_base.shape[1] - enc.input_ids.shape[1])} tokens in {dt_base:.1f}s")

    # Free base model before loading base+adapter
    del base
    torch.cuda.empty_cache()

    # ── Base + adapter generation ─────────────────────────────────────────
    log(f"loading base + adapter: {args.adapter}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    log("generating with base+adapter (greedy)...")
    with torch.no_grad():
        t0 = time.time()
        out_ft = model.generate(
            enc.input_ids,
            max_new_tokens=args.max_new, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        dt_ft = time.time() - t0
    ft_text = tok.decode(out_ft[0, int(enc.input_ids.shape[1]):], skip_special_tokens=False)
    log(f"adapter greedy: {int(out_ft.shape[1] - enc.input_ids.shape[1])} tokens in {dt_ft:.1f}s")

    # Also try sampled — sometimes greedy misses a trained pattern that
    # sampling would surface
    log("generating with base+adapter (sampled, 4 completions)...")
    sampled_outputs = []
    with torch.no_grad():
        for i in range(4):
            out_s = model.generate(
                enc.input_ids,
                max_new_tokens=args.max_new, do_sample=True, temperature=0.8, top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
            sampled_outputs.append(tok.decode(out_s[0, int(enc.input_ids.shape[1]):], skip_special_tokens=False))

    # ── Report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("GROUND TRUTH (what Sonnet said the agent should emit):")
    print("=" * 80)
    for m in ground_truth:
        print(json.dumps(m, indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print("BASE MODEL (untrained) output:")
    print("=" * 80)
    print(base_text)

    print("\n" + "=" * 80)
    print("BASE + ADAPTER (trained, GREEDY) output:")
    print("=" * 80)
    print(ft_text)

    print("\n" + "=" * 80)
    print("BASE + ADAPTER (trained, 4 SAMPLED) outputs:")
    print("=" * 80)
    for i, s in enumerate(sampled_outputs):
        print(f"--- sample {i+1} ---")
        print(s)
        print()

    # Quick feature check: does each output contain the tool call marker?
    def has_checkpoint(text: str) -> bool:
        return "checkpoint_progress" in text or "mcp__bookmarks" in text

    print("\n" + "=" * 80)
    print(f"base emits checkpoint_progress:   {has_checkpoint(base_text)}")
    print(f"adapter emits checkpoint_progress: {has_checkpoint(ft_text)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
