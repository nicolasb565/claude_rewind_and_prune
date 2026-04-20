#!/usr/bin/env python3
"""
Smoke test — does Qwen 3.5 0.8B LoRA training actually work on gfx1201?

Prior session saw 'GPU memory-access fault' on 0.8B under native attention
(suspected linear-attention kernel bug). This test forces SDPA and runs
one forward + one backward on a real training example. If it survives
both, a full training run is safe to launch.

Usage (inside rocm/pytorch):
  python -u benchmarks/smoke_qwen35_training.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    model_id = "Qwen/Qwen3.5-0.8B"
    log(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    log(f"loading tokenizer: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log(f"loading model with attn_implementation='sdpa'")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to("cuda:0")
    except Exception as e:
        log(f"FAIL model load: {type(e).__name__}: {e}")
        return 1

    free, total = torch.cuda.mem_get_info()
    log(f"vram after load: {(total-free)/(1<<30):.2f}/{total/(1<<30):.2f} GiB")

    log("attaching LoRA r=16 all-linear")
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear",
    )
    model = get_peft_model(model, peft_cfg)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Load one real training chunk to test the full path
    log("loading one training chunk")
    chunks_path = Path("data/generated/hygiene_v1.chunks.jsonl")
    if not chunks_path.exists():
        log(f"no chunks file at {chunks_path} — render v6 data first")
        return 1
    with chunks_path.open() as f:
        row = json.loads(f.readline())
    text = row["prompt"] + row["completion"]
    enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to("cuda:0")
    labels = enc.input_ids.clone()
    log(f"input: {enc.input_ids.shape[1]} tokens")

    log("=== FORWARD ===")
    try:
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels)
    except Exception as e:
        log(f"FAIL forward: {type(e).__name__}: {e}")
        return 2
    log(f"forward OK, loss={out.loss.item():.4f}")

    log("=== BACKWARD ===")
    try:
        out.loss.backward()
        torch.cuda.synchronize()
    except Exception as e:
        log(f"FAIL backward: {type(e).__name__}: {e}")
        return 3
    log("backward OK")

    free, total = torch.cuda.mem_get_info()
    log(f"peak vram: {(total-free)/(1<<30):.2f}/{total/(1<<30):.2f} GiB")
    log("SMOKE PASSED — Qwen 3.5 0.8B LoRA training viable on this card")
    return 0


if __name__ == "__main__":
    sys.exit(main())
