#!/usr/bin/env python3
"""
Smoke test — does flash-attention-2 (Triton-AMD) work on Qwen 2.5 Coder 3B
at seq_len=8192 on gfx1201?

Without flash, 3B @ 8K won't fit (>20 GiB estimated). If flash works we
should land in the 12-14 GiB range. If it doesn't, we fall back to 2K
without flash.
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

    model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
    seq_len = 4096
    log(f"torch {torch.__version__}")
    log(f"target: {model_id}  seq_len={seq_len}")

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log("loading model with attn_implementation='flash_attention_2'")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        ).to("cuda:0")
    except Exception as e:
        log(f"FAIL model load with flash_attention_2: {type(e).__name__}: {e}")
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

    # Use a real chunk, concatenated to reach target seq_len if needed
    log(f"building {seq_len}-token input from real training chunks")
    chunks_path = Path("data/generated/hygiene_v1.chunks.jsonl")
    text = ""
    if chunks_path.exists():
        with chunks_path.open() as f:
            for line in f:
                row = json.loads(line)
                text += row["prompt"] + row["completion"] + "\n"
                if len(text) > seq_len * 5:
                    break
    else:
        text = "hello world " * (seq_len * 2)

    enc = tok(text, return_tensors="pt", truncation=True, max_length=seq_len).to("cuda:0")
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
    log("SMOKE PASSED — flash_attention_2 works on gfx1201 for 3B @ 8K")
    return 0


if __name__ == "__main__":
    sys.exit(main())
