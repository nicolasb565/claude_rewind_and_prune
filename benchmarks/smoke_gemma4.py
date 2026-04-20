#!/usr/bin/env python3
"""
Minimal smoke test: does Gemma 4 load + do one LoRA forward/backward step
on ROCm 7.2.2 / gfx1201 (RX 9070 XT)?

No real data — fabricates a short chat and runs one training step. Success
criteria: model loads, optimizer steps, loss drops on a trivial overfit
microbatch. Prints VRAM after load so we know the footprint.

Run this INSIDE the rocm/pytorch container via smoke_gemma4_docker.sh.
"""
from __future__ import annotations

import os
import sys
import time

import torch


# HF IDs use capital E: google/gemma-4-E2B-it, google/gemma-4-E4B-it.
# Models are gated — needs accepted license + HF_TOKEN.
#
# Default E2B because E4B is 15.9 GB in BF16 (Google's "effective" naming
# hides ~8B real params) — doesn't fit 16 GB VRAM for training. E2B is
# ~5 GB and leaves headroom for LoRA adapters + activations.
MODEL_ID = os.environ.get("SMOKE_MODEL", "google/gemma-4-E2B-it")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    # Also stderr so it survives stdout redirection quirks / tqdm interactions.
    print(line, file=sys.stderr, flush=True)


def vram() -> str:
    if not torch.cuda.is_available():
        return "(no cuda)"
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1 << 30)
    tot = total / (1 << 30)
    return f"{used:.2f}/{tot:.2f} GiB"


def main() -> int:
    # Silence tqdm — its \r output can mask later stdout on abrupt exit.
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    log(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"device 0: {torch.cuda.get_device_name(0)}")
        log(f"vram (before load): {vram()}")

    log(f"loading tokenizer: {MODEL_ID}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    log("tokenizer loaded")

    log("loading model weights (this can take a minute)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    log(f"model instantiated on {next(model.parameters()).device}  dtype={next(model.parameters()).dtype}")

    log("moving model to cuda:0")
    model = model.to("cuda:0")
    torch.cuda.synchronize()
    model.config.use_cache = False
    log(f"vram (after cuda move): {vram()}")

    log("attaching LoRA (r=16) to all linear layers")
    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_cfg)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    log(f"vram (after LoRA): {vram()}")

    # Fabricate a tiny overfit microbatch: one short chat turn.
    chat = [
        {"role": "user", "content": "Reply with exactly 'ACK'."},
        {"role": "assistant", "content": "ACK"},
    ]
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    enc["labels"] = enc["input_ids"].clone()
    log(f"microbatch: {enc['input_ids'].shape[1]} tokens")

    model.train()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    losses = []
    for step in range(5):
        out = model(**enc)
        out.loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        losses.append(out.loss.item())
        log(f"step {step}  loss={out.loss.item():.4f}  vram={vram()}")

    improved = losses[-1] < losses[0]
    log(f"losses: {[f'{l:.4f}' for l in losses]}")
    log(f"SMOKE {'PASS' if improved else 'FAIL'} — loss {'dropped' if improved else 'did not drop'} over 5 steps")
    return 0 if improved else 1


if __name__ == "__main__":
    sys.exit(main())
