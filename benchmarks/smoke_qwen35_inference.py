#!/usr/bin/env python3
"""
Inference smoke for Qwen 3.5 on the RX 9070 XT (gfx1201).

Qwen 3.5 is a Mamba/Transformer hybrid and ships as a multimodal
ForConditionalGeneration architecture. For text-only use we try
AutoModelForCausalLM first (text-head path), falling back to the
multimodal class if transformers doesn't map it.

Run via smoke_qwen35_docker.sh.
"""
from __future__ import annotations

import os
import sys
import time

import torch

MODEL_ID = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3.5-4B")
PROMPT = os.environ.get("SMOKE_PROMPT", "In one sentence: what is context management in an LLM agent?")
MAX_NEW = int(os.environ.get("SMOKE_MAX_NEW", "60"))


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def vram() -> str:
    if not torch.cuda.is_available():
        return "(no cuda)"
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1 << 30)
    tot = total / (1 << 30)
    return f"{used:.2f}/{tot:.2f} GiB"


def load_model(model_id: str):
    from transformers import AutoModelForCausalLM
    try:
        m = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        log(f"loaded via AutoModelForCausalLM: {type(m).__name__}")
        return m
    except Exception as e:
        log(f"AutoModelForCausalLM failed: {type(e).__name__}: {str(e)[:200]}")
        log("falling back to AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        log(f"loaded via AutoModelForImageTextToText: {type(m).__name__}")
        return m


def main() -> int:
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    log(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"device: {torch.cuda.get_device_name(0)}")
        log(f"vram (before load): {vram()}")

    from transformers import AutoTokenizer

    log(f"loading tokenizer + model: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = load_model(MODEL_ID)
    model = model.to("cuda:0")
    torch.cuda.synchronize()
    model.eval()
    log(f"vram (after load): {vram()}")

    chat = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").input_ids.to(model.device)
    prompt_tokens = int(enc.shape[1])
    log(f"prompt tokens: {prompt_tokens}")

    with torch.no_grad():
        _ = model.generate(enc, max_new_tokens=4, do_sample=False)
    torch.cuda.synchronize()

    log("generating...")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            enc,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    torch.cuda.synchronize()
    dt = time.time() - t0

    gen_tokens = int(out.shape[1]) - prompt_tokens
    decoded = tok.decode(out[0, prompt_tokens:], skip_special_tokens=True)
    tok_per_s = gen_tokens / dt if dt > 0 else 0
    log(f"generated {gen_tokens} tokens in {dt:.2f}s — {tok_per_s:.1f} t/s")
    log(f"output: {decoded!r}")
    log(f"vram (after generate): {vram()}")

    ok = gen_tokens >= 5 and tok_per_s > 1
    log(f"SMOKE {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
