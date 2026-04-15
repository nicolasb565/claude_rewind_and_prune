#!/usr/bin/env python3
"""
Fine-tune phi4-mini on the windowed stuck-detection dataset using LoRA.

Training format: multi-turn chat sessions where each user message is a
rendered tool call and each assistant message is a single-letter label.
Loss is computed only on assistant label tokens; user/system tokens are
masked with -100. Causal attention ensures each label prediction only
sees prior turns (no future leakage).

Usage:
  .venv/bin/python benchmarks/finetune_train.py
  .venv/bin/python benchmarks/finetune_train.py --smoke  # 500 examples, 1 epoch, quick check
  .venv/bin/python benchmarks/finetune_train.py --model Qwen/Qwen2.5-7B-Instruct

Requires:
  torch+rocm, transformers, peft, trl, accelerate, datasets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _preflight_vram_check() -> None:
    """Fail fast if VRAM is dirty. On ROCm/gfx1201 a crashed process can
    leave a 10+ GB KFD zombie allocation that makes subsequent training
    attempts OOM immediately. We want a loud error, not a silent cascade.

    Also limits per-process VRAM so a crash only poisons our budget instead
    of the whole GPU.
    """
    import torch
    if not torch.cuda.is_available():
        print("WARN: CUDA/ROCm not available, will run on CPU (very slow)",
              flush=True)
        return
    free, total = torch.cuda.mem_get_info(0)
    free_gb = free / 1e9
    total_gb = total / 1e9
    print(f"\n=== Preflight VRAM check ===", flush=True)
    print(f"  device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  total:  {total_gb:.1f} GB", flush=True)
    print(f"  free:   {free_gb:.1f} GB", flush=True)
    if free_gb < 13.0:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB VRAM free (need >=13 GB).\n"
            f"Something else is holding VRAM — likely a zombie KFD allocation "
            f"from a previous crash.\n"
            f"Recovery: reboot OR `sudo rocm-smi --gpureset --device 0` OR "
            f"restart the desktop session (kill/restart KDE/gnome-shell)."
        )
    # Cap per-process budget at 90% so any future leak from *this* process
    # doesn't poison more than we asked for.
    torch.cuda.set_per_process_memory_fraction(0.90, 0)
    print(f"  per-process cap set to 90% ({total_gb * 0.9:.1f} GB)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-4-mini-instruct",
                    help="HF model id")
    ap.add_argument("--train-file", default="data/generated/finetune_train.jsonl")
    ap.add_argument("--val-file", default="data/generated/finetune_val.jsonl")
    ap.add_argument("--output-dir", default="proxy/experiments/phi4_mini_stuck_lora")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--smoke", action="store_true", help="500 examples, 1 epoch, no save")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Imports here so that scripts that just need --help don't need the full
    # stack installed.
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    # Layer 1: preflight VRAM check — fail fast on dirty GPU state.
    _preflight_vram_check()

    print(f"\n=== Config ===")
    print(f"  model:        {args.model}")
    print(f"  train file:   {args.train_file}")
    print(f"  val file:     {args.val_file}")
    print(f"  output dir:   {args.output_dir}")
    print(f"  max length:   {args.max_length}")
    print(f"  batch size:   {args.batch_size} (effective {args.batch_size * args.grad_accum})")
    print(f"  learning rate:{args.lr}")
    print(f"  epochs:       {args.epochs}")
    print(f"  LoRA:         r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")
    print(f"  smoke mode:   {args.smoke}")
    print(f"  torch:        {torch.__version__}")
    print(f"  cuda avail:   {torch.cuda.is_available()}")
    print(f"  hip version:  {getattr(torch.version, 'hip', None)}")
    if torch.cuda.is_available():
        print(f"  device:       {torch.cuda.get_device_name(0)}")
        print(f"  vram:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print(f"\n=== Loading tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  vocab size:   {tokenizer.vocab_size}")
    print(f"  chat template present: {tokenizer.chat_template is not None}")

    # Patch chat template to add {% generation %}...{% endgeneration %} around
    # assistant content. trl's SFTTrainer ≥1.0 requires these markers to
    # determine which tokens contribute to loss (assistant_only_loss=True).
    # Phi-4-mini's stock template is a minimal {% for %} loop with no such
    # markers, so we inject a branch for the assistant role.
    # NOTE: checking for "{% generation %}" specifically, not just "generation"
    # — Phi-4's template already contains "add_generation_prompt" as a var
    # name, which would false-match a substring check.
    if "{% generation %}" not in (tokenizer.chat_template or ""):
        print("  patching chat template with {% generation %} markers")
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' and 'tools' in message and message['tools'] is not none %}"
            "{{ '<|' + message['role'] + '|>' + message['content'] + '<|tool|>' + message['tools'] + '<|/tool|>' + '<|end|>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|assistant|>' }}{% generation %}{{ message['content'] + '<|end|>' }}{% endgeneration %}"
            "{% else %}"
            "{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %}"
        )

    # ── Model ──────────────────────────────────────────────────────────────
    # NOTE: don't pass trust_remote_code=True — Phi-4-mini's vendored
    # modeling_phi3.py imports LossKwargs which was removed in transformers 5.x.
    # The Phi3 architecture is natively supported, so we use the built-in class.
    #
    # attn_implementation="sdpa" is required — eager attention is O(seq²) in
    # memory and OOMs on 16 GB VRAM at 1024 tokens (we hit this empirically).
    # SDPA is memory-efficient and backed by torch-rocm's fused kernels.
    # Phi-4-mini is natively trained in bf16 per the model card. Loading in
    # bf16 hits different torch-rocm kernel paths than fp16 and avoids a
    # specific segfault we hit with fp16 + SDPA on gfx1201 (torch 2.11+rocm7.2).
    print(f"\n=== Loading model (bf16, device_map=auto) ===")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    # Enable gradient checkpointing for VRAM savings on long sequences
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # ── LoRA ───────────────────────────────────────────────────────────────
    print(f"\n=== Applying LoRA ===")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    # autocast_adapter_dtype=False keeps the LoRA adapter parameters in the
    # base model's dtype (fp16) instead of upcasting to fp32. The upcast
    # kernel is missing from torch-rocm 6.3 for gfx1201 (RX 9070 XT) and
    # crashes with "HIP error: invalid device function".
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    model.print_trainable_parameters()

    # ── Dataset ────────────────────────────────────────────────────────────
    print(f"\n=== Loading dataset ===")
    train_ds = load_dataset("json", data_files=str(REPO / args.train_file), split="train")
    val_ds = load_dataset("json", data_files=str(REPO / args.val_file), split="train")
    print(f"  train: {len(train_ds)} sessions")
    print(f"  val:   {len(val_ds)} sessions")

    if args.smoke:
        train_ds = train_ds.select(range(min(500, len(train_ds))))
        val_ds = val_ds.select(range(min(50, len(val_ds))))
        print(f"  [smoke] truncated to train={len(train_ds)} val={len(val_ds)}")

    # Pre-truncate sessions that exceed max_length at the message level,
    # dropping tail messages until the rendered template fits. SFTTrainer's
    # max_length is applied inconsistently in trl 1.1.0 — safer to guarantee
    # the input is small enough before handing it off.
    print(f"\n=== Truncating sessions to fit max_length={args.max_length} ===")

    def _count_tokens(messages):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return len(tokenizer.encode(text, add_special_tokens=False))

    def _truncate_to_fit(example):
        messages = example["messages"]
        tok_len = _count_tokens(messages)
        if tok_len <= args.max_length:
            return example
        # Drop pairs of (user, assistant) from the tail until it fits.
        # messages[0] is system — keep it. Pairs start at index 1.
        while tok_len > args.max_length and len(messages) >= 3:
            # Drop last user+assistant pair
            if messages[-1]["role"] == "assistant" and messages[-2]["role"] == "user":
                messages = messages[:-2]
            else:
                messages = messages[:-1]
            tok_len = _count_tokens(messages)
        example["messages"] = messages
        return example

    import time
    t0 = time.time()
    train_ds = train_ds.map(_truncate_to_fit, desc="truncating train")
    val_ds = val_ds.map(_truncate_to_fit, desc="truncating val")
    # Drop sessions that ended up with no assistant turns (shouldn't happen
    # but defensive)
    train_ds = train_ds.filter(
        lambda e: any(m["role"] == "assistant" for m in e["messages"]),
        desc="drop-empty train",
    )
    val_ds = val_ds.filter(
        lambda e: any(m["role"] == "assistant" for m in e["messages"]),
        desc="drop-empty val",
    )
    print(f"  after truncation: train={len(train_ds)} val={len(val_ds)} "
          f"({time.time()-t0:.1f}s)")

    # trl's SFTTrainer with messages format expects a "messages" field.
    # completion_only_loss=True tells it to mask user/system tokens.

    # ── Training ───────────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps" if not args.smoke else "no",
        save_steps=args.save_steps,
        save_total_limit=2,
        max_length=args.max_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        seed=args.seed,
        # Multi-turn chat format: assistant_only_loss masks everything except
        # assistant label tokens. Do NOT also set completion_only_loss=True —
        # that's for single user+completion pairs and conflicts with
        # assistant_only_loss in trl 1.1.0, causing trainer.train() to return
        # immediately with no loss.
        assistant_only_loss=True,
        report_to=[],
    )

    # trl ≥1.0 renamed `tokenizer` → `processing_class`
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print(f"\n=== After SFTTrainer init ===", flush=True)
    print(f"  train dataset: {trainer.train_dataset}", flush=True)
    print(f"  train length: {len(trainer.train_dataset)}", flush=True)
    if len(trainer.train_dataset) > 0:
        sample = trainer.train_dataset[0]
        print(f"  first example keys: {list(sample.keys())}", flush=True)
        if "input_ids" in sample:
            print(f"  first input_ids len: {len(sample['input_ids'])}", flush=True)
        if "labels" in sample:
            n_label_tok = sum(1 for t in sample["labels"] if t != -100)
            print(f"  first labels len: {len(sample['labels'])}  "
                  f"non-masked: {n_label_tok}", flush=True)
        if "assistant_masks" in sample:
            masks = sample["assistant_masks"]
            n_ones = sum(1 for m in masks if m == 1)
            print(f"  first assistant_masks len: {len(masks)}  ones: {n_ones}", flush=True)
            # Show which tokens are marked
            ids = sample["input_ids"]
            marked_toks = [ids[i] for i, m in enumerate(masks) if m == 1]
            if marked_toks:
                print(f"  first marked token: {marked_toks[0]} "
                      f"({tokenizer.decode([marked_toks[0]])!r})", flush=True)
                print(f"  decoded marked region: "
                      f"{tokenizer.decode(marked_toks[:20])!r}", flush=True)

    # (Removed manual forward+backward probe — it was a debug aid. If training
    # itself fails we'll see the trainer's own error message now.)

    print(f"\n=== Training ===", flush=True)
    import traceback
    import signal

    # Layer 3: signal handler — catches SIGTERM and attempts orderly cleanup
    # (only useful on polite kills, not SIGKILL / segfaults).
    _shutdown_requested = [False]
    def _sigterm_handler(signum, frame):
        print(f"\n!!! received signal {signum}, requesting shutdown", flush=True)
        _shutdown_requested[0] = True
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)

    # Layer 4: try/finally with explicit tear-down so a Python-level exception
    # doesn't leak the trainer object + its references into the next run.
    try:
        result = trainer.train()
        print(f"trainer.train() returned: {result}", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        try:
            print(f"\n=== Cleanup ===", flush=True)
            del trainer
        except NameError:
            pass
        try:
            del model
        except NameError:
            pass
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_after, _ = torch.cuda.mem_get_info(0)
        print(f"  vram free after cleanup: {free_after/1e9:.1f} GB", flush=True)

    if not args.smoke:
        print(f"\n=== Saving LoRA adapter ===")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"  saved: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
