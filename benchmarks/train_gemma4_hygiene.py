#!/usr/bin/env python3
"""
LoRA fine-tune Gemma 4 E2B on the hygiene_v1 training chunks.

Design (discussed 2026-04-18):
  - One chat-format training example per checkpoint
  - Goal (initial user prompt) + recent context + checkpoint emission
  - ≤ 8192 tokens per chunk, fits on 16 GB with LoRA + grad checkpointing
  - BF16 base + SDPA attention (no flash-attn on RDNA4)

Training config defaults target a conservative first run:
  - LoRA r=16, alpha=32, all-linear
  - per-device batch 1, grad accum 8 (effective batch 8)
  - 2 epochs, cosine LR, warmup 30 steps
  - Loss on all assistant tokens (trl SFTTrainer default with chat format)

Run via benchmarks/train_gemma4_hygiene_docker.sh. This script assumes
it's inside the rocm/pytorch container with transformers/peft/trl/
accelerate installed.

Usage:
  python -u benchmarks/train_gemma4_hygiene.py [args]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    print(line, file=sys.stderr, flush=True)


def vram() -> str:
    import torch
    if not torch.cuda.is_available():
        return "(no cuda)"
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1 << 30)
    tot = total / (1 << 30)
    return f"{used:.2f}/{tot:.2f} GiB"


# Tool schema — must match what the annotator/renderer used and what
# inference will use. Kept in sync with benchmarks/smoke_gemma4_inference
# by re-importing from the pipeline module.
def get_tools_schema() -> list[dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.pipeline.verify_gemma_tokenization import TOOLS_SCHEMA
    return TOOLS_SCHEMA


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/generated/hygiene_v1.chunks.jsonl"))
    # Qwen 2.5 instead of 3.5 — 3.5 uses linear-attention in some layers
    # whose ROCm fallback triggers GPU memory-access faults mid-training
    # on gfx1201. 2.5 is plain transformer, well-tested on this hardware
    # (phi4-mini LoRA worked in prior session). Coder variant gives us a
    # stronger base for the tool-use pattern we're teaching.
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--output-dir", type=Path, default=Path("proxy/experiments/qwen25_coder_3b_hygiene_v8"))
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-steps", type=int, default=30)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny training run: 16 examples, 1 epoch, for plumbing check.")
    ap.add_argument("--no-grad-checkpointing", action="store_true",
                    help="Disable gradient checkpointing. Needed for Qwen 3.5 (Mamba hybrid) on ROCm per transformers#28023.")
    ap.add_argument("--attn-impl", default="sdpa",
                    choices=["sdpa", "flash_attention_2", "eager"],
                    help="Attention backend. Default sdpa. flash_attention_2 requires source-built flash-attn with FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE on ROCm.")
    args = ap.parse_args()

    random.seed(args.seed)
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    import torch
    log(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"device: {torch.cuda.get_device_name(0)}")
        log(f"vram (before load): {vram()}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    log(f"loading tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log(f"loading model (bf16, attn={args.attn_impl})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    torch.cuda.synchronize()
    model.config.use_cache = False
    log(f"vram (after load): {vram()}")

    log(f"attaching LoRA r={args.lora_r} alpha={args.lora_alpha} all-linear")
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_cfg)
    if not args.no_grad_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        log("gradient checkpointing DISABLED (Qwen 3.5 workaround)")
    model.print_trainable_parameters()
    log(f"vram (after LoRA): {vram()}")

    # ── Load chunks ───────────────────────────────────────────────────────
    log(f"loading chunks from {args.data}")
    rows = []
    with args.data.open() as f:
        for line in f:
            rows.append(json.loads(line))
    random.Random(args.seed).shuffle(rows)
    if args.smoke:
        rows = rows[:16]
        args.epochs = 1.0

    n_val = max(1, int(len(rows) * args.val_fraction))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    log(f"chunks: {len(rows)}  train={len(train_rows)}  val={len(val_rows)}")

    tools_schema = get_tools_schema()

    # V2: prompt/completion split so completion_only_loss masks loss to
    # just the checkpoint emission (v1 failure mode: loss on all
    # trajectory tokens diluted the <tool_call> gradient signal).
    # render_for_gemma.py v2 writes these fields directly.
    def to_ds(rows_):
        out = []
        for r in rows_:
            if "prompt" in r and "completion" in r:
                out.append({"prompt": r["prompt"], "completion": r["completion"]})
            else:
                raise SystemExit(
                    "chunks file missing prompt/completion fields — rerun "
                    "render_for_gemma.py after the v2 update."
                )
        return Dataset.from_list(out)

    train_ds = to_ds(train_rows)
    val_ds = to_ds(val_rows)

    # ── Training config ───────────────────────────────────────────────────
    cfg = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        gradient_checkpointing=not args.no_grad_checkpointing,
        max_length=args.max_seq_length,
        packing=False,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        # Chat-template handling: let SFTTrainer apply the template at
        # tokenization time. We pass tools via the formatting path below.
        dataset_kwargs={"skip_prepare_dataset": False},
        # Tried use_liger_kernel=True on ROCm 7.2.2 / gfx1201:
        # activated but caused GPU memory-access faults during training
        # (after a few successful steps). The Liger kernels don't fully
        # support RDNA4/Qwen3.5 path yet. Disabled.
        use_liger_kernel=False,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    log("starting training")
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
    )
    trainer.train()
    log(f"vram (peak during train): {vram()}")

    # ── Save ──────────────────────────────────────────────────────────────
    save_path = args.output_dir / "final"
    log(f"saving adapter to {save_path}")
    trainer.save_model(str(save_path))
    tok.save_pretrained(str(save_path))
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
