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
            f"Only {free_gb:.1f} GB VRAM free (need >=13 GB)."
        )


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
    )
    from peft import LoraConfig, get_peft_model

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
    # bf16 matches the weight dtype and hits the fused bf16 kernel paths.
    print(f"\n=== Loading model (bf16, device_map=auto) ===")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    # NOTE: do NOT call model.gradient_checkpointing_enable() here. We let
    # SFTConfig(gradient_checkpointing=True) handle that, but only AFTER
    # LoRA is applied. Enabling it manually before peft wraps the model
    # can cause grad flow through the frozen base to fail silently.

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
    # base model's dtype (bf16) instead of upcasting to fp32. The upcast
    # kernel has historically been missing from torch-rocm for gfx1201.
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    model.print_trainable_parameters()

    # Critical for gradient checkpointing + LoRA: the frozen input embeddings
    # normally have requires_grad=False, which breaks autograd's ability to
    # propagate gradients *through* the embedding layer back to the LoRA
    # adapters in deeper layers. enable_input_require_grads() inserts a
    # forward hook that forces the embedding output to require grad, which
    # makes the gradient path complete without unfreezing the embedding.
    # Without this, the training loop computes zero gradients and
    # trainer.train() silently exits at step 0 — the exact symptom we hit.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def _make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

    # Enable gradient checkpointing NOW, immediately after LoRA + input_grads.
    # Must happen BEFORE the sanity forward or it will OOM trying to hold
    # full activations for a 3.8B model at seq 995.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

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

    # ── Sanity forward: same pattern as docker_real_test.py ─────────────
    # If this crashes, something in our setup (preflight, imports, etc.)
    # broke the model. If it works, the crash is in our training loop.
    import time
    _breadcrumb_path = "/tmp/finetune_breadcrumb.log"
    with open(_breadcrumb_path, "w") as _bf:
        _bf.write(f"[{time.time():.1f}] sanity-begin\n")
        _bf.flush()
        os.fsync(_bf.fileno())

    def _crumb(msg):
        with open(_breadcrumb_path, "a") as _bf:
            _bf.write(f"[{time.time():.1f}] {msg}\n")
            _bf.flush()
            os.fsync(_bf.fileno())

    print(f"\n=== Sanity forward (random seq 995) ===", flush=True)
    _crumb("sanity-print-done")
    _sanity_dev = next(model.parameters()).device
    _crumb(f"sanity-device={_sanity_dev}")
    _x = torch.randint(0, 200000, (1, 995), device=_sanity_dev, dtype=torch.long)
    _crumb("sanity-x-created")
    _y = _x.clone()
    _attn = torch.ones_like(_x)
    _crumb("sanity-tensors-ready")
    _t0 = time.time()
    model.train()
    _crumb("sanity-model-train")
    _out = model(input_ids=_x, attention_mask=_attn, labels=_y)
    _crumb(f"sanity-forward-done loss={_out.loss.item():.4f}")
    print(f"  sanity forward ok loss={_out.loss.item():.4f} took={time.time()-_t0:.1f}s", flush=True)
    _t0 = time.time()
    _out.loss.backward()
    _crumb("sanity-backward-done")
    print(f"  sanity backward ok took={time.time()-_t0:.1f}s", flush=True)
    # Clear grads before real training starts
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    del _x, _y, _attn, _out
    torch.cuda.empty_cache()
    _crumb("sanity-complete")

    # ── Pre-tokenize the dataset manually ─────────────────────────────────
    # We use a manual training loop instead of trl's SFTTrainer. SFTTrainer
    # 1.1.0 silently exits at step 0 on our multi-turn chat data and we can't
    # debug it — the failure has no Python traceback. A hand-written loop is
    # ~20 lines, fully under our control, and we've already validated
    # (docker_lora_test.py) that forward+backward works with this exact
    # LoRA + grad checkpointing setup.
    print(f"\n=== Tokenizing ===", flush=True)

    PAD = tokenizer.pad_token_id
    IGNORE = -100

    def tokenize_session(messages):
        """Apply chat template, return (input_ids, labels) where labels have
        -100 everywhere except inside assistant spans. Uses the template's
        {% generation %} markers via return_assistant_tokens_mask=True."""
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        input_ids = enc["input_ids"]
        assistant_mask = enc["assistant_masks"]
        # Labels = input_ids where assistant_mask==1, IGNORE elsewhere
        labels = [tok if m == 1 else IGNORE for tok, m in zip(input_ids, assistant_mask)]
        return input_ids, labels

    t0 = time.time()
    train_examples = []
    for ex in train_ds:
        input_ids, labels = tokenize_session(ex["messages"])
        # Truncate tail if over max_length (shouldn't happen after the earlier
        # message-level truncation, but defensive)
        if len(input_ids) > args.max_length:
            input_ids = input_ids[: args.max_length]
            labels = labels[: args.max_length]
        if sum(1 for l in labels if l != IGNORE) == 0:
            continue  # no assistant tokens → nothing to train on
        train_examples.append((input_ids, labels))

    val_examples = []
    for ex in val_ds:
        input_ids, labels = tokenize_session(ex["messages"])
        if len(input_ids) > args.max_length:
            input_ids = input_ids[: args.max_length]
            labels = labels[: args.max_length]
        if sum(1 for l in labels if l != IGNORE) == 0:
            continue
        val_examples.append((input_ids, labels))

    print(f"  train examples after tokenize: {len(train_examples)}", flush=True)
    print(f"  val examples after tokenize: {len(val_examples)}", flush=True)
    print(f"  tokenize time: {time.time()-t0:.1f}s", flush=True)

    if len(train_examples) == 0:
        print("ERROR: no trainable examples. Check chat template + masking.", flush=True)
        return 1

    # Sanity: first example
    first_ids, first_labels = train_examples[0]
    n_label_tok = sum(1 for l in first_labels if l != IGNORE)
    label_positions = [i for i, l in enumerate(first_labels) if l != IGNORE]
    print(f"  first example: input_ids len={len(first_ids)} "
          f"labels non-masked={n_label_tok}", flush=True)
    if label_positions:
        decoded = tokenizer.decode([first_ids[i] for i in label_positions[:16]])
        print(f"  first label tokens decoded: {decoded!r}", flush=True)

    # ── Collator (pads to batch max, sets labels to -100 at pad positions) ─
    def collate(batch):
        max_len = max(len(ids) for ids, _ in batch)
        input_ids_tensor = torch.full((len(batch), max_len), PAD, dtype=torch.long)
        labels_tensor = torch.full((len(batch), max_len), IGNORE, dtype=torch.long)
        attn_tensor = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, (ids, labs) in enumerate(batch):
            L = len(ids)
            input_ids_tensor[i, :L] = torch.tensor(ids, dtype=torch.long)
            labels_tensor[i, :L] = torch.tensor(labs, dtype=torch.long)
            attn_tensor[i, :L] = 1
        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "attention_mask": attn_tensor,
        }

    # ── Training loop ─────────────────────────────────────────────────────
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    # (gradient_checkpointing_enable already called up above after LoRA)

    train_loader = DataLoader(
        train_examples,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_examples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    # Only LoRA params get trained
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.0)

    total_steps = (len(train_examples) // args.batch_size + 1) * int(args.epochs)
    total_steps = max(total_steps // args.grad_accum, 1)
    print(f"  total optimizer steps: ~{total_steps}", flush=True)

    device = next(model.parameters()).device
    print(f"  device: {device}", flush=True)

    print(f"\n=== Training ===", flush=True)
    import traceback
    # Breadcrumb file: survives silent crashes that swallow stdout.
    # Each major phase writes a line here; if the process dies, the last
    # line tells us where it got.
    breadcrumb_path = "/tmp/finetune_breadcrumb.log"
    def crumb(msg):
        with open(breadcrumb_path, "a") as bf:
            bf.write(f"[{time.time():.1f}] {msg}\n")
            bf.flush()
            os.fsync(bf.fileno())
    # Start fresh each run
    with open(breadcrumb_path, "w") as bf:
        bf.write("")
    crumb("training-start")

    model.train()
    crumb("model-set-train")
    global_step = 0
    running_loss = 0.0
    running_n = 0
    t_train = time.time()

    try:
        for epoch in range(int(args.epochs)):
            crumb(f"epoch-{epoch}-start")
            for step, batch in enumerate(train_loader):
                if step < 3:
                    crumb(f"e{epoch}-s{step}-pre-move")
                batch = {k: v.to(device) for k, v in batch.items()}
                if step < 3:
                    crumb(f"e{epoch}-s{step}-pre-forward shape={tuple(batch['input_ids'].shape)}")
                out = model(**batch)
                if step < 3:
                    crumb(f"e{epoch}-s{step}-post-forward loss={out.loss.item():.4f}")
                loss = out.loss / args.grad_accum
                loss.backward()
                if step < 3:
                    crumb(f"e{epoch}-s{step}-post-backward")
                running_loss += out.loss.item()
                running_n += 1

                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % 20 == 0:
                        avg_loss = running_loss / max(running_n, 1)
                        elapsed = time.time() - t_train
                        print(f"  epoch={epoch} step={global_step}/{total_steps} "
                              f"loss={avg_loss:.4f} "
                              f"elapsed={elapsed:.0f}s "
                              f"vram={torch.cuda.memory_allocated()/1e9:.1f}GB",
                              flush=True)
                        running_loss = 0.0
                        running_n = 0

                    if global_step % args.eval_steps == 0 and len(val_examples) > 0:
                        model.eval()
                        val_loss = 0.0
                        val_n = 0
                        with torch.no_grad():
                            for vbatch in val_loader:
                                vbatch = {k: v.to(device) for k, v in vbatch.items()}
                                vout = model(**vbatch)
                                val_loss += vout.loss.item()
                                val_n += 1
                        print(f"  [eval @ step {global_step}] val_loss="
                              f"{val_loss/max(val_n,1):.4f} over {val_n} batches",
                              flush=True)
                        model.train()

        # Final eval
        if len(val_examples) > 0:
            model.eval()
            val_loss = 0.0
            val_n = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = {k: v.to(device) for k, v in vbatch.items()}
                    vout = model(**vbatch)
                    val_loss += vout.loss.item()
                    val_n += 1
            print(f"\n=== Final val_loss: {val_loss/max(val_n,1):.4f} "
                  f"over {val_n} batches ===", flush=True)

    except Exception:
        traceback.print_exc()
        raise
    finally:
        try:
            print(f"\n=== Cleanup ===", flush=True)
            optimizer.zero_grad(set_to_none=True)
        except NameError:
            pass
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        try:
            free_after, _ = torch.cuda.mem_get_info(0)
            print(f"  vram free after cleanup: {free_after/1e9:.1f} GB", flush=True)
        except Exception:
            pass

    if not args.smoke:
        print(f"\n=== Saving LoRA adapter ===", flush=True)
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"  saved: {args.output_dir}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
