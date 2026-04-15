#!/usr/bin/env python3
"""
Minimal fine-tuning script built up from docker_real_test.py.

Strategy: start from the code we verified works (load model, apply LoRA,
forward+backward on one real example), and add a training loop on top.
No trl, no SFTTrainer, no HF datasets, no preflight VRAM check, no chat-
template patching (we compute assistant masks ourselves from the rendered
text). As few moving parts as possible.

Usage (inside docker):
  python benchmarks/finetune_minimal.py --smoke          # 500 examples
  python benchmarks/finetune_minimal.py                  # full run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    ap.add_argument("--train-file", default="data/generated/finetune_train.jsonl")
    ap.add_argument("--val-file", default="data/generated/finetune_val.jsonl")
    ap.add_argument("--output-dir", default="proxy/experiments/phi4_mini_stuck_lora")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--balance-classes", action="store_true",
                    help="oversample S-dominant sessions to match P-dominant count")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"torch: {torch.__version__}", flush=True)
    print(f"hip:   {torch.version.hip}", flush=True)
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print(f"loading tokenizer {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Patch the chat template to add {% generation %} markers around the
    # assistant content. This lets apply_chat_template return an assistant
    # token mask that's a precise bit-vector over the tokenized sequence —
    # no incremental-delta ambiguity. Same patch as docker_real_test.py.
    if "{% generation %}" not in tok.chat_template:
        tok.chat_template = (
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
    print("loading model bf16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    print(f"model loaded  vram={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora, autocast_adapter_dtype=False)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── Load training data from jsonl ──────────────────────────────────────
    # Each line is a session with {session_id, source, messages, n_labeled}
    def load_jsonl(path):
        sessions = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    sessions.append(json.loads(line))
        return sessions

    print("loading jsonl", flush=True)
    train_raw = load_jsonl(REPO / args.train_file)
    val_raw = load_jsonl(REPO / args.val_file)
    if args.smoke:
        train_raw = train_raw[:500]
        val_raw = val_raw[:50]
    print(f"  train: {len(train_raw)}  val: {len(val_raw)}", flush=True)

    # ── Tokenize sessions with assistant mask ──────────────────────────────
    # Strategy: render the chat template WITHOUT the generation marker, then
    # build an assistant mask by finding the positions of each assistant
    # response in the rendered token stream. This avoids needing to patch
    # the chat template.
    #
    # Simpler approach used here: render messages up to-and-including each
    # assistant turn, tokenize incrementally, and mark the delta as the
    # assistant label region. This guarantees we have the exact token boundary.
    IGNORE = -100

    def tokenize_session(messages, max_length):
        """Return (input_ids, labels) where labels has IGNORE everywhere
        except inside assistant response tokens. Uses the patched chat
        template's assistant_masks for precise bit-level masking."""
        # If sequence > max_length, drop tail messages until it fits.
        # We need to be careful to not truncate mid-assistant-turn.
        msgs = list(messages)
        while True:
            enc = tok.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
                return_dict=True,
            )
            input_ids = enc["input_ids"]
            assistant_mask = enc["assistant_masks"]
            if len(input_ids) <= max_length:
                break
            # Drop last (user, assistant) pair
            if len(msgs) >= 3 and msgs[-1]["role"] == "assistant" and msgs[-2]["role"] == "user":
                msgs = msgs[:-2]
            elif len(msgs) >= 2:
                msgs = msgs[:-1]
            else:
                # Can't drop further — hard-truncate tail
                input_ids = input_ids[:max_length]
                assistant_mask = assistant_mask[:max_length]
                break

        labels = [tid if m == 1 else IGNORE for tid, m in zip(input_ids, assistant_mask)]
        return input_ids, labels

    print("tokenizing train", flush=True)
    t0 = time.time()
    train_examples = []
    for sess in train_raw:
        ids, labs = tokenize_session(sess["messages"], args.max_length)
        if any(l != IGNORE for l in labs):
            train_examples.append((ids, labs))
    print(f"  train: {len(train_examples)}  ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    val_examples = []
    for sess in val_raw:
        ids, labs = tokenize_session(sess["messages"], args.max_length)
        if any(l != IGNORE for l in labs):
            val_examples.append((ids, labs))
    print(f"  val: {len(val_examples)}  ({time.time()-t0:.1f}s)", flush=True)

    if not train_examples:
        print("ERROR: no trainable examples", flush=True)
        return 1

    # ── Class balance analysis + optional oversampling ────────────────────
    # Count P vs S labels per example, classify dominance, report distribution.
    P_TOK = tok.encode("P", add_special_tokens=False)[0]
    S_TOK = tok.encode("S", add_special_tokens=False)[0]

    p_dominant = []
    s_dominant = []
    mixed = []
    total_p_labels = 0
    total_s_labels = 0
    for ids, labs in train_examples:
        n_p = sum(1 for l in labs if l == P_TOK)
        n_s = sum(1 for l in labs if l == S_TOK)
        total_p_labels += n_p
        total_s_labels += n_s
        if n_s > n_p:
            s_dominant.append((ids, labs))
        elif n_p > n_s:
            p_dominant.append((ids, labs))
        else:
            mixed.append((ids, labs))

    total_labels = total_p_labels + total_s_labels
    print(f"\n=== class distribution ===", flush=True)
    print(f"  token-level: P={total_p_labels} ({100*total_p_labels/max(total_labels,1):.1f}%)  "
          f"S={total_s_labels} ({100*total_s_labels/max(total_labels,1):.1f}%)", flush=True)
    print(f"  session-level: P-dom={len(p_dominant)}  S-dom={len(s_dominant)}  "
          f"mixed={len(mixed)}", flush=True)

    if args.balance_classes:
        # Oversample whichever class is minority at the SESSION level
        import random
        random.seed(args.seed)
        target_count = max(len(p_dominant), len(s_dominant))
        if len(s_dominant) < target_count:
            extra = random.choices(s_dominant, k=target_count - len(s_dominant))
            s_dominant = s_dominant + extra
        elif len(p_dominant) < target_count:
            extra = random.choices(p_dominant, k=target_count - len(p_dominant))
            p_dominant = p_dominant + extra
        train_examples = p_dominant + s_dominant + mixed
        random.shuffle(train_examples)
        # Recount after rebalance
        new_p = new_s = 0
        for ids, labs in train_examples:
            new_p += sum(1 for l in labs if l == P_TOK)
            new_s += sum(1 for l in labs if l == S_TOK)
        new_total = new_p + new_s
        print(f"  after oversample: {len(train_examples)} sessions, "
              f"P={new_p} ({100*new_p/max(new_total,1):.1f}%) "
              f"S={new_s} ({100*new_s/max(new_total,1):.1f}%)", flush=True)

    first_ids, first_labs = train_examples[0]
    n_label_tok = sum(1 for l in first_labs if l != IGNORE)
    print(f"first example: len={len(first_ids)}  labels={n_label_tok}", flush=True)
    sample_labels = [first_ids[i] for i, l in enumerate(first_labs) if l != IGNORE][:16]
    if sample_labels:
        print(f"  first label tokens: {tok.decode(sample_labels)!r}", flush=True)

    # ── Training loop ──────────────────────────────────────────────────────
    from torch.optim import AdamW
    PAD = tok.pad_token_id

    def collate(batch):
        max_len = max(len(ids) for ids, _ in batch)
        n = len(batch)
        input_ids = torch.full((n, max_len), PAD, dtype=torch.long)
        labels = torch.full((n, max_len), IGNORE, dtype=torch.long)
        attn = torch.zeros((n, max_len), dtype=torch.long)
        for i, (ids, labs) in enumerate(batch):
            L = len(ids)
            input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
            labels[i, :L] = torch.tensor(labs, dtype=torch.long)
            attn[i, :L] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.0)

    import random
    random.seed(args.seed)

    device = next(model.parameters()).device
    total_examples = len(train_examples) * args.epochs
    total_opt_steps = max(total_examples // args.batch_size // args.grad_accum, 1)
    print(f"training: {len(train_examples)} × {args.epochs} epochs / "
          f"({args.batch_size} × {args.grad_accum}) ≈ {total_opt_steps} opt steps",
          flush=True)

    def run_val():
        """Run a pass over val set, return (avg_loss, n_batches, pred_counts)
        where pred_counts is the argmax breakdown over P/S/U at labeled positions."""
        model.eval()
        total_loss = 0.0
        n_batches = 0
        p_cnt = s_cnt = u_cnt = 0
        u_tok = tok.encode("U", add_special_tokens=False)[0]
        with torch.no_grad():
            for vi in range(0, len(val_examples), args.batch_size):
                vbatch = collate(val_examples[vi : vi + args.batch_size])
                vbatch = {k: v.to(device) for k, v in vbatch.items()}
                vout = model(**vbatch)
                total_loss += vout.loss.item()
                n_batches += 1
                labs = vbatch["labels"]
                mask = labs != IGNORE
                if mask.any():
                    logits = vout.logits
                    psu = torch.stack([
                        logits[..., P_TOK],
                        logits[..., S_TOK],
                        logits[..., u_tok],
                    ], dim=-1)
                    preds = psu.argmax(dim=-1)[mask]
                    p_cnt += (preds == 0).sum().item()
                    s_cnt += (preds == 1).sum().item()
                    u_cnt += (preds == 2).sum().item()
        model.train()
        return total_loss / max(n_batches, 1), n_batches, {"P": p_cnt, "S": s_cnt, "U": u_cnt}

    model.train()
    global_step = 0
    running_loss = 0.0
    running_n = 0
    val_history = []
    t_train = time.time()
    eval_every = max(total_opt_steps // 8, 50)
    print(f"val checks every {eval_every} optimizer steps", flush=True)

    for epoch in range(args.epochs):
        idxs = list(range(len(train_examples)))
        random.shuffle(idxs)

        for step_in_epoch in range(0, len(idxs), args.batch_size):
            batch_idxs = idxs[step_in_epoch : step_in_epoch + args.batch_size]
            batch = collate([train_examples[i] for i in batch_idxs])
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running_loss += out.loss.item()
            running_n += 1

            if running_n % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    avg = running_loss / running_n
                    elapsed = time.time() - t_train
                    vram = torch.cuda.memory_allocated() / 1e9
                    print(f"  e{epoch} step {global_step}/{total_opt_steps} "
                          f"loss={avg:.4f} elapsed={elapsed:.0f}s vram={vram:.1f}GB",
                          flush=True)
                    running_loss = 0.0
                    running_n = 0

                if val_examples and (global_step % eval_every == 0):
                    v_loss, v_n, v_preds = run_val()
                    val_history.append((global_step, v_loss, v_preds))
                    total_preds = sum(v_preds.values())
                    p_pct = 100 * v_preds["P"] / max(total_preds, 1)
                    s_pct = 100 * v_preds["S"] / max(total_preds, 1)
                    print(f"  [VAL @ step {global_step}] loss={v_loss:.4f} "
                          f"preds: P={v_preds['P']}({p_pct:.0f}%) "
                          f"S={v_preds['S']}({s_pct:.0f}%) "
                          f"U={v_preds['U']}",
                          flush=True)

    # Final eval
    if val_examples:
        v_loss, v_n, v_preds = run_val()
        val_history.append((global_step, v_loss, v_preds))
        print(f"\nfinal val_loss = {v_loss:.4f} over {v_n} batches", flush=True)

    print(f"\n=== val history ===", flush=True)
    print(f"{'step':>8}{'val_loss':>12}{'P%':>8}{'S%':>8}{'U%':>8}", flush=True)
    for s, l, p in val_history:
        tot = sum(p.values()) or 1
        print(f"{s:>8}{l:>12.4f}"
              f"{100*p['P']/tot:>8.1f}{100*p['S']/tot:>8.1f}{100*p['U']/tot:>8.1f}",
              flush=True)
    if val_history:
        best = min(val_history, key=lambda x: x[1])
        print(f"best val_loss: step {best[0]}  loss {best[1]:.4f}", flush=True)

    if not args.smoke:
        out_dir = REPO / args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        print(f"saved: {out_dir}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
