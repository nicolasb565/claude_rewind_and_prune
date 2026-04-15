#!/usr/bin/env python3
"""
Evaluate a fine-tuned phi4-mini LoRA adapter on the OOD stuck-detection
benchmark. Mirrors slm_stuck.py's windowed framing so numbers are directly
comparable to the zero-shot / few-shot baselines.

Usage (inside the rocm/pytorch container):
  python benchmarks/finetune_eval.py \
      --adapter proxy/experiments/phi4_mini_stuck_lora

Runs ~682 forward passes on the OOD benchmark, greedy-samples one token
per step, parses as P/S/U, computes P/R/F1/AUC against Sonnet ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    ap.add_argument("--adapter", default="proxy/experiments/phi4_mini_stuck_lora")
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--context-steps", type=int, default=5)
    ap.add_argument("--max-output-chars", type=int, default=400)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from sklearn.metrics import roc_auc_score
    import numpy as np

    from src.pipeline.parsers.nlile import parse_session
    from src.pipeline.label_session import _render_step

    RUN_DIR = REPO / "benchmarks" / "results" / "comparison_off"

    print(f"torch: {torch.__version__}  hip: {torch.version.hip}", flush=True)
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

    # ── Load base + adapter ───────────────────────────────────────────────
    adapter_dir = REPO / args.adapter
    print(f"loading tokenizer from {adapter_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained(str(adapter_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"loading base model {args.model} bf16", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )
    base.config.use_cache = True  # enable KV cache for fast inference

    print(f"loading LoRA adapter from {adapter_dir}", flush=True)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    print(f"model ready  vram={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # Same system prompt as training (short variant, matches slm_stuck.py)
    SYSTEM_PROMPT = """\
You are evaluating a Claude Code coding session for stuck detection.

Classify the TARGET step (the last step shown) as exactly one letter:
  P - productive: the agent is making progress (new approach, first-time \
action, finding new bugs, legitimate iterative build/test loops with fixes \
in between)
  S - stuck: the agent is in a loop (same command/error/edit repeated \
without progress, trying the same thing from different angles)
  U - unsure: genuine ambiguity that cannot be resolved from the context

Rules:
- First attempt at any command → P
- Legitimate build/test iteration where fixes are being made between runs → P
- Same command with same error twice or more, no visible change → S
- Different approach or tool after a failure → P (first step of the new approach)
- Agent hitting the same underlying error from different files/angles → S

Output format: reply with a single character — P, S, or U — and NOTHING ELSE."""

    def render_step(step, i):
        rendered = _render_step(step, i)
        lines = rendered.split("\n")
        out = []
        for ln in lines:
            if ln.startswith("  → ") and len(ln) > args.max_output_chars:
                ln = ln[: args.max_output_chars] + " [...]"
            out.append(ln)
        return "\n".join(out)

    def build_messages(context_steps, target_step, target_index):
        """Build the same windowed chat format as slm_stuck.py — system + user
        (context + target) → assistant will classify with one letter."""
        parts = [SYSTEM_PROMPT, ""]
        if context_steps:
            parts.append("Context (prior steps):")
            start = target_index - len(context_steps)
            for offset, s in enumerate(context_steps):
                parts.append(render_step(s, start + offset))
        parts.append("")
        parts.append("TARGET step:")
        parts.append(render_step(target_step, target_index))
        parts.append("")
        parts.append("Your single-character label (P, S, or U):")
        return [
            {"role": "user", "content": "\n".join(parts)},
        ]

    def parse_transcript(path):
        messages = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") in ("user", "assistant"):
                m = ev.get("message", {})
                if isinstance(m, dict):
                    messages.append(m)
        return parse_session(messages)

    # ── Iterate OOD benchmark ─────────────────────────────────────────────
    tasks = []
    for td in sorted(RUN_DIR.iterdir()):
        if not td.is_dir():
            continue
        t = td / "transcript_1.jsonl"
        lp = td / "sonnet_labels.json"
        if not (t.exists() and lp.exists()):
            continue
        steps = parse_transcript(t)
        labels = json.loads(lp.read_text())["labels"]
        n = min(len(steps), len(labels))
        tasks.append((td.name, steps[:n], labels[:n]))

    print(f"loaded {len(tasks)} tasks, "
          f"{sum(len(s) for _, s, _ in tasks)} total steps", flush=True)

    # Find token IDs for P, S, U (used for greedy argmax over 3 tokens only)
    p_id = tok.encode("P", add_special_tokens=False)[0]
    s_id = tok.encode("S", add_special_tokens=False)[0]
    u_id = tok.encode("U", add_special_tokens=False)[0]
    print(f"token ids: P={p_id} S={s_id} U={u_id}", flush=True)

    all_preds = []
    all_labels = []
    all_scores = []  # P(S) for AUC computation
    per_task = []

    t_start = time.time()
    total_steps = sum(len(s) for _, s, _ in tasks)
    progress = 0

    for task_name, steps, labels in tasks:
        task_preds = []
        task_scores = []
        for i, step in enumerate(steps):
            context = steps[max(0, i - args.context_steps): i]
            messages = build_messages(context, step, i)
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tok(
                text, return_tensors="pt", truncation=True,
                max_length=args.max_length,
            ).to("cuda:0")

            with torch.no_grad():
                out = model(**inputs)
            # Last position logits → softmax over P/S/U only
            logits = out.logits[0, -1]
            # Extract just the three token probs for stable argmax
            psu_logits = torch.stack([logits[p_id], logits[s_id], logits[u_id]])
            psu_probs = torch.softmax(psu_logits, dim=-1)
            pred_idx = psu_probs.argmax().item()
            pred = ["P", "S", "U"][pred_idx]
            s_prob = psu_probs[1].item()

            task_preds.append(pred)
            task_scores.append(s_prob)
            progress += 1
            if progress % 50 == 0:
                elapsed = time.time() - t_start
                rate = progress / elapsed
                eta = (total_steps - progress) / rate
                print(f"  {progress}/{total_steps}  rate={rate:.1f}/s  "
                      f"eta={eta:.0f}s", flush=True)

        # Per-task
        tp = sum(1 for p, g in zip(task_preds, labels) if p == "S" and g == "STUCK")
        fp = sum(1 for p, g in zip(task_preds, labels) if p == "S" and g == "PRODUCTIVE")
        fn = sum(1 for p, g in zip(task_preds, labels) if p != "S" and g == "STUCK")
        n_stk = sum(1 for g in labels if g == "STUCK")
        per_task.append({
            "task": task_name, "n": len(steps), "stk": n_stk,
            "tp": tp, "fp": fp, "fn": fn,
        })

        for p, g, sc in zip(task_preds, labels, task_scores):
            if g == "UNSURE":
                continue
            all_preds.append(1 if p == "S" else 0)
            all_labels.append(1 if g == "STUCK" else 0)
            all_scores.append(sc)

    # ── Pooled metrics ────────────────────────────────────────────────────
    print(f"\n{'task':<22}{'n':>5}{'stk':>5}{'tp':>5}{'fp':>5}{'fn':>5}")
    for r in per_task:
        print(f"{r['task']:<22}{r['n']:>5}{r['stk']:>5}"
              f"{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}")

    arr_p = np.array(all_preds)
    arr_l = np.array(all_labels)
    arr_sc = np.array(all_scores)

    if 0 < arr_l.sum() < len(arr_l):
        auc_binary = roc_auc_score(arr_l, arr_p)
        auc_cont = roc_auc_score(arr_l, arr_sc)
    else:
        auc_binary = auc_cont = float("nan")

    tp = int(((arr_p == 1) & (arr_l == 1)).sum())
    fp = int(((arr_p == 1) & (arr_l == 0)).sum())
    fn = int(((arr_p == 0) & (arr_l == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    total_elapsed = time.time() - t_start
    print(f"\n=== fine-tuned phi4-mini pooled ===")
    print(f"  AUC (binary argmax): {auc_binary:.4f}")
    print(f"  AUC (S-prob score):  {auc_cont:.4f}")
    print(f"  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  total: {total_elapsed:.0f}s  ({total_elapsed/total_steps*1000:.0f}ms/step)")
    print(f"\n  LR baseline:         F1=0.326  AUC=0.736 (for reference)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
