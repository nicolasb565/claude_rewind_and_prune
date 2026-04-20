#!/usr/bin/env python3
"""
Compatibility probe — does our v6 training pipeline transfer unchanged
to Qwen 3.5 9B?

Checks:
  1. Tokenizer loads
  2. Chat template accepts tools= kwarg and our TOOLS_SCHEMA
  3. Template supports enable_thinking=False (or equivalent)
  4. render_for_gemma's prompt/completion split holds:
       full_text.startswith(prompt_text)
  5. Positive and negative completions share a common opener
     (that's the whole v6 insight — both must start with the same
     tool_call marker for the model to learn name-only discrimination)
  6. Completion contains the expected tool_call structure
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline.verify_gemma_tokenization import TOOLS_SCHEMA


def apply(tok, msgs, *, add_generation_prompt: bool):
    for kwargs in (
        dict(enable_thinking=False),
        dict(),  # fallback if template doesn't accept enable_thinking
    ):
        try:
            return tok.apply_chat_template(
                msgs, tools=TOOLS_SCHEMA, tokenize=False,
                add_generation_prompt=add_generation_prompt, **kwargs,
            )
        except TypeError:
            continue
    raise RuntimeError("chat template rejected all kwargs")


def main():
    from transformers import AutoTokenizer

    model_id = "Qwen/Qwen3.5-9B"
    print(f"=== probing {model_id} ===\n")

    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"FAIL loading tokenizer: {e}")
        return 1
    print(f"tokenizer: vocab={tok.vocab_size}  chat_template set: {tok.chat_template is not None}")

    # Check for thinking-mode tokens that could break the prompt/completion split
    for tok_name in ("<think>", "</think>", "<|think|>", "<|/think|>"):
        ids = tok(tok_name, add_special_tokens=False)["input_ids"]
        if ids and len(ids) <= 3:
            print(f"  thinking token present: {tok_name!r} -> {ids}")

    # Load one positive + one negative chunk from our data
    rows = [json.loads(l) for l in open("data/generated/hygiene_v1.chunks.jsonl")]
    pos = next(r for r in rows if r.get("meta", {}).get("label") == "positive")
    neg = next(r for r in rows if r.get("meta", {}).get("label") == "negative")

    print("\n--- positive sample ---")
    check_chunk(tok, pos, "positive")
    print("\n--- negative sample ---")
    check_chunk(tok, neg, "negative")

    # Common-opener check: do both completions start with the same N tokens?
    pos_full = apply(tok, pos["messages"], add_generation_prompt=False)
    pos_prompt = apply(tok, pos["messages"][:-2], add_generation_prompt=True)
    neg_full = apply(tok, neg["messages"], add_generation_prompt=False)
    # Negative's "completion" is the last 1-2 msgs (v6: always tool_call + tool_response)
    neg_prompt = apply(tok, neg["messages"][:-2], add_generation_prompt=True)

    pos_completion = pos_full[len(pos_prompt):] if pos_full.startswith(pos_prompt) else "<SPLIT-FAILED>"
    neg_completion = neg_full[len(neg_prompt):] if neg_full.startswith(neg_prompt) else "<SPLIT-FAILED>"

    print("\n--- completion openers (first 80 chars) ---")
    print(f"positive: {pos_completion[:80]!r}")
    print(f"negative: {neg_completion[:80]!r}")

    common_prefix_len = 0
    for a, b in zip(pos_completion, neg_completion):
        if a != b:
            break
        common_prefix_len += 1
    print(f"common prefix length: {common_prefix_len} chars")
    print(f"common prefix: {pos_completion[:common_prefix_len]!r}")

    if "<tool_call>" not in pos_completion:
        print("\nWARN: Qwen3.5 may use a different tool_call marker")
        print(f"  positive completion start: {pos_completion[:200]!r}")

    return 0


def check_chunk(tok, chunk, kind):
    msgs = chunk["messages"]
    print(f"  {len(msgs)} messages, meta: {chunk.get('meta', {})}")
    try:
        prompt_text = apply(tok, msgs[:-2], add_generation_prompt=True)
        full_text = apply(tok, msgs, add_generation_prompt=False)
    except Exception as e:
        print(f"  FAIL template: {e}")
        return
    if not full_text.startswith(prompt_text):
        # Diagnostic: where does it diverge?
        for i in range(min(len(prompt_text), len(full_text))):
            if prompt_text[i] != full_text[i]:
                print(f"  FAIL prefix check at char {i}")
                print(f"    prompt around: ...{prompt_text[max(0,i-30):i+30]!r}")
                print(f"    full around:   ...{full_text[max(0,i-30):i+30]!r}")
                return
        print("  prompt is not a prefix, but no diverge found — length issue")
        return
    completion = full_text[len(prompt_text):]
    print(f"  OK split. prompt={len(prompt_text)} chars, completion={len(completion)} chars")
    print(f"  completion[:120]: {completion[:120]!r}")


if __name__ == "__main__":
    sys.exit(main())
