#!/usr/bin/env python3
"""
Three mini-smokes for the shadow-logging harness:

1. batch_2_timing:   batch size 2 with divergent prompts, wall-time vs batch 1.
                     Answers: is the "free shadow" assumption true in practice?
2. thinking_off:     enable_thinking=False produces a clean, direct summary.
                     Answers: can we silence the default reasoning preamble?
3. tool_call_sanity: apply_chat_template with a tools list; model emits a
                     structured tool call (not XML/free text).
                     Answers: is the agent-loop tool path viable?

Each test prints PASS/FAIL with a short reason. Exit code = number of fails.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import torch

MODEL_ID = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3.5-4B")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def vram() -> str:
    if not torch.cuda.is_available():
        return "(no cuda)"
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1 << 30)
    tot = total / (1 << 30)
    return f"{used:.2f}/{tot:.2f} GiB"


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.eval()
    return model, tok


def build_prefix_context():
    """~400-token shared context simulating mid-session agent state."""
    return [
        {"role": "system", "content": "You are a debugging assistant working on a C project."},
        {"role": "user", "content": "The function `compute_offset` returns -1 for inputs > 2^16. Find and fix the bug."},
        {"role": "assistant", "content": "I'll investigate. Let me search for the function."},
        {"role": "user", "content": "Tool result: compute_offset is defined in src/offset.c line 42. It uses int16_t for the accumulator.\nint16_t acc = 0; for (int i = 0; i < n; i++) acc += arr[i];\nreturn acc;"},
        {"role": "assistant", "content": "The accumulator type is too small. For n > ~2^15 / max(arr) it overflows. I should widen it to int32_t."},
        {"role": "user", "content": "Tool result: I changed acc to int32_t and rebuilt. Tests now pass for n=65536."},
    ]


def test_batch_2_timing(model, tok) -> tuple[bool, str]:
    """Compare batch 1 generation vs batch 2 generation with divergent prompts."""
    ctx = build_prefix_context()
    # Two divergent continuations on the same prefix.
    ctx_a = ctx + [{"role": "user", "content": "What's the next action you'd take?"}]
    ctx_b = ctx + [{"role": "user", "content": "Without acting, write a checkpoint summary of progress so far in 2 sentences."}]

    text_a = tok.apply_chat_template(ctx_a, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    text_b = tok.apply_chat_template(ctx_b, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    enc_a = tok(text_a, return_tensors="pt", padding=True).to("cuda:0")
    enc_b = tok(text_b, return_tensors="pt", padding=True).to("cuda:0")
    batch = tok([text_a, text_b], return_tensors="pt", padding=True).to("cuda:0")

    # Warmup (first kernel launch pays a cost)
    with torch.no_grad():
        _ = model.generate(enc_a.input_ids, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()

    MAX_NEW = 40

    # Batch 1 for each separately
    t0 = time.time()
    with torch.no_grad():
        out_a = model.generate(enc_a.input_ids, attention_mask=enc_a.attention_mask, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt_a = time.time() - t0

    t0 = time.time()
    with torch.no_grad():
        out_b = model.generate(enc_b.input_ids, attention_mask=enc_b.attention_mask, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt_b = time.time() - t0

    # Batch 2 (both at once)
    t0 = time.time()
    with torch.no_grad():
        out_batch = model.generate(batch.input_ids, attention_mask=batch.attention_mask, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt_batch = time.time() - t0

    sequential = dt_a + dt_b
    speedup = sequential / dt_batch if dt_batch > 0 else 0.0
    log(f"  batch=1 (A): {dt_a:.2f}s")
    log(f"  batch=1 (B): {dt_b:.2f}s")
    log(f"  sequential (A then B): {sequential:.2f}s")
    log(f"  batch=2 (together):    {dt_batch:.2f}s")
    log(f"  speedup vs sequential: {speedup:.2f}x   (1.0x = no savings, 2.0x = ideal)")
    log(f"  VRAM after: {vram()}")

    text_a_out = tok.decode(out_a[0, enc_a.input_ids.shape[1]:], skip_special_tokens=True)[:150]
    text_b_out = tok.decode(out_b[0, enc_b.input_ids.shape[1]:], skip_special_tokens=True)[:150]
    log(f"  A out: {text_a_out!r}")
    log(f"  B out: {text_b_out!r}")

    # Pass if batch 2 at least 1.3x faster than sequential (generous threshold).
    ok = speedup >= 1.3
    return ok, f"speedup={speedup:.2f}x (need >=1.3)"


def test_thinking_off(model, tok) -> tuple[bool, str]:
    """Check enable_thinking=False produces non-thinking output."""
    ctx = build_prefix_context()
    ctx = ctx + [{"role": "user", "content": "Write a one-sentence checkpoint summary of progress so far."}]

    text = tok.apply_chat_template(ctx, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        out = model.generate(enc.input_ids, max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
    decoded = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    log(f"  output ({len(decoded)} chars): {decoded!r}")

    has_thinking_preamble = bool(re.search(r"(?i)thinking process|let me analyze|^\s*1\.\s*\*\*", decoded[:200]))
    has_think_tags = "<think>" in decoded.lower()
    ok = not has_thinking_preamble and not has_think_tags and len(decoded.strip()) > 10
    reason = f"preamble={has_thinking_preamble} think_tags={has_think_tags} len={len(decoded.strip())}"
    return ok, reason


def test_tool_call_sanity(model, tok) -> tuple[bool, str]:
    """Verify the model emits structured tool calls when given tools in the template."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a source file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "description": "Path to the file"}},
                    "required": ["path"],
                },
            },
        }
    ]
    msgs = [
        {"role": "system", "content": "You are a debugging assistant. Use tools when needed."},
        {"role": "user", "content": "Read the file src/offset.c so we can find the bug in compute_offset."},
    ]
    try:
        text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception as e:
        return False, f"chat template rejected tools: {type(e).__name__}: {str(e)[:150]}"

    enc = tok(text, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(enc.input_ids, max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
    decoded = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    log(f"  output: {decoded!r}")

    # Qwen formats: <tool_call>{...}</tool_call>  or similar. Also check for XML-tag-only degraded output.
    has_tool_call_tag = "<tool_call>" in decoded
    # Extract the JSON between <tool_call>...</tool_call> if present
    parsed = None
    valid_json = False
    if has_tool_call_tag:
        match = re.search(r"<tool_call>\s*(.+?)\s*</tool_call>", decoded, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                valid_json = True
            except Exception:
                valid_json = False
    has_read_file = "read_file" in decoded
    has_path_arg = "offset.c" in decoded

    ok = has_tool_call_tag and valid_json and has_read_file and has_path_arg
    reason = f"tag={has_tool_call_tag} valid_json={valid_json} name_ok={has_read_file} arg_ok={has_path_arg}"
    if parsed:
        log(f"  parsed tool call: {parsed}")
    return ok, reason


def main() -> int:
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    log(f"torch {torch.__version__} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    log(f"loading {MODEL_ID}")
    model, tok = load()
    log(f"loaded. vram: {vram()}")

    fails = 0
    for name, fn in [
        ("batch_2_timing", test_batch_2_timing),
        ("thinking_off", test_thinking_off),
        ("tool_call_sanity", test_tool_call_sanity),
    ]:
        log(f"--- {name} ---")
        try:
            ok, reason = fn(model, tok)
            status = "PASS" if ok else "FAIL"
            log(f"{name}: {status}  ({reason})")
            if not ok:
                fails += 1
        except Exception as e:
            log(f"{name}: ERROR  {type(e).__name__}: {str(e)[:200]}")
            import traceback; traceback.print_exc()
            fails += 1

    log(f"=== {3 - fails}/3 tests passed ===")
    return fails


if __name__ == "__main__":
    sys.exit(main())
