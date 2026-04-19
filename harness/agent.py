"""Agent loop: render messages → generate → parse tool_call → execute → repeat.

One agent per (fixture, mode) run. The caller provides:
- a preloaded (model, tokenizer, tools_schema)
- a working directory
- a goal string
- a mode: "baseline" (no hygiene) or "hygiene" (elide on checkpoint)
"""
from __future__ import annotations

import time
from pathlib import Path

from harness.parse import parse_tool_call, is_checkpoint
from harness.tools import ToolRunner
from harness.hygiene import apply_checkpoint_elision


SYSTEM_PROMPT = (
    "You are an engineer working in a sandboxed repository. Use the provided "
    "tools to diagnose and fix issues. When you achieve a concrete milestone "
    "or rule out an approach, emit a checkpoint_progress tool call to consolidate "
    "progress. Finish the task when the stated success criterion is met."
)


def run_agent(
    *,
    model,
    tokenizer,
    tools_schema: list[dict],
    work_dir: Path,
    goal: str,
    mode: str,
    max_steps: int = 30,
    max_new_tokens: int = 256,
    max_context_tokens: int = 8000,
    temperature: float = 0.0,
    top_p: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Run one agent session. Returns metrics dict.

    Modes:
      baseline: base model only (adapter disabled throughout)
      hygiene:  adapter drives the whole turn — if checkpoint, elide
      sidecar:  base model picks the action; adapter runs in parallel
                purely to decide "is this a checkpoint moment?". On yes,
                append a checkpoint event + elide; otherwise skip adapter.
                Base's action executes either way.
    """
    assert mode in ("baseline", "hygiene", "sidecar")
    tools = ToolRunner(work_dir)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]

    total_input_tokens = 0
    max_context_size = 0
    n_checkpoints = 0
    n_tool_calls = 0
    n_elided = 0
    t0 = time.time()
    stop_reason = "max_steps"

    for step in range(max_steps):
        # Render prompt
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tools=tools_schema,
                tokenize=False, add_generation_prompt=True,
            )
        except Exception as e:
            stop_reason = f"template_error: {e}"
            break

        enc = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")
        n_prompt_tokens = int(enc.input_ids.shape[1])
        total_input_tokens += n_prompt_tokens
        max_context_size = max(max_context_size, n_prompt_tokens)

        if n_prompt_tokens > max_context_tokens:
            stop_reason = f"context_overflow ({n_prompt_tokens} > {max_context_tokens})"
            if verbose:
                print(f"  [step {step}] CONTEXT OVERFLOW: {n_prompt_tokens} tokens")
            break

        if verbose:
            print(f"  [step {step}] ctx={n_prompt_tokens}t, msgs={len(messages)}")

        import torch
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)

        # Generate the primary action.
        # - baseline: adapter DISABLED the whole time (base model only)
        # - hygiene:  adapter always active, drives the turn
        # - sidecar:  base model picks the action (adapter disabled)
        with torch.no_grad():
            if mode in ("baseline", "sidecar") and hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    out = model.generate(enc.input_ids, **gen_kwargs)
            else:
                out = model.generate(enc.input_ids, **gen_kwargs)
        gen_text = tokenizer.decode(
            out[0, int(enc.input_ids.shape[1]):],
            skip_special_tokens=False,
        )
        tool_call = parse_tool_call(gen_text)

        # Sidecar mode: ask the adapter if this is a checkpoint moment.
        # If yes, emit a synthetic checkpoint event BEFORE the base's action.
        if mode == "sidecar" and hasattr(model, "disable_adapter"):
            with torch.no_grad():
                sc_out = model.generate(enc.input_ids, **gen_kwargs)
            sc_text = tokenizer.decode(
                sc_out[0, int(enc.input_ids.shape[1]):],
                skip_special_tokens=False,
            )
            sc_call = parse_tool_call(sc_text)
            if sc_call and is_checkpoint(sc_call):
                n_checkpoints += 1
                if verbose:
                    fi = sc_call["arguments"].get("finding", "")[:80]
                    print(f"  [step {step}] SIDECAR CHECKPOINT: {fi!r}")
                ckpt_id = f"call_sc_{step}"
                messages.append({
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": ckpt_id, "type": "function",
                        "function": {"name": sc_call["name"], "arguments": sc_call["arguments"]},
                    }],
                })
                messages.append({
                    "role": "tool", "tool_call_id": ckpt_id,
                    "name": sc_call["name"],
                    "content": "Checkpoint saved. Prior exploration may now be condensed.",
                })
                elided = apply_checkpoint_elision(messages, sc_call["arguments"])
                n_elided += elided
                if verbose and elided:
                    print(f"    → elided {elided} prior tool_results")
                # Base's tool_call still executes below (the checkpoint was
                # additive — base agent continues the work on the compressed
                # history).

        if tool_call is None:
            # Model ended turn with text only — consider task done or stuck
            stop_reason = "no_tool_call"
            if verbose:
                print(f"  [step {step}] NO TOOL CALL. Output: {gen_text[:150]!r}")
            break

        call_id = f"call_{step}"

        if is_checkpoint(tool_call):
            n_checkpoints += 1
            if verbose:
                fi = tool_call["arguments"].get("finding", "")[:80]
                print(f"  [step {step}] CHECKPOINT: {fi!r}")

            # Append the checkpoint call + response (shared between modes)
            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": call_id, "type": "function",
                    "function": {"name": tool_call["name"], "arguments": tool_call["arguments"]},
                }],
            })
            messages.append({
                "role": "tool", "tool_call_id": call_id,
                "name": tool_call["name"],
                "content": "Checkpoint saved. Prior exploration may now be condensed.",
            })

            # Hygiene: elide prior tool_results
            if mode == "hygiene":
                elided = apply_checkpoint_elision(messages, tool_call["arguments"])
                n_elided += elided
                if verbose and elided:
                    print(f"    → elided {elided} prior tool_results")
            continue

        # Normal tool execution
        n_tool_calls += 1
        result = tools.run(tool_call["name"], tool_call["arguments"])
        if verbose:
            # Log tool name + first arg (abbreviated) so we can see variety
            arg_key = next(iter(tool_call["arguments"]), None)
            arg_val = str(tool_call["arguments"].get(arg_key, ""))[:80]
            print(f"    → {tool_call['name']}({arg_key}={arg_val!r})")

        messages.append({
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": tool_call["name"], "arguments": tool_call["arguments"]},
            }],
        })
        messages.append({
            "role": "tool", "tool_call_id": call_id,
            "name": tool_call["name"], "content": result,
        })

    return {
        "mode": mode,
        "stop_reason": stop_reason,
        "steps": step + 1,
        "n_tool_calls": n_tool_calls,
        "n_checkpoints": n_checkpoints,
        "n_elided": n_elided,
        "total_input_tokens": total_input_tokens,
        "max_context_size": max_context_size,
        "wall_time_s": time.time() - t0,
        "final_messages": len(messages),
    }
