"""Agent loop with batch-2 shadow logging.

Phase 1 of the "shadow summary" experiment: at each turn we ask the model
twice at batch=2 — once to take the next agent action (batch 1) and once
to emit a structured shadow analysis (batch 2). The shadow output is
logged but never acted on. This answers two questions cheaply, in one
run:

  1. Does the model produce coherent self-summaries across many turns?
  2. Does its own "should I checkpoint now?" signal fire at sensible moments?

Acting on the shadow output is deferred to phase 2.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import torch

from harness.parse import parse_tool_call
from harness.tools import ToolRunner


AGENT_SYSTEM_PROMPT = (
    "You are an engineer working in a sandboxed repository. Use the provided "
    "tools to diagnose and fix issues. Finish the task when the stated "
    "success criterion is met."
)


SHADOW_QUERY = (
    "STOP. You are now in REFLECTION mode, NOT action mode. Do NOT emit any\n"
    "tool_call tags. Your entire response must be plain text starting with\n"
    "'SHOULD_CHECKPOINT:' and using exactly these three labeled fields:\n\n"
    "SHOULD_CHECKPOINT: YES if the most recent tool_result just provided\n"
    "concrete evidence of a verified outcome. Two firing cases:\n"
    "  - A change was just made AND verified by an observable signal (a\n"
    "    test passing, a build succeeding, expected program output, a\n"
    "    required artifact present, or equivalent confirmation). If the\n"
    "    original task has multiple independent goals, the verified\n"
    "    completion of ANY ONE of them fires YES — do not wait for the\n"
    "    whole task to finish before firing.\n"
    "  - An approach was proven wrong — a concrete failure means the\n"
    "    current direction cannot work and the session must change\n"
    "    strategy.\n"
    "A file write, read, edit, or search is an ACTION, not an outcome —\n"
    "those cannot justify YES on their own. Only the tool_result that\n"
    "VERIFIES the change counts.\n"
    "NO only if: the most recent change has not yet been verified, OR no\n"
    "change has happened yet. Do NOT answer NO merely because OTHER\n"
    "pending work remains elsewhere in the task — that is a separate\n"
    "question for future turns. Focus on whether the LAST change just\n"
    "got verified.\n"
    "REASON: <one-line explanation, <=20 words, citing the specific evidence>\n"
    "SUMMARY: <2-3 sentences that MUST include concrete references the\n"
    "agent will need to continue: specific file paths, function names,\n"
    "test names, or other identifiers. Cover: (1) what was verified\n"
    "completed and WHERE (which file/module/function), (2) what work\n"
    "remains in the original task and WHERE, (3) the next concrete step.\n"
    "Do NOT write vague references like 'the function' or 'the module';\n"
    "use locators appropriate to the task domain — e.g. a relative file\n"
    "path with a symbol name ('path/to/file.ext::symbol'), a test id, an\n"
    "issue/PR identifier, or a concrete build target.>\n"
)


_SHOULD_RE = re.compile(r"SHOULD_CHECKPOINT:\s*(YES|NO)\b", re.I)
_REASON_RE = re.compile(r"REASON:\s*(.+)")
_SUMMARY_RE = re.compile(r"SUMMARY:\s*(.+)", re.DOTALL)


def parse_shadow(text: str) -> dict:
    should = _SHOULD_RE.search(text)
    reason = _REASON_RE.search(text)
    summary = _SUMMARY_RE.search(text)
    return {
        "should": (should.group(1).upper() if should else None),
        "reason": (reason.group(1).strip().splitlines()[0] if reason else None),
        "summary": (summary.group(1).strip() if summary else None),
        "raw": text,
    }


def _decode_gen(tok, ids: torch.Tensor, input_len: int) -> str:
    """Decode only the newly generated portion."""
    return tok.decode(ids[input_len:], skip_special_tokens=True)


def apply_rewind(messages: list[dict], shadow_summary: str) -> tuple[list[dict], int]:
    """Compress messages: keep system + goal + most recent assistant+tool
    pair, then splice a synthetic user message containing the shadow summary.

    Returns (new_messages, n_elided) where n_elided is the count of dropped
    intermediate messages. Returns unchanged if nothing to compress.
    """
    if len(messages) < 4:
        return messages, 0

    last_tool_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "tool":
            last_tool_idx = i
            break
    if last_tool_idx is None or last_tool_idx < 3:
        return messages, 0

    last_assist_idx = last_tool_idx - 1
    preserved = messages[:2] + messages[last_assist_idx:last_tool_idx + 1]
    preserved.append({
        "role": "user",
        "content": (
            f"[Checkpoint note from your prior reasoning]\n"
            f"{shadow_summary}\n\n"
            f"The turn immediately above shows the verification evidence. "
            f"Continue with any remaining work from the original task."
        ),
    })
    n_elided = last_assist_idx - 2
    return preserved, n_elided


def run_shadow_agent(
    *,
    model,
    tokenizer,
    tools_schema: list[dict],
    work_dir: Path,
    goal: str,
    max_steps: int = 30,
    max_new_tokens: int = 256,
    max_context_tokens: int = 8000,
    log_path: Path | None = None,
    verbose: bool = True,
    act_on_shadow: bool = False,
    rewind_cooldown: int = 3,
) -> dict:
    """Run one shadow-logged session.

    Returns metrics dict; writes per-turn JSONL to log_path if provided.
    """
    # Left padding is required for correct batch-2 generation on decoder-only models.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Qwen 3.5 marks end-of-turn with <|im_end|>; the default eos is
    # <|endoftext|>. Without <|im_end|> in the stop set, the model keeps
    # generating past turn boundaries and hallucinates role tokens.
    eos_ids = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end > 0 and im_end not in eos_ids:
        eos_ids.append(im_end)

    tools = ToolRunner(work_dir)
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]

    log_fh = log_path.open("w") if log_path else None
    t_start = time.time()
    stop_reason = "max_steps"
    n_tool_calls = 0
    n_shadow_yes = 0
    n_rewinds = 0
    cooldown_remaining = 0
    total_input_tokens = 0

    try:
        for step in range(max_steps):
            agent_text = tokenizer.apply_chat_template(
                messages, tools=tools_schema,
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            # No tools in shadow render: the model cannot be tempted to emit
            # a tool_call to "act" on the shadow prompt. Costs a little
            # prefix-share efficiency; buys reliable structured output.
            shadow_text = tokenizer.apply_chat_template(
                messages + [{"role": "user", "content": SHADOW_QUERY}],
                tools=None,
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )

            batch = tokenizer(
                [agent_text, shadow_text],
                return_tensors="pt", padding=True,
            ).to("cuda:0")
            input_len = int(batch.input_ids.shape[1])
            agent_prompt_tokens = int(batch.attention_mask[0].sum())
            total_input_tokens += agent_prompt_tokens

            if agent_prompt_tokens > max_context_tokens:
                stop_reason = f"context_overflow ({agent_prompt_tokens} > {max_context_tokens})"
                break

            if verbose:
                print(f"  [step {step}] ctx={agent_prompt_tokens}t msgs={len(messages)}")

            t_turn = time.time()
            with torch.no_grad():
                out = model.generate(
                    batch.input_ids,
                    attention_mask=batch.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=eos_ids,
                )
            torch.cuda.synchronize()
            turn_s = time.time() - t_turn

            agent_raw = _decode_gen(tokenizer, out[0], input_len)
            shadow_raw = _decode_gen(tokenizer, out[1], input_len)
            shadow = parse_shadow(shadow_raw)
            if shadow["should"] == "YES":
                n_shadow_yes += 1

            tool_call = parse_tool_call(agent_raw)

            turn_log: dict[str, Any] = {
                "turn": step,
                "t_turn_s": round(turn_s, 3),
                "agent_prompt_tokens": agent_prompt_tokens,
                "agent_raw_output": agent_raw,
                "tool_call": tool_call,
                "shadow": shadow,
                "rewind_applied": False,
                "rewind_n_elided": 0,
            }

            if tool_call is None:
                stop_reason = "no_tool_call"
                turn_log["tool_result"] = None
                if verbose:
                    print(f"    NO TOOL CALL. agent: {agent_raw[:100]!r}")
                    print(f"    shadow should={shadow['should']} summary={(shadow['summary'] or '')[:100]!r}")
                if log_fh:
                    log_fh.write(json.dumps(turn_log) + "\n")
                    log_fh.flush()
                break

            # Execute tool (phase 1: shadow is logged, NEVER acted on).
            n_tool_calls += 1
            call_id = f"call_{step}"
            result = tools.run(tool_call["name"], tool_call["arguments"])
            turn_log["tool_result"] = result[:2000]

            if verbose:
                arg_key = next(iter(tool_call["arguments"]), None)
                arg_val = str(tool_call["arguments"].get(arg_key, ""))[:60]
                print(f"    → {tool_call['name']}({arg_key}={arg_val!r})  shadow={shadow['should']}")

            # Phase 2: act on shadow YES. Rewind BEFORE appending this turn's
            # pair — the pre-rewind messages end with the previous turn's
            # tool_result, which is the verification evidence the shadow read.
            # apply_rewind preserves that tool_result (via the last-assist+tool
            # pair), splices the summary, and drops the earlier exploration.
            if (
                act_on_shadow
                and shadow["should"] == "YES"
                and cooldown_remaining == 0
                and shadow["summary"]
            ):
                messages, n_elided = apply_rewind(messages, shadow["summary"])
                if n_elided > 0:
                    n_rewinds += 1
                    cooldown_remaining = rewind_cooldown
                    turn_log["rewind_applied"] = True
                    turn_log["rewind_n_elided"] = n_elided
                    if verbose:
                        print(f"    ** REWIND applied: elided {n_elided} messages")
            elif cooldown_remaining > 0:
                cooldown_remaining -= 1

            if log_fh:
                log_fh.write(json.dumps(turn_log) + "\n")
                log_fh.flush()

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
    finally:
        if log_fh:
            log_fh.close()

    return {
        "stop_reason": stop_reason,
        "steps": step + 1,
        "n_tool_calls": n_tool_calls,
        "n_shadow_yes": n_shadow_yes,
        "n_rewinds": n_rewinds,
        "act_on_shadow": act_on_shadow,
        "total_input_tokens": total_input_tokens,
        "wall_time_s": time.time() - t_start,
    }
