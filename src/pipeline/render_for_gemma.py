#!/usr/bin/env python3
"""
Render annotated Sessions into per-checkpoint training chunks for
Gemma 4 SFT.

Design (discussed in session 2026-04-18):
  - Sessions are too long to fit in 8K seq_len as single examples.
  - Simple right-truncation loses most checkpoints (median fires at 40K
    tokens, well past any realistic seq_len budget).
  - Solution: one training example per checkpoint. Each chunk preserves
    the initial user prompt (the goal), then the last N tokens of
    context leading up to the checkpoint emission, then the emission
    itself (assistant tool_call + tool ack).

Token budget (defaults, overridable):
  - total:   8,192 tokens
  - goal:    up to 2,048 tokens reserved for the initial user prompt
             (right-truncated at character boundary if over)
  - recent:  remainder after subtracting goal tokens and checkpoint
             emission tokens. Messages added from newest-first at
             message boundaries until budget is full.

Transformations applied:
  1) checkpoint events — each becomes its own training example. The
     assistant's checkpoint tool_call + tool ack form the target the
     model should learn to emit given the preceding context.

Expire events are NOT rendered into training data for v1. Expire is
applied at inference-time by the harness (separation discussed in
session). Training only teaches checkpoint emission.

Output: JSONL, one {messages: [...]} per checkpoint. File is suitable
for SFT with trl's SFTTrainer + Gemma 4 chat template.

Usage:
  set -a; source .env; set +a
  .venv/bin/python -m src.pipeline.render_for_gemma \\
      --in data/generated/hygiene_v1_filtered.jsonl \\
      --out data/generated/hygiene_v1.messages.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from src.pipeline.hygiene_types import Session


CHECKPOINT_TOOL_NAME = "mcp__bookmarks__checkpoint_progress"
CHECKPOINT_ACK = "Checkpoint saved. Prior exploration may now be condensed."


def _load_env_token():
    if os.environ.get("HF_TOKEN"):
        return
    env = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
            return


def render_tool_call(tool_name: str, cmd: str, input_file: str | None, call_id: str) -> dict:
    """Tool_call block with dict-typed arguments.

    Qwen 3+ chat templates expect arguments as a dict (they iterate via
    arguments.items()); stringified JSON causes a Jinja TypeError. Gemma
    and OpenAI-compatible templates accept dict arguments too.
    """
    args: dict = {}
    if tool_name == "Bash":
        args = {"command": cmd}
    elif tool_name in ("Read", "Write", "Edit", "MultiEdit"):
        args = {"file_path": input_file or cmd}
        if tool_name in ("Edit", "Write", "MultiEdit"):
            args["content"] = "[elided]"
    elif tool_name in ("Grep", "Glob"):
        args = {"pattern": cmd}
    else:
        args = {"input": cmd}
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": tool_name, "arguments": args},
    }


def render_checkpoint_tool_call(checkpoint: dict, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": CHECKPOINT_TOOL_NAME,
            "arguments": {
                "progress_type": checkpoint["progress_type"],
                "finding": checkpoint["finding"],
                "evidence": checkpoint["evidence"],
                "next_direction": checkpoint["next_direction"],
            },
        },
    }


def session_to_messages(session: Session) -> tuple[list[dict], list[tuple[int, dict]]]:
    """Render a session into a flat chat-messages list AND a list of
    (message_index, checkpoint_dict) for each checkpoint annotation.

    The messages list does NOT include checkpoint emissions. Those are
    produced separately per-chunk so each chunk can pair the emission
    with its preceding context.
    """
    messages: list[dict] = []
    step_idx_to_msg_idx: dict[int, int] = {}  # maps step.idx → last message index produced for that step

    for step in session.steps:
        if step.role == "user_text":
            messages.append({"role": "user", "content": step.text})
        elif step.role == "assistant_text":
            messages.append({"role": "assistant", "content": step.text})
        else:  # tool
            call_id = f"call_{step.idx}"
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [render_tool_call(step.tool_name, step.cmd, step.input_file, call_id)],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": step.tool_name,
                "content": step.output,
            })
        step_idx_to_msg_idx[step.idx] = len(messages) - 1

    # Find checkpoints, pair each with the message index of its after_step
    ckpts: list[tuple[int, dict]] = []
    for ev in session.events:
        if isinstance(ev, dict) and "checkpoint" in ev:
            cp = ev["checkpoint"]
            after = int(cp["after_step"])
            msg_idx = step_idx_to_msg_idx.get(after)
            if msg_idx is None:
                continue
            ckpts.append((msg_idx, cp))

    return messages, ckpts


def measure_tokens(tokenizer, messages: list[dict], tools_schema: list[dict]) -> int:
    """Rough token count of rendering these messages through the chat template."""
    # enable_thinking=False turns off Qwen 3+ chain-of-thought preamble;
    # ignored by tokenizers that don't use that kwarg.
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tools=tools_schema, tokenize=False,
            add_generation_prompt=False, enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages, tools=tools_schema, tokenize=False,
            add_generation_prompt=False,
        )
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def goal_fits(tokenizer, goal_msg: dict, max_tokens: int) -> bool:
    """True iff the goal prompt fits within max_tokens.

    We drop-rather-than-truncate long goals: teaching the model to emit
    checkpoints based on a partial goal understanding is worse training
    signal than excluding the session entirely.
    """
    content = goal_msg.get("content") or ""
    n = len(tokenizer(content, add_special_tokens=False)["input_ids"])
    return n <= max_tokens


def _apply_template(tokenizer, msgs, tools_schema, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            msgs, tools=tools_schema, tokenize=False,
            add_generation_prompt=add_generation_prompt, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tools=tools_schema, tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def _build_chunk(
    tokenizer,
    messages: list[dict],
    context_end_idx: int,
    completion_msgs: list[dict],
    budget_total: int,
    budget_goal: int,
    min_context_msgs: int,
    tools_schema: list[dict],
) -> dict | None:
    """Left-truncate recent context and render a prompt/completion pair.

    The target the model learns to emit is `completion_msgs` (one or two
    messages). For positive chunks that's the checkpoint call + ack; for
    negatives it's whatever the session actually did next.
    """
    goal = messages[0] if messages and messages[0].get("role") == "user" else None
    if goal is None or not goal_fits(tokenizer, goal, budget_goal):
        return None

    fixed_tokens = measure_tokens(tokenizer, [goal] + completion_msgs, tools_schema)
    if budget_total - fixed_tokens < 512:
        return None

    context_window: list[dict] = []
    for i in range(context_end_idx, 0, -1):
        trial = [messages[i]] + context_window
        full = [goal] + trial + completion_msgs
        if measure_tokens(tokenizer, full, tools_schema) > budget_total:
            break
        context_window = trial

    if len(context_window) < min_context_msgs:
        return None

    final_messages = [goal] + context_window + completion_msgs
    prompt_text = _apply_template(tokenizer, [goal] + context_window, tools_schema, add_generation_prompt=True)
    full_text = _apply_template(tokenizer, final_messages, tools_schema, add_generation_prompt=False)
    if not full_text.startswith(prompt_text):
        return None

    return {
        "messages": final_messages,
        "prompt": prompt_text,
        "completion": full_text[len(prompt_text):],
        "n_tokens": measure_tokens(tokenizer, final_messages, tools_schema),
        "n_context_msgs": len(context_window),
    }


def build_chunk_for_checkpoint(
    tokenizer, messages, ckpt_msg_idx, checkpoint, chunk_idx,
    budget_total, budget_goal, min_context_msgs, tools_schema,
    post_checkpoint_steps: int = 1,
    tool_response_max_chars: int = 500,
) -> dict | None:
    """v17: optionally append post-checkpoint steps to the completion.

    Extends the completion with the next N assistant tool_call + tool_response
    pairs that follow the checkpoint in the real session. Teaches the model
    that checkpoints are mid-session pivots, not terminal actions — and adds
    more "emit normal tool call" training signal to counter catastrophic
    forgetting.
    """
    call_id = f"call_ckpt_{chunk_idx}"
    emission = [
        {
            "role": "assistant", "content": None,
            "tool_calls": [render_checkpoint_tool_call(checkpoint, call_id)],
        },
        {
            "role": "tool", "tool_call_id": call_id,
            "name": CHECKPOINT_TOOL_NAME, "content": CHECKPOINT_ACK,
        },
    ]

    # v17: append up to post_checkpoint_steps real follow-up turns after
    # the checkpoint anchor. Each turn = assistant tool_call + tool response.
    if post_checkpoint_steps > 0:
        added = 0
        i = ckpt_msg_idx + 1
        while i < len(messages) and added < post_checkpoint_steps:
            m = messages[i]
            if m.get("role") == "assistant" and m.get("tool_calls"):
                emission.append(m)
                if i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                    resp = messages[i + 1]
                    content = resp.get("content") or ""
                    if tool_response_max_chars and len(content) > tool_response_max_chars:
                        content = content[:tool_response_max_chars] + "\n[truncated]"
                    emission.append({**resp, "content": content})
                    i += 2
                else:
                    i += 1
                added += 1
            else:
                i += 1

    chunk = _build_chunk(
        tokenizer, messages, ckpt_msg_idx, emission,
        budget_total, budget_goal, min_context_msgs, tools_schema,
    )
    if chunk is not None:
        chunk["label"] = "positive"
        chunk["checkpoint_step"] = int(checkpoint["after_step"])
        chunk["progress_type"] = checkpoint["progress_type"]
        chunk["post_checkpoint_steps"] = min(post_checkpoint_steps, added if post_checkpoint_steps > 0 else 0)
    return chunk


def build_chunk_for_negative(
    tokenizer, messages, anchor_idx,
    budget_total, budget_goal, min_context_msgs, tools_schema,
    neg_tool_response_max_chars: int | None = 500,
) -> dict | None:
    """Negative anchor: messages[anchor_idx] is a non-checkpoint assistant
    turn. Target completion = that assistant's tool_call in full + the
    following tool response (content truncated to bound gradient mass).

    v15 change — we truncate the tool_response content in place, NOT the
    final completion string. Rationale: earlier (v4-v13) we capped the
    completion text at ~100 tokens. That cut the negative class off
    mid-tool_call, so the model never saw a structurally complete
    "normal" assistant turn (tool_call + response + end-of-turn). At
    inference from cold start, the model's only "complete" completion
    pattern was checkpoint_progress → it emitted checkpoints reflexively.
    Now both pos and neg completions have identical end-of-turn
    structure; only the tool name and content body differ.
    """
    if anchor_idx <= 0 or anchor_idx >= len(messages):
        return None
    target = messages[anchor_idx]
    if target.get("role") != "assistant":
        return None
    completion_msgs = [target]
    if target.get("tool_calls") and anchor_idx + 1 < len(messages):
        nxt = messages[anchor_idx + 1]
        if nxt.get("role") == "tool":
            content = nxt.get("content") or ""
            if neg_tool_response_max_chars and len(content) > neg_tool_response_max_chars:
                content = content[:neg_tool_response_max_chars] + "\n[truncated]"
            completion_msgs.append({**nxt, "content": content})
    chunk = _build_chunk(
        tokenizer, messages, anchor_idx - 1, completion_msgs,
        budget_total, budget_goal, min_context_msgs, tools_schema,
    )
    if chunk is None:
        return None
    chunk["label"] = "negative"
    chunk["negative_anchor_idx"] = anchor_idx
    return chunk


def build_chunk_for_cold_start(
    tokenizer, messages, tools_schema,
    neg_tool_response_max_chars: int = 500,
) -> dict | None:
    """Cold-start chunk: context = goal only, completion = first assistant
    tool_call turn + its tool response.

    Purpose (v16): fix the agent-loop collapse — v15 had strong in-context
    discrimination (14% FP) but reflexively emitted checkpoints at step 0
    because every training example had ≥10 messages of context. The model
    never saw "goal only → emit exploration tool" as a completion pattern.
    This chunk class teaches exactly that.
    """
    if not messages or messages[0].get("role") != "user":
        return None
    goal = messages[0]

    first_ast_idx = None
    for i in range(1, len(messages)):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            first_ast_idx = i
            break
    if first_ast_idx is None:
        return None

    target = messages[first_ast_idx]
    completion_msgs = [target]
    if first_ast_idx + 1 < len(messages):
        nxt = messages[first_ast_idx + 1]
        if nxt.get("role") == "tool":
            content = nxt.get("content") or ""
            if neg_tool_response_max_chars and len(content) > neg_tool_response_max_chars:
                content = content[:neg_tool_response_max_chars] + "\n[truncated]"
            completion_msgs.append({**nxt, "content": content})

    final_messages = [goal] + completion_msgs
    prompt_text = _apply_template(tokenizer, [goal], tools_schema, add_generation_prompt=True)
    full_text = _apply_template(tokenizer, final_messages, tools_schema, add_generation_prompt=False)
    if not full_text.startswith(prompt_text):
        return None

    return {
        "messages": final_messages,
        "prompt": prompt_text,
        "completion": full_text[len(prompt_text):],
        "n_tokens": measure_tokens(tokenizer, final_messages, tools_schema),
        "n_context_msgs": 0,
        "label": "cold_start",
    }


def select_negative_anchors(
    messages: list[dict], ckpt_msg_idxs: set[int], n: int, rng: random.Random,
) -> list[int]:
    """Candidate anchor indexes for negatives: assistant TOOL-CALL
    messages whose preceding message is a tool_response or user turn,
    where the preceding context did NOT just produce a checkpoint-worthy
    state.

    v6: restrict to assistant turns with tool_calls (not plain text).
    Reason: if negatives include assistant_text continuations, the model
    learns "end turn with a short prose summary" instead of the
    discrimination we actually want. Keeping only tool-call negatives
    makes every completion start with `<tool_call>\\n{"name": "…"}` so
    the learned decision is purely which tool name to emit.
    """
    candidates: list[int] = []
    for k in range(1, len(messages)):
        cur = messages[k]
        if cur.get("role") != "assistant":
            continue
        if not cur.get("tool_calls"):
            continue
        if messages[k - 1].get("role") not in ("tool", "user"):
            continue
        if (k - 1) in ckpt_msg_idxs:
            continue
        candidates.append(k)
    rng.shuffle(candidates)
    return candidates[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", dest="outp", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B",
                    help="HF model id for tokenizer + chat template")
    ap.add_argument("--budget-total", type=int, default=8192)
    ap.add_argument("--budget-goal", type=int, default=2048)
    ap.add_argument("--min-context-msgs", type=int, default=10,
                    help="Drop chunks whose recent-context window has fewer than N messages. Prevents training on checkpoint examples with too little grounding evidence. Default 10 (roughly 5 tool invocations).")
    ap.add_argument("--neg-ratio", type=float, default=1.0,
                    help="Negatives per positive within a session. 0 disables. Default 1.0.")
    ap.add_argument("--neg-tool-response-max-chars", type=int, default=500,
                    help="Truncate the tool_response CONTENT inside a negative's completion (not the whole completion text). v15 fix: prior versions chopped the completion string mid-tool_call, so the model never saw a complete 'normal' assistant turn and collapsed to always-emit-checkpoint at inference cold start. Truncating content only keeps both pos/neg completions structurally complete. 0 disables. Default 500.")
    ap.add_argument("--cold-start-per-session", type=int, default=1,
                    help="Emit N cold-start chunks per session (v16): context=goal only, completion=first assistant tool_call. Teaches the model what to emit with no prior trajectory (fixes agent-loop collapse). 0 disables.")
    ap.add_argument("--post-checkpoint-steps", type=int, default=1,
                    help="v17: extend positive chunks' completion by N post-checkpoint turns. Teaches the model that checkpoints are mid-session pivots, not terminal actions. 0 disables (restores v15 behavior). Default 1.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for sampling negative anchors.")
    args = ap.parse_args()

    _load_env_token()

    from transformers import AutoTokenizer
    from src.pipeline.verify_gemma_tokenization import TOOLS_SCHEMA

    print(f"loading tokenizer: {args.model}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model)

    args.outp.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    total_ckpts = 0
    produced_pos = 0
    produced_neg = 0
    produced_cold = 0
    skipped_pos = 0
    skipped_neg = 0
    token_counts: list[int] = []
    with args.inp.open() as fin, args.outp.open("w") as fout:
        for i, line in enumerate(fin):
            row = json.loads(line)
            session = Session.from_dict(row)
            messages, ckpts = session_to_messages(session)
            total_ckpts += len(ckpts)

            session_positives = 0
            for chunk_idx, (msg_idx, cp) in enumerate(ckpts):
                chunk = build_chunk_for_checkpoint(
                    tok, messages, msg_idx, cp, chunk_idx,
                    args.budget_total, args.budget_goal,
                    args.min_context_msgs, TOOLS_SCHEMA,
                    post_checkpoint_steps=args.post_checkpoint_steps,
                    tool_response_max_chars=args.neg_tool_response_max_chars,
                )
                if chunk is None:
                    skipped_pos += 1
                    continue
                fout.write(json.dumps({
                    "prompt": chunk["prompt"],
                    "completion": chunk["completion"],
                    "messages": chunk["messages"],
                    "meta": {
                        "label": "positive",
                        "source_session": session.session_id,
                        "checkpoint_step": chunk["checkpoint_step"],
                        "progress_type": chunk["progress_type"],
                        "n_tokens": chunk["n_tokens"],
                        "n_context_msgs": chunk["n_context_msgs"],
                    },
                }) + "\n")
                token_counts.append(chunk["n_tokens"])
                produced_pos += 1
                session_positives += 1

            # Negatives: up to neg-ratio × session_positives per session.
            target_negs = int(round(session_positives * args.neg_ratio))
            if target_negs > 0:
                ckpt_anchor_idxs = {msg_idx for (msg_idx, _) in ckpts}
                # Oversample candidates since some will fail the min_context filter.
                candidates = select_negative_anchors(
                    messages, ckpt_anchor_idxs, n=target_negs * 3, rng=rng,
                )
                built = 0
                for anchor in candidates:
                    if built >= target_negs:
                        break
                    chunk = build_chunk_for_negative(
                        tok, messages, anchor,
                        args.budget_total, args.budget_goal,
                        args.min_context_msgs, TOOLS_SCHEMA,
                        neg_tool_response_max_chars=args.neg_tool_response_max_chars,
                    )
                    if chunk is None:
                        skipped_neg += 1
                        continue
                    fout.write(json.dumps({
                        "prompt": chunk["prompt"],
                        "completion": chunk["completion"],
                        "messages": chunk["messages"],
                        "meta": {
                            "label": "negative",
                            "source_session": session.session_id,
                            "negative_anchor_idx": chunk["negative_anchor_idx"],
                            "n_tokens": chunk["n_tokens"],
                            "n_context_msgs": chunk["n_context_msgs"],
                        },
                    }) + "\n")
                    token_counts.append(chunk["n_tokens"])
                    produced_neg += 1
                    built += 1

            # Cold-start chunks (v16): one per session by default.
            for _ in range(max(0, args.cold_start_per_session)):
                chunk = build_chunk_for_cold_start(
                    tok, messages, TOOLS_SCHEMA,
                    neg_tool_response_max_chars=args.neg_tool_response_max_chars,
                )
                if chunk is None:
                    break
                fout.write(json.dumps({
                    "prompt": chunk["prompt"],
                    "completion": chunk["completion"],
                    "messages": chunk["messages"],
                    "meta": {
                        "label": "cold_start",
                        "source_session": session.session_id,
                        "n_tokens": chunk["n_tokens"],
                        "n_context_msgs": 0,
                    },
                }) + "\n")
                token_counts.append(chunk["n_tokens"])
                produced_cold += 1

            if (i + 1) % 10 == 0:
                print(
                    f"  {i+1} sessions  pos={produced_pos} neg={produced_neg} "
                    f"cold={produced_cold} "
                    f"(skipped pos={skipped_pos} neg={skipped_neg})",
                    file=sys.stderr,
                )

    if token_counts:
        token_counts.sort()
        def p(q): return token_counts[int(len(token_counts) * q)]
        print(
            f"\ndone: sessions in={i+1}  checkpoints={total_ckpts}  "
            f"positives={produced_pos}  negatives={produced_neg}  cold_starts={produced_cold}  "
            f"skipped(pos/neg)={skipped_pos}/{skipped_neg}",
            file=sys.stderr,
        )
        print(
            f"chunk token count  min={min(token_counts):,}  p25={p(0.25):,}  "
            f"p50={p(0.50):,}  p75={p(0.75):,}  p95={p(0.95):,}  max={max(token_counts):,}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
