#!/usr/bin/env python3
"""
Verify a rendered training example tokenizes cleanly through Gemma 4's
chat template. Pre-training sanity: catch template mismatches, tool-schema
confusion, and token-count surprises before they waste training cycles.

Runs the tokenizer (CPU) — no GPU, no container needed. Uses transformers
already in the project venv.

Usage:
  set -a; source .env; set +a  # HF_TOKEN for gated tokenizer config
  .venv/bin/python -m src.pipeline.verify_gemma_tokenization \\
      --messages data/generated/hygiene_v1.messages.jsonl \\
      --line 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Tool schemas we pass to apply_chat_template. Needs to cover everything
# the training data references so the tokenizer knows about each function.
# Schema shape matches OpenAI functions schema, which Gemma's template
# understands.
TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "Bash",
        "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "Read",
        "description": "Read a file.",
        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]},
    }},
    {"type": "function", "function": {
        "name": "Write",
        "description": "Write contents to a file.",
        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "Edit",
        "description": "Edit a file.",
        "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "Grep",
        "description": "Grep the codebase for a pattern.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "Glob",
        "description": "Find files by glob pattern.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "mcp__bookmarks__checkpoint_progress",
        "description": (
            "Record a checkpoint when you have concrete evidence of progress "
            "(milestone_achieved or approach_eliminated). Do not checkpoint "
            "when still exploring."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "progress_type": {
                    "type": "string",
                    "enum": ["milestone_achieved", "approach_eliminated"],
                },
                "finding": {"type": "string"},
                "evidence": {"type": "string"},
                "next_direction": {"type": "string"},
            },
            "required": ["progress_type", "finding", "evidence", "next_direction"],
        },
    }},
]


def _load_env_token():
    """Source ANTHROPIC-style HF_TOKEN from .env if not set."""
    if os.environ.get("HF_TOKEN"):
        return
    env = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", type=Path,
                    default=Path(__file__).resolve().parent.parent.parent / "data" / "generated" / "hygiene_v1.messages.jsonl")
    ap.add_argument("--line", type=int, default=0, help="0-based example index")
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--print-head", type=int, default=800, help="chars of rendered text to print")
    args = ap.parse_args()

    _load_env_token()

    from transformers import AutoTokenizer

    # Unknown tool names (anything not in our schema) need to be mapped
    # or passed through. We'll filter messages to only reference known
    # tools and raise a clear warning otherwise.
    known_tools = {t["function"]["name"] for t in TOOLS_SCHEMA}

    with args.messages.open() as f:
        for i, line in enumerate(f):
            if i == args.line:
                example = json.loads(line)
                break
        else:
            raise SystemExit(f"line {args.line} not found")

    messages = example["messages"]
    unknown = set()
    for m in messages:
        for tc in m.get("tool_calls", []) or []:
            name = tc["function"]["name"]
            if name not in known_tools:
                unknown.add(name)
        if m.get("name") and m.get("name") not in known_tools:
            unknown.add(m["name"])

    if unknown:
        print(f"WARN: messages reference tools not in schema: {sorted(unknown)}")
        print("  These will still tokenize but the template won't have their definition.")
        print()

    print(f"loading tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)

    print(f"applying chat template (with {len(TOOLS_SCHEMA)} tools, {len(messages)} messages)...")
    try:
        rendered = tok.apply_chat_template(
            messages,
            tools=TOOLS_SCHEMA,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        print(f"\nERROR during apply_chat_template: {type(e).__name__}: {e}")
        raise

    print(f"  rendered length: {len(rendered):,} chars")

    # Tokenize to get token count
    ids = tok(rendered, return_tensors=None)["input_ids"]
    print(f"  token count:     {len(ids):,}")

    # Validate round-trip: decode should approximately match input
    decoded = tok.decode(ids)
    if len(decoded) < len(rendered) * 0.9:
        print(f"  WARN: decoded length {len(decoded):,} is much shorter than rendered — possible tokenization loss")

    print()
    print(f"=== first {args.print_head} chars of rendered output ===")
    print(rendered[:args.print_head])
    print("...")
    print()
    print(f"=== last {args.print_head} chars ===")
    print(rendered[-args.print_head:])


if __name__ == "__main__":
    main()
