#!/usr/bin/env python3
"""
Offline eval for the hygiene LoRA adapter.

Positives (val chunks): strip last 2 msgs (assistant checkpoint call +
tool ack), generate with base and base+adapter. Measure whether the
next-turn output contains a well-formed checkpoint_progress call and
how close its arguments are to what Sonnet produced.

Negatives: strip further back to a pre-checkpoint state. Measure how
often the adapter *falsely* emits a checkpoint_progress call at a
position where Sonnet decided none was warranted.

Emits a per-example JSONL plus a printed summary table.

Usage (inside rocm/pytorch via eval_adapter_docker.sh):
  python -u benchmarks/eval_adapter.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_json_object_containing(text: str, marker: str) -> dict | None:
    """Extract the smallest balanced JSON object containing `marker`."""
    idx = text.find(marker)
    if idx < 0:
        return None
    start = text.rfind("{", 0, idx)
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


_XML_FUNCTION_RE = re.compile(
    r"<function=mcp__bookmarks__checkpoint_progress>(.*?)</function>",
    re.DOTALL,
)
_XML_PARAMETER_RE = re.compile(
    r"<parameter=(\w+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def _parse_xml_checkpoint(text: str) -> dict | None:
    """Qwen 3.5 format: <function=X><parameter=k>v</parameter>...</function>."""
    m = _XML_FUNCTION_RE.search(text)
    if m is None:
        return None
    body = m.group(1)
    args = {k: v for k, v in _XML_PARAMETER_RE.findall(body)}
    return args if args else None


def parse_checkpoint(text: str) -> dict | None:
    """Return the arguments dict if the output contains a checkpoint_progress call.

    Handles both Qwen 2.5 Coder's JSON form ({"name": X, "arguments": {...}})
    and Qwen 3.5's XML form (<function=X><parameter=k>v</parameter>...</function>).
    """
    # Qwen 3.5 XML form first — the marker is more specific.
    xml = _parse_xml_checkpoint(text)
    if xml is not None:
        return xml
    # Qwen 2.5 Coder JSON form
    obj = find_json_object_containing(text, "mcp__bookmarks__checkpoint_progress")
    if obj is None:
        return None
    args = obj.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return None
    return args if isinstance(args, dict) else None


_WORD_RE = re.compile(r"\w+")


def token_jaccard(a: str, b: str) -> float:
    ta = {w for w in _WORD_RE.findall(a.lower()) if len(w) > 2}
    tb = {w for w in _WORD_RE.findall(b.lower()) if len(w) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/generated/hygiene_v1.chunks.jsonl"))
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--adapter", type=Path, default=Path("proxy/experiments/qwen25_coder_3b_hygiene_v8/final"))
    ap.add_argument("--seed", type=int, default=42, help="must match training seed for val split")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--neg-trim", type=int, default=6,
                    help="extra msgs to trim from end to build a pre-checkpoint prompt")
    ap.add_argument("--out", type=Path,
                    default=Path("proxy/experiments/qwen25_coder_1.5b_hygiene_v1/eval.jsonl"))
    args = ap.parse_args()

    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.pipeline.verify_gemma_tokenization import TOOLS_SCHEMA

    # Reproduce training-time val split exactly (same shuffle seed + fraction).
    rows = []
    with args.data.open() as f:
        for line in f:
            rows.append(json.loads(line))
    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_fraction))
    val_rows = rows[:n_val]
    log(f"val chunks: {len(val_rows)}")

    # v3: chunks carry an explicit label in meta. v2: infer by looking for
    # the checkpoint tool_call at msgs[-2]; construct synthetic negatives
    # by trimming further back from the same chunk.
    positives = []
    negatives = []
    has_labels = any(r.get("meta", {}).get("label") for r in val_rows)
    log(f"data has v3 labels: {has_labels}")

    for i, r in enumerate(val_rows):
        msgs = r["messages"]
        label = r.get("meta", {}).get("label")

        if label == "positive" or (label is None
                                   and len(msgs) >= 3
                                   and msgs[-2].get("role") == "assistant"
                                   and msgs[-2].get("tool_calls")):
            tc = msgs[-2].get("tool_calls") or []
            gt_args = tc[0].get("function", {}).get("arguments") if tc else None
            if isinstance(gt_args, str):
                try:
                    gt_args = json.loads(gt_args)
                except Exception:
                    gt_args = {}
            if isinstance(gt_args, dict):
                positives.append({
                    "kind": "positive",
                    "chunk_idx": i,
                    "prompt_msgs": msgs[:-2],
                    "ground_truth_args": gt_args,
                })
        elif label == "negative":
            # v3 negative: completion is the last 1-2 msgs (assistant +
            # optional tool_response). Strip them to build the prompt.
            k = 1 if (len(msgs) >= 2 and msgs[-1].get("role") == "assistant") else 2
            prompt_msgs = msgs[:-k] if k <= len(msgs) - 2 else msgs[:-1]
            negatives.append({
                "kind": "negative",
                "chunk_idx": i,
                "prompt_msgs": prompt_msgs,
            })

    if not has_labels:
        # v2 fallback: synthesize negatives by trimming further back from
        # positive chunks.
        for i, r in enumerate(val_rows):
            msgs = r["messages"]
            cut = len(msgs) - 2 - args.neg_trim
            while cut > 4 and msgs[cut - 1].get("role") not in ("tool", "user"):
                cut -= 1
            if cut < 4:
                continue
            negatives.append({
                "kind": "negative",
                "chunk_idx": i,
                "prompt_msgs": msgs[:cut],
            })
    log(f"positives: {len(positives)}  negatives: {len(negatives)}")

    log(f"loading tokenizer: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model)

    log("loading base model")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    log(f"attaching adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    def gen(prompt_msgs):
        text = tok.apply_chat_template(
            prompt_msgs, tools=TOOLS_SCHEMA,
            tokenize=False, add_generation_prompt=True,
        )
        enc = tok(text, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(
                enc.input_ids,
                max_new_tokens=args.max_new, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0, int(enc.input_ids.shape[1]):], skip_special_tokens=False)

    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    examples = positives + negatives

    log(f"=== BASE pass (adapter disabled), {len(examples)} examples ===")
    t0 = time.time()
    for j, ex in enumerate(examples):
        with model.disable_adapter():
            text = gen(ex["prompt_msgs"])
        results.append({
            "variant": "base",
            "kind": ex["kind"],
            "chunk_idx": ex["chunk_idx"],
            "ground_truth_args": ex.get("ground_truth_args"),
            "output": text,
            "parsed": parse_checkpoint(text),
        })
        if (j + 1) % 10 == 0:
            log(f"  base {j+1}/{len(examples)}  {time.time()-t0:.0f}s")

    log(f"=== ADAPTER pass, {len(examples)} examples ===")
    t0 = time.time()
    for j, ex in enumerate(examples):
        text = gen(ex["prompt_msgs"])
        results.append({
            "variant": "adapter",
            "kind": ex["kind"],
            "chunk_idx": ex["chunk_idx"],
            "ground_truth_args": ex.get("ground_truth_args"),
            "output": text,
            "parsed": parse_checkpoint(text),
        })
        if (j + 1) % 10 == 0:
            log(f"  adapter {j+1}/{len(examples)}  {time.time()-t0:.0f}s")

    with args.out.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"wrote {len(results)} rows to {args.out}")

    def summarize(variant: str, kind: str) -> dict:
        rs = [r for r in results if r["variant"] == variant and r["kind"] == kind]
        n = len(rs)
        emit = [r for r in rs if r["parsed"] is not None]
        out = {"n": n, "emit": f"{len(emit)}/{n}", "emit_rate": len(emit) / n if n else 0.0}
        if kind == "positive":
            ptype = sum(1 for r in emit
                        if r["parsed"].get("progress_type")
                        == (r["ground_truth_args"] or {}).get("progress_type"))
            out["ptype_acc"] = ptype / n if n else 0.0
            for field in ("finding", "evidence", "next_direction"):
                js = []
                for r in emit:
                    gt = str((r["ground_truth_args"] or {}).get(field, ""))
                    pr = str(r["parsed"].get(field, ""))
                    js.append(token_jaccard(pr, gt))
                out[f"{field[:4]}_jac"] = (sum(js) / n) if n else 0.0
        return out

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for variant in ("base", "adapter"):
        for kind in ("positive", "negative"):
            s = summarize(variant, kind)
            cells = []
            for k, v in s.items():
                cells.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
            print(f"{variant:>8s} / {kind:>8s}  " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
