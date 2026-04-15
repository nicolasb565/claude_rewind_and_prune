#!/usr/bin/env python3
"""
Write per-step Python-side content features for every OOD transcript into
data/generated/content_features_ood_python.json so a JS harness can
cross-validate its own extractor against them.

Output JSON:
  {
    "sessions": [
      {"session_id": "bench_03_llvm_loop_vec",
       "steps": [
         {"step": 0,
          "parsed": {"tool": "...", "cmd": "...", "file": "...", "output": "..."},
          "features": {"match_ratio_5": ..., ..., "err_volume_ratio_vs_5": ...}
         },
         ...
       ]
      }, ...
    ]
  }

The `parsed` dict is the same shape the JS ContentFeatureExtractor will
consume (so the JS test can just read the same JSON and pass `parsed`
through `addStep`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.pipeline.parsers.nlile import parse_session  # noqa: E402
from src.pipeline.extract_features import compute_step_features  # noqa: E402
from benchmarks.v9_tier1_train import compute_tier1_features  # noqa: E402
from benchmarks.v9_content_features import compute_content_features  # noqa: E402

FEATS = [
    "match_ratio_5", "self_sim_max", "repeat_no_error",
    "cur_bash_and_match_ratio",
    "unique_err_sigs_6", "new_token_ratio_vs_5",
    "has_success_marker", "err_volume_ratio_vs_5",
]

OOD_DIR = REPO / "benchmarks" / "results" / "comparison_off"
OUT_PATH = REPO / "data" / "generated" / "content_features_ood_python.json"


def main() -> int:
    sessions_out = []
    for td in sorted(OOD_DIR.iterdir()):
        if not td.is_dir():
            continue
        t = td / "transcript_1.jsonl"
        if not t.exists():
            continue
        messages = []
        for line in t.read_text().splitlines():
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
        steps = parse_session(messages)
        v9 = compute_step_features(steps)
        compute_tier1_features(v9)
        content = compute_content_features(steps)

        steps_out = []
        for i, (s, f, c) in enumerate(zip(steps, v9, content)):
            row = {"step": i, "parsed": {
                "tool": s.get("tool", "other"),
                "cmd": s.get("cmd") or "",
                "file": s.get("file"),
                "output": s.get("output") or "",
                "tool_name": s.get("tool_name"),
            }, "features": {}}
            # Tier-1 syntactic
            for k in ("match_ratio_5", "self_sim_max", "repeat_no_error",
                      "cur_bash_and_match_ratio"):
                row["features"][k] = float(f[k])
            # Tier-3 content
            for k in ("unique_err_sigs_6", "new_token_ratio_vs_5",
                      "has_success_marker", "err_volume_ratio_vs_5"):
                row["features"][k] = float(c[k])
            steps_out.append(row)

        sessions_out.append({
            "session_id": f"bench_{td.name}",
            "steps": steps_out,
        })
        print(f"  {td.name}: {len(steps_out)} steps", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"sessions": sessions_out}))
    total = sum(len(s["steps"]) for s in sessions_out)
    print(f"wrote {OUT_PATH}  ({len(sessions_out)} sessions, {total} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
