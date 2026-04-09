# New Datasets Found (2026-04-08)

## Priority 1 — Native Claude Code format (solves distribution gap)

- **nlile/misc-merged-claude-code-traces-v1** — 32K Claude Code traces, merged from 10 sources. Native Claude Code tool format (Bash, Read, Edit, Grep). HuggingFace.
- **DataClaw datasets** (peteromallet/dataclaw-*) — Real Claude Code session logs from multiple users. Native JSONL format. Various sizes.
- **sammshen/wildclaw-opus-traces** — Claude Opus 4.6 traces with full tool calls/results.
- **zai-org/CC-Bench-trajectories** — 74 tasks using Claude Code as harness. Small but high-quality.

## Priority 2 — Claude model, different scaffold

- **SWE-bench/SWE-smith-trajectories** — 76K rows, Claude 3.7 Sonnet via SWE-agent. MIT.
- **ByteDance-Seed/Multi-SWE-bench_trajs** — Claude 3.5/3.7 Sonnet, 6 languages (Python, Go, C, C++, Rust, Java).
- **AmanPriyanshu cleaned MEnvData** — Better-formatted version of MEnvData we already use. Apache-2.0.

## Priority 3 — Non-Claude but useful

- **Kwai-Klear SWE-smith 66K** — MIT, tool calls present.
- **OpenThoughts-Agent-v1-SFT** — 15K, C#/Java/bash, Apache-2.0.
- **AlienKevin SWE-smith-rs** — 5.3K + 3.9K Rust trajectories, MIT.
- **davongluck quality subsets** — Quality filter for nebius 67K.

## Impact on CNN training

The nlile/misc-merged dataset (32K native Claude Code traces) would eliminate the feature inversion problem.
Current CNN has inverted since_cmd/since_file signals because SWE-bench uses different tool format.
Native Claude Code data would let the CNN learn the correct patterns directly.
