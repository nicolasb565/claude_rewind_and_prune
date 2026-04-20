# Context Hygiene for Claude Code

Research into reducing token waste and extending effective session length by mutating the `messages[]` array in outgoing Anthropic API requests. A local HTTP proxy rewrites history on the fly — truncating stale tool outputs, dropping dead-end exploration, and (eventually) exposing primitives that let the agent manage its own memory. **No patches to Claude Code required.**

## Motivation

The Agent SDK, MCP servers, and skills all append to context; none can mutate past messages. Claude Code sends the full history on every turn, so any intervention that shrinks it recovers tokens on every subsequent request. The proxy is the only place to do this cleanly.

Two things we want to measure:

1. **Cheaper at same quality** — short/medium tasks complete as well, using fewer tokens per turn.
2. **Now possible at all** — long tasks that previously hit context limits now succeed.

Neither is systematically reported in existing literature.

## Prior direction (closed 2026-04-16)

This repo started as a stuck-loop detector — a per-step classifier trained on real Claude Code sessions, nudging the agent when it went in circles. The classifier works (LR + 2-tier filter, p≈0.001 selectivity), but the intervention's effect size on our benchmark was null (d=0.02, n=11). Root cause traced to training-data distribution mismatch. That code has been removed; see `git log` for history. The proxy, message utilities, and A/B harness are kept as foundation for the hygiene work.

## Shadow-summary + rewind (2026-04-19)

A prompt-only variant of the hygiene idea that requires **no training and no classifier**. At each agent turn, a batch-2 inference asks the same model a structured question (`SHOULD_CHECKPOINT / REASON / SUMMARY`). When the shadow answers YES and a cooldown is clear, the proxy elides prior turns and splices the summary as a synthetic user message — preserving the system prompt, the original goal, and the most recent tool_result as safety valves.

Validated locally on Qwen 3.5 4B (bf16) + RX 9070 XT via `harness/shadow_agent.py` and `harness/run_shadow_docker.sh`. Zero fine-tuning; the same base model serves as both agent and shadow.

### Headline results

A/B on four controlled Python + JS fixtures (17 runs total, `harness/fixtures/bug_0{1..7}_*`):

| Fixture | Bugs / shape | Rewinds/run | Steps Δ | Tokens Δ | Max ctx Δ |
|---|---|---|---|---|---|
| bug_02_compound | 2 Python, iterative | 1.5 | -4% | **-23%** | -26% |
| bug_04_trio | 3 Python, iterative | 3.0 | -12% | **-44%** | -50% |
| bug_06_node_trio | 3 JS, batched | 0.0 | 0% | 0% | 0% |
| bug_07_pentagon | 5 Python, iterative | 3.0 | +5% | **-30%** | -31% |

On `08_express_async` (real Express 5.2.0 bug), phase-2 **extended the exploration envelope 2× deeper** before overflow (15 vs 9 turns, 9 vs 4 Reads, 3 vs 1 Greps) — neither condition solved on 4B, but phase-2 reached targeted search stages (`Grep defineGetter.*query`) that baseline didn't. On `04_sqlite_cte` the stuck-detection branch of the prompt correctly caught a 46× identical-Grep loop and fired rewind — but 4B's capability ceiling on SQLite's 200K-line C code means it couldn't translate the rescue into success.

### What the data says

- **Phase-2 is a multiplier on rewind opportunities that fire.** On iterative tasks the model creates verified sub-milestones and each rewind compounds. On batched strategies (single big verification at end) the mechanism is inert — harmless but not useful.
- **Rewind trades turns for cheaper turns.** Sometimes +1 step but -30% tokens: the agent needs a brief "re-orient turn" after compression but runs every subsequent turn on much lighter context.
- **The stuck-detection clause extends the intervention to exploration-heavy single-bug tasks** (Express, SQLite) — but has to be calibrated carefully: too loose and it over-fires on legitimate fix-verify iteration (bug_07 regressed from 23 to 37 steps); too tight and it misses clear-loop cases. Current prompt aims for the middle ground.
- **Safety valves hold**: across 17 runs on controlled fixtures, zero catastrophic failures. Preserving system + goal + most-recent tool_result means the agent can self-correct when a summary is imperfect.

### Usage

```bash
# Single session on a fixture (see harness/fixtures/)
FIXTURE=bug_04_trio ACT_ON_SHADOW=1 bash harness/run_shadow_docker.sh

# A/B by flipping ACT_ON_SHADOW
FIXTURE=bug_04_trio ACT_ON_SHADOW=0 bash harness/run_shadow_docker.sh

# Real existing benchmark (requires setup.sh first)
FIXTURE=08_express_async FIXTURES_DIR=benchmarks/fixtures ACT_ON_SHADOW=1 \
    bash harness/run_shadow_docker.sh
```

Per-turn JSONL logs land in `harness/results/shadow/`.

### Why this matters

The previous stuck-detector project required labeling 250+ sessions, training, and found that intervention-from-features can't be made useful because the classifier can't close the loop. Shadow-summary is the *same problem flipped inside out*: the model judges itself (richer than any feature vector), and detection + intervention are one mechanism. The architecture drops straight into a frontier deployment — swap the local 4B for Haiku (shadow) + Sonnet (agent) through the existing proxy, using prompt caching so the shadow call costs ~10% of a full call.

## Repository layout

```
proxy/
  proxy.mjs        HTTP server, request mutation hook
  compact.mjs      Bash tool_result truncation (primitive #1, seed)
  messages.mjs     Message-array utilities (session key, tool extraction)
  upstream.mjs     Upstream fetch with retry/backoff
  log.mjs          Structured event log
  test/            Node test runner tests + smoke_http.mjs

benchmarks/
  run.sh           Single benchmark run (Docker-isolated per task)
  run_ab.sh        Paired OFF/ON A/B runner
  analyze_ab.py    Paired t-test, pass rates, duration stats
  compare.py       Per-task duration diff between two run dirs
  manifest.json    Tasks in scope + expected outcomes
  tasks/           Per-task prompt + grading script
  fixtures/        Cloned + prebuilt upstream repos (not committed, run setup.sh)
  Dockerfile, entrypoint.sh, setup.sh

harness/           Local agent harness for the shadow-summary + rewind work
  shadow_agent.py  Agent loop + SHADOW_QUERY prompt + apply_rewind
  run_shadow.py    Runner with --act-on-shadow flag
  run_shadow_docker.sh  rocm/pytorch container with Node + build-essential
  tools.py, parse.py    Tool runner + Qwen XML tool-call parser
  fixtures/        Controlled Python + JS bug fixtures (bug_0{1..7}_*)
  results/shadow/  Per-turn JSONL logs + .meta.json

src/pipeline/parsers/
  nlile.py         Parse Anthropic API format transcripts (replay sims)

requirements.txt   Python deps for analysis & simulation
```

## Primitives (priority order)

1. **Bash output truncation** — rewrite stale `tool_result` blocks to first N + last N lines. `proxy/compact.mjs` is the seed. Highest ROI: 30–50% token reduction on build-heavy sessions, low correctness risk since the agent already extracted what it needed before the turn moved on.
2. **Bookmarks / landmarks** — `mark(name, summary)` + `recall(name)` / `rewind_to(name)` tools via MCP. Proxy maintains the index and rewrites history on every subsequent request.
3. **Active forgetting** — `mark_stale(range, reason)` to elide dead-end exploration. Audit trail preserved for debugging.
4. **Structured reflection at subtask boundaries** — forced `{facts_learned, decisions, next_steps}` format; raw exploration compresses once the structured summary exists.

## Usage

```bash
node proxy/proxy.mjs &

# Run vanilla Claude Code through the proxy
ANTHROPIC_BASE_URL=http://localhost:8080 claude "your prompt"

# A/B is just toggling the env var
```

### Configuration

| Variable | Default | Description |
|---|---|---|
| `PROXY_PORT` | `8080` | Listen port |
| `PROXY_UPSTREAM` | `https://api.anthropic.com` | Upstream API |
| `COMPACT_ENABLED` | `0` | Enable Bash output compaction |

Compaction thresholds (`COMPACT_STALE_TURNS`, `COMPACT_KEEP_FIRST`, `COMPACT_KEEP_LAST`, `COMPACT_MIN_LINES`) are documented in `proxy/compact.mjs`.

## Prior art

- **A-MEM** (Feb 2025, arXiv:2502.12110) — agentic memory via Zettelkasten
- **Active Context Compression** (Jan 2026, arXiv:2601.07190)
- **SimpleMem** (2026) — semantic lossless compression
- **Memory for Autonomous LLM Agents** (survey, 2026, arXiv:2603.07670) — identifies the cost-vs-capability gap
- Shipped: Anthropic Memory tool + Context editing + Compaction + Auto Dream; OpenAI native compaction in GPT-5.4

## Tests

```bash
cd proxy && node --test test/*.test.mjs
```

## License

MIT — see `LICENSE`.
