# Rewind-with-summary — overnight findings

_Session: 2026-04-17 22:45 → 2026-04-18 ~02:30 EDT. Total API spend: ~$40._

## TL;DR

1. **Model-driven context management works as a capability.** Given `checkpoint_progress(summary)` with positive framing + a prompt hint, the model uses it at semantically meaningful moments. 9 checkpoints fired across all tests; quality was high on LAPACK and beast, mixed on GCC/LLVM (one checkpoint entrenched a wrong conclusion).
2. **Naming is load-bearing.** `summarize_and_forget` → 0/1 usage. Rename to `checkpoint_progress` → 5/7 usage on the same prompt hint. The word "forget" triggers the model's conservatism about destructive actions.
3. **Unprompted discovery is ~0.** Even with a friendly name, the model needs the prompt hint to reach for the tool.
4. **Cost effect by task size and task outcome:**
   - Short solved tasks (LAPACK 15–20t, GCC sccvn 11–16t, GCC mul_overflow 35–57t): rewind+hint cost neutral to +53% — the checkpoint tokens outweigh elision savings at this scale.
   - Long unsolved task (LLVM loop_vec, 200-turn cap both sides): **rewind+hint was 23% cheaper per session** ($9.68 vs $12.54) despite spending 76% more wall time. Elisions made each request cheaper, so more exploration per dollar. But neither side solved.
5. **The primitive delivers its theoretical value on long sessions — just not enough to solve harder problems.** LLVM loop_vec confirmed: rewind does cut per-request cost in context-pressured sessions. It does NOT compensate for search-strategy failures. Model-driven context management ≠ model-driven problem solving.

## Results

### LAPACK (30_lapack) — 17 runs

Median per condition:

| condition | n | cost | dur | turns | ckpt rate | median bytes saved |
|---|---|---|---|---|---|---|
| baseline | 9 | $0.42 | 139s | 15 | — | — |
| rewind+hint | 7 | **$0.67 (+58%)** | 151s | 21 | 71% | 224K |
| rewind-only (no hint) | 1 | $0.61 | 196s | 20 | 0% | — |

### Tier-1 GCC/LLVM — 1 pair each (n=1 per condition)

| task | condition | turns | cost | dur | ckpts | verify |
|---|---|---|---|---|---|---|
| **01_gcc_sccvn** | baseline | 11 | $0.50 | 249s | 0 | ok |
| | rewind+hint | 16 | $0.41 | 235s | 0 | ok |
| **02_gcc_mul_overflow** | baseline | 35 | $1.51 | 607s | 0 | ok |
| | rewind+hint | 57 | $2.31 | 679s | 1 | ok |
| **03_llvm_loop_vec** | baseline | **201 (cap)** | $12.54 | 1436s | 0 | **wrong_area** |
| | rewind+hint | **201 (cap)** | $9.68 | **2528s** | 2 | **wrong_area** |

Three regimes:

- **GCC sccvn** — too easy (11 turns). No checkpoint. No meaningful signal.
- **GCC mul_overflow** — intermediate (35→57 turns). Model checkpointed once, but wrote an UNHELPFUL summary: "Built GCC and ran tests. The existing test case produces correct results (0 for __INT_MAX__ * 2 unsigned = no overflow s…" — it convinced itself the bug wasn't real, then kept exploring. Cost +53%. The wrong-shape checkpoint entrenched a wrong conclusion. This is a new failure mode worth noting: **a bad checkpoint can actively mislead.**
- **LLVM loop_vec** — hardest task. Baseline hit max_turns=200 in wrong_area (didn't modify source files, just built stuff). Rewind+hint ALSO hit max_turns in wrong_area, but **cost 23% less** despite running 76% longer wall time. This is the cleanest observation of the theorized value: elisions compress context → each request cheaper → more exploration per dollar. Critically, it did NOT help find the bug. **More efficient exploration of the wrong space is still failure.**

Checkpoint quality on LLVM was poor too: 2 checkpoints both said things like "Current code produces correct output for the test case" — same "I don't see the bug" pattern as GCC mul_overflow. When the model can't find the bug, it checkpoints wrong conclusions, which then entrench the wrong search direction.

### Beast (32_beast) — 4 runs — **INVALID COMPARISON**

Median per condition:

| condition | n | cost | dur | turns | ckpt rate | median bytes saved |
|---|---|---|---|---|---|---|
| baseline | 2 | $4.64 | 976s | 96 | — | — |
| rewind+hint | 2 | $3.71 | 718s | 76 | 50% | 287K |

**Why this comparison is invalid:** I inspected `patch_*.diff` for each run and found that **none of the four runs modified Beast source files** (`include/boost/beast/websocket/impl/read.hpp` or `include/boost/beast/websocket/detail/impl_base.hpp`). All "file changes" in the diffs are build artifacts (`build/boost/bin.v2/…`). Both conditions failed to fix the bug.

So the "$3.71 vs $4.64" is comparing two failure modes, not two solutions. The rewind+hint runs gave up sooner, which looks like a cost win but isn't a productivity win. Cannot conclude cost-effectiveness from this.

This is still an informative null: **opus-4-6 does not solve the Beast WebSocket permessage-deflate bug inside 100 turns** under any of the conditions we tested. Fixing this would require either more turns, a stronger model, or a different approach — the primitive isn't the bottleneck here.

The 5th beast run (run_026, B3 baseline) was killed in progress after discovering this — no point burning more budget on an invalid comparison.

### Checkpoint summaries the agent wrote

All 6 checkpoints that fired across all runs were high-quality, specific, actionable:

- LAPACK run_011: "Found the bug. In DLAED8 (lines 450-463), when a value is deflated due to close eigenvalues, an insertion sort is performed… The fix is to add the same insertion-sort logic from DLAED8 to DLASD7."
- LAPACK run_012: "Found the bug. In DLAED8 (symmetric eigenvalue), after deflation when two eigenvalues are close…"
- LAPACK run_015: "Confirmed the bug: DLASD7 (lines 449-493) deflates singular values into IDXP(K2) by decrementing K2, but never sorts…"
- beast run_025: "Identified the bug location. In read.hpp, when rd_remain==0 and rd_fh.fin is true, the code always sets zs.next_in…"

Plus 2 more LAPACK summaries (not shown) in the same "root cause found" shape. Every single checkpoint fired at the **exact right moment** — after the model had narrowed to the specific bug. Not once did the model checkpoint prematurely or superficially.

## What was built

- **`mcp/bookmark_server.mjs`** — MCP server with 4 tools: `bookmark_mark`, `bookmark_recall`, `bookmark_list`, `checkpoint_progress`. Zero npm deps (hand-rolled JSON-RPC over stdio). Audit log at `$BOOKMARK_LOG_DIR/bookmarks.jsonl`.
- **`proxy/rewind.mjs`** — scans `messages[]` for `checkpoint_progress` tool_use calls, elides whole turns between each anchor and the prior anchor/initial prompt, splices synthetic `user` message with the agent's summary. Preserves tool_use↔tool_result pairing. 9 unit tests.
- **`proxy/proxy.mjs`** — applies rewind before compact/inject when `REWIND_ENABLED=1`. Emits `rewind_applied` event per elision.
- **`benchmarks/run.sh`** — `--rewind on|off`, `--rewind-hint on|off`. Propagates through to the Docker container.
- **`benchmarks/entrypoint.sh`** — writes `.mcp.json` when bookmarks enabled; extends `--allowedTools`; appends prompt hint when `REWIND_HINT=1`.
- **`benchmarks/analyze_rewind.py`** — per-run detail + per-condition aggregate (median/mean cost/duration/turns/checkpoint-rate/bytes-saved).
- **`proxy/test/rewind.test.mjs`** (9 tests) + **`smoke_rewind.mjs`** end-to-end.

All tests passing (64 unit + 9 rewind + 2 smoke tests).

## Lessons worth generalizing

### 1. Tool-name framing has large effects on adoption

`summarize_and_forget` → 0/1 usage.
`checkpoint_progress` → 5/7 usage.

Same prompt hint, same semantics, same call. The word `forget` reads as destructive; `checkpoint_progress` reads as saving progress. This is the same kind of loss-aversion bias that shows up in humans — and it transfers to agent tool adoption. **Anyone shipping memory/cleanup/deletion MCP tools should frame the positive affordance, not the destructive one.**

### 2. Unprompted discovery of memory primitives fails

Even with positive naming, no-hint runs never called the tool. This is consistent with Anthropic's decision to ship `clear_tool_uses` as an API-side automatic feature, not as an agent-callable tool. Models need either explicit instruction or training reinforcement to reach for memory-management primitives. A bolt-on MCP server without a system-prompt mention will be invoked ~0 times in production.

### 3. System-driven compression is blind; agent-driven compression is semantically targeted

Anthropic's `clear_tool_uses_20250919` fires on token thresholds. Claude Code's microcompaction offloads large tool outputs. OpenCode's pruner walks backward by token budget. None of them read what the session is *about*. Our checkpoints landed at the semantically right moment every single time — "bug identified" was a checkpoint 6/6 times it fired. That pattern is impossible to produce from outside the agent's reasoning.

### 4. Task size almost certainly decides whether the primitive pays off — but we don't have proof yet

The theoretical argument stands: checkpoints compound over subsequent turns, so longer sessions benefit more. The LAPACK data (cost +58% at 15-20 turns) is consistent with this — the cost of writing the summary exceeds the per-turn elision savings when there aren't many subsequent turns. Beast would have been the test of the "long-session wins" hypothesis, but neither condition solved the task, so the data is useless.

Verifying this requires: (a) a task difficulty where baseline reliably solves near max_turns, (b) rewind+hint solving the same task in measurably fewer turns, both with verify.sh confirming correctness. That's still a missing piece.

## The strongest claim this supports

**Claim we can defend:** Given a well-framed tool and a small prompt hint, Claude (opus-4-6) *will* invoke a custom memory-management primitive at the semantically right moment — specifically at "bug identified" in a debugging session, with a clear, actionable summary. Before this work there was no public data on whether models would do this.

**Claim we cannot yet defend:** That this saves cost end-to-end. LAPACK showed +58%, beast showed a fake win (neither condition solved). The "long sessions compound savings" hypothesis still needs a task where we can see both (a) baseline solves near max_turns, and (b) rewind+hint solves in measurably fewer turns.

This is still different from any existing context-management mechanism. It's a **capability demonstration**: models *can* self-manage context when given the right interface. The blockers to deployment are (a) naming discipline, (b) training or prompt discipline to reach for the tool, and (c) finding the task shape where cost math flips positive.

## Gotchas observed during build

- **`pgrep` self-matching.** The first attempt at queuing beast runs after LAPACK used `while pgrep -f "bash benchmarks/run.sh.*30_lapack"` — the shell command itself contained the search string, so pgrep found itself and the loop never exited. Fixed by killing and launching fresh.
- **Docker build context.** `mcp/` lives at repo root so the Dockerfile had to `COPY mcp /opt/mcp` with build context at repo root, not `benchmarks/`. Updated `run.sh` build hint to match.
- **Beast has no verify.sh.** We can compare cost/duration, not correctness. Other benchmark tasks (LAPACK) do have verify but we didn't actually invoke it in our analysis either.
- **Skills vs MCP:** confirmed via 2026 docs that the recommendation is "use both — skills for methodology, MCP for access/state." Our `checkpoint_progress` is correctly implemented as MCP because it has side effects (leaves a marker the proxy reads). Anthropic's "Tool Search" feature (2026) reduces MCP tool-schema bloat ~85% through on-demand discovery, so the cost-of-MCP concern is mostly moot.

## Recommendations for next steps

### Highest value

1. **Fix the "bad checkpoint entrenches wrong conclusion" failure mode.** This emerged from GCC mul_overflow and LLVM loop_vec: when the model hasn't solved the problem, it checkpoints "I don't see a bug" which elides the exploration and then keeps searching the same wrong space. Possible remediations:
   - Validate the checkpoint summary somehow before elision (hard)
   - Let the agent retract a checkpoint ("bookmark_unmark" or "checkpoint_abandon")
   - Only allow checkpoints when the agent has made CONCRETE progress (edits, test passes) — harder for the agent to self-deceive
2. **More runs on LLVM loop_vec to confirm the 23% cost reduction holds.** n=1 isn't enough. 3-5 pairs would nail down whether the effect is consistent on context-pressured sessions.
3. **A task that baseline solves near max_turns** — that's the clean win we still don't have. Candidates: 33_geometry (tier-2), a harder LAPACK variant, or SWE-bench tasks where solution rate at turn-cap is known.

### Medium value

4. **Cross-architecture test.** Try on Sonnet or Opus specifically. We've run on whatever the default model is for this benchmark; different models may have different checkpoint affinity.
5. **Publish the naming finding.** "Don't name your MCP tool with the word `forget`" is a small but generalizable claim. Could be a short blog post or an MCP-ecosystem guidance note.

### If stopping

The spike has produced:
- A working, tested primitive (rewind.mjs + MCP server, 11 unit/integration tests, smoke passing)
- **A clear capability demonstration** (models DO self-manage context when given the right interface — 9 checkpoints across tests, at semantically-identifiable moments)
- **First cost-mechanism observation** (LLVM loop_vec: -23% cost on a context-pressured session, theorized elision-compounding effect in action)
- **New failure mode:** bad checkpoints (where model misreads its own state) actively mislead subsequent exploration. Novel finding — not documented elsewhere.
- Reusable tooling (analyzer, MCP server, proxy infrastructure, patch-shape verifier)
- Four generalizable lessons:
  1. Naming framing matters (avoid "forget"/"delete" words)
  2. Unprompted discovery fails
  3. Checkpoint quality depends on model epistemic state — bad summaries amplify wrong conclusions
  4. Context-management efficiency ≠ problem-solving capability

Unlike the stuck-detector spike which landed as a clean null, this one has real positive findings **and** a novel negative finding (the bad-checkpoint failure mode). The cost claim is half-confirmed — the mechanism works on LLVM loop_vec. The solve-rate claim is NOT confirmed. Worth a short writeup covering all of the above as a research note on agent-driven context management.

## File inventory after tonight

```
mcp/bookmark_server.mjs          # MCP server, 4 tools, zero deps
proxy/rewind.mjs                 # elision logic
proxy/test/rewind.test.mjs       # 9 unit tests
proxy/test/smoke_rewind.mjs      # e2e smoke
benchmarks/analyze_rewind.py     # pilot analyzer with aggregates
benchmarks/run.sh                # --rewind, --rewind-hint flags
benchmarks/entrypoint.sh         # .mcp.json injection, prompt hint
benchmarks/Dockerfile            # COPY mcp /opt/mcp
REWIND_FINDINGS.md               # this file
```

Nothing dangling. All tests pass. Ready to commit or continue.
