# Benchmarks v1 — Implementation Plan

This document is the authoritative spec for building the `benchmarks/` harness.
Written after extensive design discussion. Read top-to-bottom in one pass before
implementing. Scope is v1 only — anything tagged "v2" or "deferred" is out.

---

## Goals

1. **Reproducibly measure** whether the v5 stuck detector + nudge improves
   real-world Claude Code performance on out-of-distribution tasks.
2. **Two-mode comparison**: proxy off (baseline) vs proxy on (with stuck
   detection enabled). Same tasks, same model, same fixtures.
3. **Per-task Docker isolation** so 12 tasks run in parallel without state leak
   and with clean reproducibility.
4. **Headline metric**: `(off median duration) - (on median duration)` per
   stuck-prone task. We are NOT chasing a pass/fail score for v1.

## Non-goals (v1)

- Strict compile-and-test verify scripts for every task. Only 3 trivial verify
  scripts (sqlite, rbtree, minicoro). Other tasks: just measure turns + duration.
- Nudge strategy A/B (current vs long-loop focused). That is a separate
  experiment after the baseline benchmark works. Tracked in
  `memory/project_next_steps.md`.
- Cross-model comparison (Sonnet vs Opus). Default to Opus 4.6 for everything
  in v1; the manifest schema supports per-task `model` override but we won't
  use it yet.
- CI integration. Local-Linux-only for v1.
- Pretty result dashboards. Plain JSON + a small Python script that prints a
  comparison table is enough.

---

## Folder structure

```
benchmarks/
  PLAN.md                  # this file
  README.md                # how to use the harness
  manifest.json            # canonical task list + global config
  Dockerfile               # debian:trixie + pinned Claude Code + toolchains
  setup.sh                 # build image, shallow-clone fixtures
  run.sh                   # spawn proxy, fan out tasks in parallel containers
  compare.py               # read results/run_NNN/, print table
  tasks/
    01_gcc_sccvn/
      task.md              # the prompt for `claude -p`
      verify.sh            # optional, exit 0 = pass
      fixture.json         # repo url, ref, optional build steps
    02_gcc_mul_overflow/
    03_llvm_loop_vec/
    04_sqlite_cte/
    06_django_async/
    07_react_hooks/
    08_express_async/
    24_rbtree/
    27_minicoro_cet/
    30_lapack/
    32_beast/
    33_geometry/
  fixtures/                # gitignored — populated by setup.sh
  results/                 # gitignored — per-run logs/stats
```

Add to root `.gitignore`:
```
benchmarks/fixtures/
benchmarks/results/
benchmarks/.env
```

---

## `manifest.json` schema

JSON (matches the rest of the codebase — `training_manifest.json`,
`stuck_config.json`, `package.json`).

```json
{
  "schema_version": 1,
  "claude_code_version": "1.0.45",
  "default_model": "claude-opus-4-6",
  "default_max_turns": 100,
  "tasks": [
    {
      "id": "01_gcc_sccvn",
      "tier": 1,
      "max_turns": 200,
      "fixture": {
        "url": "git://gcc.gnu.org/git/gcc.git",
        "ref": "6225251b9~1",
        "shallow": true,
        "build_cmd": null
      }
    },
    {
      "id": "27_minicoro_cet",
      "tier": 1,
      "max_turns": 200,
      "model": "claude-opus-4-6",
      "fixture": {
        "url": "https://github.com/edubart/minicoro.git",
        "ref": "main",
        "shallow": true,
        "build_cmd": "make tests"
      }
    }
  ]
}
```

Per-task fields:
- `id` — directory name under `tasks/`, used as session key in proxy logs
- `tier` — 1 (high-value stuck-prone), 2 (medium), 3 (fast / FP control)
- `max_turns` — passed to `claude -p --max-turns`. Per-task in manifest.
- `model` — optional override. Defaults to `default_model` if absent.
- `fixture.url`, `fixture.ref` — for `git clone --filter=blob:none --depth 1`
- `fixture.build_cmd` — optional build step run during `setup.sh` so the
  fixture has a known compiled state before the agent starts. Most tasks
  set this to `null` (the agent does its own build).

---

## The 12 tasks

| Tier | ID | Source repo | What | Why |
|---|---|---|---|---|
| 1 | `01_gcc_sccvn` | gcc | Bug PR 123310 (`-1U` vs `-1` in tree-ssa-sccvn.cc) | Known stuck case from previous benchmark, max 34 turns |
| 1 | `02_gcc_mul_overflow` | gcc | Bug PR 123864 (match.pd unsigned 0xFFFFFFFF) | Known stuck case, 29-140 turns |
| 1 | `03_llvm_loop_vec` | llvm-project | Loop vectorization miscompile | The worst stuck case in previous data: 1-161 turns, 8s-53min |
| 1 | `24_rbtree` | synthetic | Fix red-black tree deletion rebalancing | The famous 671s→45s case from earlier nudge experiments |
| 1 | `27_minicoro_cet` | edubart/minicoro | Fix segfault in debian:trixie docker (CET shadow stack) | User's real-world systems debugging case, completely OOD |
| 2 | `06_django_async` | django/django | Async middleware error not caught by error handler | Variable 14-51 turns |
| 2 | `30_lapack` | Reference-LAPACK/lapack | Numerical bug in a routine | Fortran, completely OOD |
| 2 | `32_beast` | boostorg/boost (libs/beast) | HTTP/WebSocket bug | C++ template-heavy, 22-60 turns |
| 2 | `33_geometry` | boostorg/boost (libs/geometry) | Add a feature | Long but consistent — should NOT trigger nudge (FP control) |
| 3 | `04_sqlite_cte` | sqlite/sqlite | Wrong results with recursive CTE | Trivially easy, has verify.sh |
| 3 | `07_react_hooks` | facebook/react | useEffect stale closure | Quick |
| 3 | `08_express_async` | expressjs/express | Async middleware error | The known FP case — critical FP control |

**6 stuck-prone, 6 productive** — balanced for measuring both halves of F1.

`max_turns` per tier:
- Tier 1: **200** (capture even the 161-turn worst case with headroom)
- Tier 2: **100**
- Tier 3: **60**

---

## Dockerfile

`benchmarks/Dockerfile`:

```dockerfile
FROM debian:trixie

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ gfortran make cmake ninja-build \
    python3 python3-pip python3-venv \
    nodejs npm \
    git curl wget ca-certificates \
    libssl-dev zlib1g-dev \
    linux-headers-generic \
    && rm -rf /var/lib/apt/lists/*

# Pin Claude Code version for reproducibility
RUN npm install -g @anthropic-ai/claude-code@1.0.45

# Entry point script that takes a task id, runs claude -p, captures output
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /work
ENTRYPOINT ["/entrypoint.sh"]
```

Notes:
- `linux-headers-generic` is needed for the minicoro CET reproduction (kernel 6.4+ shadow stack API).
- Pin `claude-code@1.0.45` — bump deliberately, don't rely on `latest`.
- Keep the image lean — fixtures are bind-mounted, not baked in.
- Build once, used by all tasks. ~2 GB image is fine.

`benchmarks/entrypoint.sh` (called inside container with task id as arg):

```bash
#!/bin/bash
set -euo pipefail
TASK_ID="$1"
SCRATCH=/scratch
OUTPUT=/output

# Copy fixture to writable scratch — original mount is read-only
cp -a /work/. "$SCRATCH/"
cd "$SCRATCH"

# Read task metadata from manifest
MODEL=$(jq -r ".tasks[] | select(.id==\"$TASK_ID\") | .model // .default_model" /manifest.json)
MAX_TURNS=$(jq -r ".tasks[] | select(.id==\"$TASK_ID\") | .max_turns // .default_max_turns" /manifest.json)
PROMPT=$(cat /tasks/"$TASK_ID"/task.md)

START=$(date +%s)
claude -p \
  --model "$MODEL" \
  --max-turns "$MAX_TURNS" \
  --allowedTools "Bash,Edit,Write,Read,Grep,Glob" \
  "$PROMPT" \
  > "$OUTPUT/stdout.log" 2> "$OUTPUT/stderr.log"
EXIT=$?
END=$(date +%s)

# Record summary
cat > "$OUTPUT/summary.json" <<EOF
{
  "task_id": "$TASK_ID",
  "model": "$MODEL",
  "max_turns": $MAX_TURNS,
  "duration_seconds": $((END - START)),
  "exit_code": $EXIT
}
EOF

# Run verify if present
if [ -x /tasks/"$TASK_ID"/verify.sh ]; then
  bash /tasks/"$TASK_ID"/verify.sh "$SCRATCH" \
    > "$OUTPUT/verify.log" 2>&1
  echo "{\"verify_exit\": $?}" > "$OUTPUT/verify.json"
fi
```

---

## Per-task Docker isolation

Each task runs in its own ephemeral container. Benefits:
1. **Fixture isolation by construction** — no `git clean` needed
2. **Trivial parallelism** — 12 containers in parallel, each gets its own
   session key in the proxy (proxy keys by first user message; all 12 prompts
   are different)
3. **No state leak** — when the task ends, the container dies, FS dies with it
4. **Session attribution is clean** — proxy logs show exactly which session
   belongs to which task

Per-task `docker run` invocation (from `run.sh`):

```bash
docker run --rm -d \
  --name "bench_${RUN_ID}_${TASK_ID}" \
  --network host \
  -v "$REPO/benchmarks/manifest.json:/manifest.json:ro" \
  -v "$REPO/benchmarks/tasks:/tasks:ro" \
  -v "$REPO/benchmarks/fixtures/$TASK_ID:/work:ro" \
  -v "$REPO/benchmarks/results/run_${RUN_ID}/$TASK_ID:/output" \
  $AUTH_MOUNTS \
  -e ANTHROPIC_BASE_URL="$PROXY_URL" \
  benchmark-runner "$TASK_ID"
```

`--network host` lets the container reach the proxy on `localhost:8080`
directly. This works on Linux only — v1 is Linux-only.

`AUTH_MOUNTS` depends on the chosen auth mode (next section).

---

## Dual auth

Two modes, chosen at runtime via `run.sh --auth subscription|env`:

### Subscription mode (default)
Mounts the host's Claude credentials into the container:
```bash
AUTH_MOUNTS="-v $HOME/.claude/.credentials.json:/root/.claude/.credentials.json:ro"
```
Uses the same login token as `claude` on the host. No extra setup needed if
the user is already logged in. This is the default because that's how the
user actually runs Claude Code.

### Env mode (CI / no subscription)
Reads from `benchmarks/.env`:
```bash
AUTH_MOUNTS="--env-file $REPO/benchmarks/.env"
```
The `.env` file contains:
```
ANTHROPIC_API_KEY=sk-ant-...
```
`benchmarks/.env.example` is committed (with placeholder); `benchmarks/.env`
is gitignored.

---

## Fixture caching

Cloning huge repos (gcc, llvm) over the network on every run is a no-go.
Strategy: **shallow-clone once into a host-side cache, copy on every run**.

`setup.sh` does:
```bash
for task in $(jq -r '.tasks[].id' manifest.json); do
  url=$(jq -r ".tasks[] | select(.id==\"$task\") | .fixture.url" manifest.json)
  ref=$(jq -r ".tasks[] | select(.id==\"$task\") | .fixture.ref" manifest.json)
  fixture_dir="benchmarks/fixtures/$task"

  if [ -d "$fixture_dir" ]; then
    echo "$task already cached"; continue
  fi

  # Shallow clone with single ref — minimizes disk usage
  # For non-HEAD refs, fetch the SHA directly
  git clone --filter=blob:none --depth 1 "$url" "$fixture_dir"
  (cd "$fixture_dir" && git fetch --depth 1 origin "$ref" && git checkout "$ref")

  # Optional pre-build (set in fixture.build_cmd)
  build=$(jq -r ".tasks[] | select(.id==\"$task\") | .fixture.build_cmd" manifest.json)
  if [ "$build" != "null" ]; then
    (cd "$fixture_dir" && bash -c "$build")
  fi
done
```

At run time the container mounts `benchmarks/fixtures/$TASK_ID` **read-only**;
`entrypoint.sh` does `cp -a /work/. /scratch/` to get a writable copy. No
fixture corruption possible from a misbehaving agent.

`cp -a` of a 200MB tree takes ~2 seconds; `git clone` from network takes ~30s.
Per-run savings dominate.

Some refs may need special handling — gcc/llvm git servers don't always allow
shallow fetch by raw SHA. Fall back to `git clone --depth 1 --branch <branch>`
+ `git checkout <sha>` when needed; document per-task in `fixture.json`
comments if any task needs a special fetch sequence.

---

## verify.sh strategy

Three tasks have `verify.sh`. Others rely on turn count / duration metrics
and have no verify script.

### `tasks/04_sqlite_cte/verify.sh`
Runs the test query against the patched sqlite, compares output to expected:
```bash
#!/bin/bash
SCRATCH=$1
cd "$SCRATCH"
make sqlite3 || exit 1
./sqlite3 :memory: < /tasks/04_sqlite_cte/test.sql > actual.txt
diff actual.txt /tasks/04_sqlite_cte/expected.txt
```

### `tasks/24_rbtree/verify.sh`
Runs the rbtree unit test binary:
```bash
#!/bin/bash
SCRATCH=$1
cd "$SCRATCH"
make test
```

### `tasks/27_minicoro_cet/verify.sh`
Runs the minicoro test binary, expects exit code 0 (no segfault):
```bash
#!/bin/bash
SCRATCH=$1
cd "$SCRATCH"
make tests || exit 2  # build failed
./tests/test_minicoro
```

For the other 9 tasks, `verify.sh` does not exist. The runner records
`verify_exit: null` in summary.json. The benchmark conclusion comes from
turn/duration trend, not pass/fail.

---

## Run mechanics

`benchmarks/run.sh` flags:
```bash
benchmarks/run.sh \
  --runs 1 \                      # how many times to repeat each task (default 1)
  --proxy on \                    # on | off (default off)
  --auth subscription \           # subscription | env (default subscription)
  --tasks 01_gcc_sccvn,03_llvm_loop_vec  # optional filter (default: all)
```

Behavior:
1. Auto-increment run ID by scanning `benchmarks/results/` for highest
   `run_NNN/` and using `run_$((NNN+1))/`
2. If `--proxy on`, kill any process on :8080, start `node proxy/proxy.mjs &`
   in background, wait 2s, verify it's alive
3. For each requested task:
   - Spawn the docker container (background, with `&`)
   - Track PID
4. Wait for all PIDs to finish
5. If proxy was started, kill it
6. Print summary by reading `results/run_NNN/*/summary.json`

Repeating with `--runs 5` runs each task 5 times in the same `run_NNN/`
folder, with `summary_1.json`, `summary_2.json`, etc. The user said the
number-of-runs *is* the cost-protection mechanism, so make it a clear
required-thinking parameter.

### A typical full benchmark cycle

```bash
# One-time
bash benchmarks/setup.sh                          # builds image, clones fixtures (~5 GB total)

# Baseline pass
bash benchmarks/run.sh --runs 3 --proxy off       # ~3 × ~10 min = ~30 min wall

# Treatment pass
bash benchmarks/run.sh --runs 3 --proxy on        # same scope, with stuck detector

# Compare
python benchmarks/compare.py results/run_001 results/run_002
```

`compare.py` reads both run dirs, prints a per-task table:
```
task                duration off (med)   duration on (med)   delta    nudge fired
03_llvm_loop_vec    2840s                450s                -84%     7 times
01_gcc_sccvn        920s                 720s                -22%     2 times
...
```

---

## Minicoro task prompt

`benchmarks/tasks/27_minicoro_cet/task.md`:

```markdown
The minicoro test program in this directory runs successfully on the host
machine (Ubuntu 25.10, kernel 6.x) but crashes with a segfault when run
inside the debian:trixie Docker container we are currently in.

Investigate the root cause of the segfault, then fix the crash so that
running `make tests && ./tests/test_minicoro` inside this container exits
cleanly with no segfault.

You may modify the minicoro source as needed. The fix should not require
upgrading the kernel or the C library — it must work in this exact
container environment.
```

The verify is "did the test binary exit 0 inside the container?" The agent
already knows about CET shadow stack (per Nicolas's prior runs with Opus 4.6),
but it failed to actually fix the crash. This task measures whether the
nudge can break the agent out of the loop where it diagnoses but doesn't fix.

---

## Output / results format

`benchmarks/results/run_NNN/`:
```
run_NNN/
  manifest_snapshot.json   # copy of manifest.json at run time
  proxy_events.jsonl       # if --proxy on, copied from ~/.stuck-detector/logs/
  run_meta.json            # {start, end, proxy_on, auth_mode, runs}
  01_gcc_sccvn/
    summary_1.json         # {task_id, duration_seconds, exit_code, model, max_turns}
    stdout_1.log
    stderr_1.log
    verify_1.json          # if verify.sh ran
    verify_1.log
    summary_2.json         # second run, if --runs >= 2
    ...
  02_gcc_mul_overflow/
  ...
```

Everything is JSON or plain log. `compare.py` is the only consumer; it can
evolve without breaking the on-disk format.

---

## Open questions deferred to v2

- **Nudge strategy A/B**: the long-loop-focused variant (raise silent buffer
  to ~5 turns, skip soft level). Implement after baseline benchmark works,
  via a `NUDGE_STRATEGY=A|B` env var on the proxy.
- **Per-model comparison**: same task with Sonnet vs Opus to see which gets
  more value from the proxy.
- **macOS / Windows support**: currently `--network host` is Linux-only.
- **Verify scripts for the other 9 tasks**: incremental, can be added one
  at a time later.
- **Cost dashboard / total token spend tracking**: out of scope.
- **Statistical comparison** (Wilcoxon, etc.): for v1 just eyeball medians.

---

## Implementation order

1. **`benchmarks/Dockerfile` + `entrypoint.sh`** — build the image, run a
   trivial `echo hello` task to verify the image works
2. **`benchmarks/manifest.json`** — full 12-task spec, per-task `task.md` and
   `fixture.json` files (no verify scripts yet)
3. **`benchmarks/setup.sh`** — fixture clone loop, run it for one task
   (`24_rbtree`, smallest), verify the fixture lands in
   `benchmarks/fixtures/24_rbtree/`
4. **`benchmarks/run.sh`** — single-task path first (`--tasks 24_rbtree`),
   subscription auth only, no proxy, no parallelism. Verify the container
   runs claude -p end-to-end and produces summary.json
5. **Add proxy startup/teardown** to run.sh, verify proxy session shows up
   in `~/.stuck-detector/logs/events-*.jsonl`
6. **Add parallelism** — 2-task parallel test, then full 12
7. **Add `--auth env` mode**, test with a dummy API key in `.env`
8. **Write the 3 verify.sh scripts** (sqlite, rbtree, minicoro)
9. **Run setup.sh** for all 12 fixtures
10. **First real benchmark**: `--runs 3 --proxy off` baseline, then
    `--runs 3 --proxy on`
11. **`benchmarks/compare.py`** — table output
12. **`benchmarks/README.md`** — usage, prerequisites, expected duration

Each step is a single small commit. Don't batch.

---

## Acceptance criteria (v1 done when)

1. `bash benchmarks/setup.sh` finishes without errors and produces 12
   fixtures under `benchmarks/fixtures/`
2. `bash benchmarks/run.sh --runs 1 --proxy off` finishes within ~30 minutes
   wall time and produces a `results/run_NNN/` directory with all 12
   summary.json files
3. `bash benchmarks/run.sh --runs 1 --proxy on` finishes and the proxy
   event log shows MLP scores for each task's session
4. `python benchmarks/compare.py results/run_001 results/run_002` prints a
   table comparing off vs on
5. README.md explains how to run the benchmark from a clean checkout
6. The 3 stuck-prone "trivially verifiable" tasks (sqlite, rbtree, minicoro)
   produce a `verify_*.json` exit code

That's it. No claims about whether the nudge actually helps until v1 is
running and we have data.

---

## Things to NOT do during implementation

- **Don't tune the nudge strategy** while building the harness. The current
  proxy config is the baseline; nudge tuning is a v2 experiment with its own
  measurement.
- **Don't add features to the manifest schema** beyond what's specified here.
  YAGNI — extend later when there's a real need.
- **Don't try to make verify scripts for all 12 tasks**. The 3 specified are
  enough for v1.
- **Don't bake fixtures into the Docker image**. Bind-mounted with cp-to-scratch
  is the right pattern.
- **Don't run tasks sequentially "to be safe"**. Parallel-by-container is
  the design — each container is isolated by construction.
- **Don't change the proxy code while building benchmarks**. The proxy is
  shipped (commit 13624a9). Treat it as a fixed dependency.
- **Don't add the no_step_idx_no_has_prior or any other ablation variant**.
  We decided to keep all 7 features. Move on.

---

## Reference: what the current proxy does

(Recap so a fresh-context Claude doesn't need to re-derive it.)

- Listens on :8080, intercepts `/v1/messages` POSTs to api.anthropic.com
- For each request, parses tool calls from the message history, scores each
  tool call with the v5 MLP (42-dim input, 7 features, F1 ≈ 0.955)
- If the latest score crosses threshold (0.5), the NudgeController decides
  whether to fire a corrective message into the next user turn
- Nudge levels: -1 silent absorb (first hit), 0 soft, 1 medium, 2 hard
- Default cooldowns: [1, 4, 8, 8] turns indexed by `nudgeLevel + 1`
- Logs every event to `~/.stuck-detector/logs/events-YYYY-MM-DD.jsonl`
- Session keying: first 200 chars of the first user message

The proxy is started with `node proxy/proxy.mjs &`. It needs no args. It
reads `proxy/stuck_weights.json` and `proxy/stuck_config.json` from disk on
startup.
