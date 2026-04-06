# Task: Build a Context-Aware Proxy for Claude Code

## Goal

Build a local HTTP proxy that sits between Claude Code and the Anthropic API. The proxy intercepts API requests and responses, giving full control over the conversation without patching Claude Code or depending on plugins the model ignores.

Three capabilities:
1. **Auto-compact** — truncate stale Bash tool outputs in the messages array before forwarding
2. **Stuck detection** — analyze thinking blocks for circular reasoning, inject corrective messages
3. **Context pruning** — remove failed approach turns when stuck is detected

Works with vanilla Claude Code via `ANTHROPIC_BASE_URL=http://localhost:PORT claude`.

```
Claude Code (unmodified)
    │
    │  ANTHROPIC_BASE_URL=http://localhost:8080
    │
    ▼
Proxy (localhost:8080)
    │
    ├── Intercept outgoing request
    │   ├── Read messages array
    │   ├── Auto-compact: truncate old Bash tool_results
    │   ├── Stuck check: analyze recent thinking blocks
    │   │   └── If stuck: prune failed turns, inject summary
    │   └── Forward modified request to api.anthropic.com
    │
    ├── Intercept response stream
    │   ├── Buffer thinking deltas for analysis
    │   └── Pass through to Claude Code
    │
    └── Log everything for training data
```

---

## Architecture

### Proxy server (Node.js or Python)

```
stuck-detector-proxy/
├── proxy.mjs               # HTTP proxy server
├── compact.mjs              # Auto-compact logic (truncate old Bash outputs)
├── stuck.mjs                # Stuck detection (heuristic + classifier)
├── prune.mjs                # Context pruning (remove failed turns, inject summary)
├── config.mjs               # Configuration from env vars
├── log.mjs                  # Telemetry logging
├── classifier/
│   ├── model.pkl (or model.onnx)
│   └── inference.py         # is_stuck(text) -> float
├── dataset/                 # Training data (Phase 1)
│   ├── round_01.json
│   └── combined.json
├── training/
│   ├── prepare_data.py
│   ├── train.py
│   └── eval_results.json
└── README.md
```

### What the proxy sees

**Outgoing request body** (POST to /v1/messages):
```json
{
  "model": "claude-opus-4-6",
  "max_tokens": 16000,
  "system": "You are Claude Code...",
  "messages": [
    {"role": "user", "content": [...]},
    {"role": "assistant", "content": [
      {"type": "thinking", "thinking": "Let me look at..."},
      {"type": "text", "text": "I'll read the file."},
      {"type": "tool_use", "id": "toolu_01...", "name": "Bash", "input": {"command": "npm test"}}
    ]},
    {"role": "user", "content": [
      {"type": "tool_result", "tool_use_id": "toolu_01...", "content": "... 200 lines of test output ..."}
    ]}
  ]
}
```

The proxy has direct access to the messages array — it can read, modify, truncate, and prune before forwarding.

**Incoming response stream** (SSE):
```
event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"Let me try..."}}
```

The proxy can buffer thinking deltas for stuck analysis while passing them through.

### Configuration (all via environment variables)

```bash
# Required
ANTHROPIC_BASE_URL=http://localhost:8080  # Tell Claude Code to use proxy

# Proxy settings
PROXY_UPSTREAM=https://api.anthropic.com  # Where to forward (default)
PROXY_PORT=8080                           # Listen port (default)

# Auto-compact
COMPACT_ENABLED=1                         # Enable Bash output truncation (default: 1)
COMPACT_STALE_TURNS=2                     # Turns before truncation (default: 2)
COMPACT_KEEP_FIRST=30                     # Lines to keep from start (default: 30)
COMPACT_KEEP_LAST=10                      # Lines to keep from end (default: 10)
COMPACT_MIN_LINES=50                      # Minimum lines to trigger (default: 50)

# Stuck detection
STUCK_ENABLED=1                           # Enable stuck detection (default: 1)
STUCK_THRESHOLD=0.8                       # Classifier threshold (default: 0.8)
STUCK_COOLDOWN=5                          # Turns between nudges (default: 5)
STUCK_PRUNE_TURNS=3                       # Turns to prune when stuck (default: 3)

# Logging
LOG_DIR=~/.stuck-detector/logs            # Telemetry directory
LOG_REQUESTS=0                            # Log full request bodies (default: 0, large!)
LOG_THINKING=1                            # Log thinking blocks (default: 1, training data)
```

### Usage

```bash
# Start the proxy
node stuck-detector-proxy/proxy.mjs &

# Run vanilla Claude Code through it
ANTHROPIC_BASE_URL=http://localhost:8080 claude "debug this GCC bug..."

# Or for benchmarking — easy A/B:
# Stock (direct to API):
claude -p "..."

# With proxy (compact + stuck detection):
ANTHROPIC_BASE_URL=http://localhost:8080 claude -p "..."
```

No patches, no plugins, no --plugin-dir. Just an environment variable.

---

## Proxy implementation

### Request interception (outgoing)

```javascript
// proxy.mjs — simplified
import { createServer } from 'http';
import { compact } from './compact.mjs';
import { detectStuck, pruneIfStuck } from './stuck.mjs';

const UPSTREAM = process.env.PROXY_UPSTREAM || 'https://api.anthropic.com';
const PORT = process.env.PROXY_PORT || 8080;

createServer(async (req, res) => {
  // Collect request body
  let body = '';
  for await (const chunk of req) body += chunk;

  if (req.url === '/v1/messages' && req.method === 'POST') {
    let parsed = JSON.parse(body);

    // 1. Auto-compact old Bash tool results
    if (process.env.COMPACT_ENABLED !== '0') {
      parsed.messages = compact(parsed.messages);
    }

    // 2. Stuck detection on recent thinking blocks
    if (process.env.STUCK_ENABLED !== '0') {
      parsed.messages = pruneIfStuck(parsed.messages);
    }

    body = JSON.stringify(parsed);
  }

  // Forward to upstream with all original headers (including OAuth)
  const upstream = await fetch(UPSTREAM + req.url, {
    method: req.method,
    headers: {
      ...Object.fromEntries(
        Object.entries(req.headers).filter(([k]) => k !== 'host')
      ),
      'content-length': Buffer.byteLength(body),
    },
    body,
  });

  // Stream response back
  res.writeHead(upstream.status, Object.fromEntries(upstream.headers));
  for await (const chunk of upstream.body) {
    res.write(chunk);
    // Optionally buffer thinking deltas here for analysis
  }
  res.end();
}).listen(PORT, () => console.log(`Proxy on :${PORT} → ${UPSTREAM}`));
```

### Auto-compact (compact.mjs)

```javascript
// Same logic as our v4 patches but operating on API message format
// API format: messages[].role = "user"|"assistant"
// Tool results: role="user", content=[{type:"tool_result", tool_use_id, content}]
// Tool uses: role="assistant", content=[{type:"tool_use", id, name, input}]

export function compact(messages) {
  // Build tool_use_id → tool_name map
  const toolNames = new Map();
  for (const msg of messages) {
    if (msg.role !== 'assistant' || !Array.isArray(msg.content)) continue;
    for (const block of msg.content) {
      if (block.type === 'tool_use') toolNames.set(block.id, block.name);
    }
  }

  // Count assistant turns for staleness
  let turnCount = 0;
  const turnAt = new Map();
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role === 'assistant') turnCount++;
    turnAt.set(i, turnCount);
  }

  const cfg = {
    staleTurns: parseInt(process.env.COMPACT_STALE_TURNS || '2'),
    keepFirst: parseInt(process.env.COMPACT_KEEP_FIRST || '30'),
    keepLast: parseInt(process.env.COMPACT_KEEP_LAST || '10'),
    minLines: parseInt(process.env.COMPACT_MIN_LINES || '50'),
  };

  return messages.map((msg, i) => {
    if (msg.role !== 'user' || !Array.isArray(msg.content)) return msg;
    if (turnCount - (turnAt.get(i) || 0) < cfg.staleTurns) return msg;

    const newContent = msg.content.map(block => {
      if (block.type !== 'tool_result') return block;
      const name = toolNames.get(block.tool_use_id) || '';
      if (name !== 'Bash') return block; // Only compact Bash
      
      const text = typeof block.content === 'string' ? block.content
        : Array.isArray(block.content)
          ? block.content.filter(b => b.type === 'text').map(b => b.text).join('\n')
          : '';
      if (!text || text.startsWith('[COMPACTED')) return block;
      
      const lines = text.split('\n');
      if (lines.length < cfg.minLines) return block;

      const truncated = [
        ...lines.slice(0, cfg.keepFirst),
        `\n[... ${lines.length - cfg.keepFirst - cfg.keepLast} lines compacted ...]\n`,
        ...lines.slice(-cfg.keepLast),
      ].join('\n');

      return {
        ...block,
        content: `[COMPACTED — ${lines.length} lines → ${cfg.keepFirst + cfg.keepLast}]\n${truncated}`,
      };
    });

    return { ...msg, content: newContent };
  });
}
```

### Stuck detection (stuck.mjs)

```javascript
// Analyzes thinking blocks from recent assistant messages.
// If the heuristic flags circular reasoning, injects a nudge
// and optionally prunes the last N turns.

let lastNudgeTurn = -999;
let turnCounter = 0;

export function pruneIfStuck(messages) {
  turnCounter++;
  const cooldown = parseInt(process.env.STUCK_COOLDOWN || '5');
  if (turnCounter - lastNudgeTurn < cooldown) return messages;

  // Extract thinking from the last assistant message
  const lastAssistant = [...messages].reverse().find(m => m.role === 'assistant');
  if (!lastAssistant || !Array.isArray(lastAssistant.content)) return messages;

  let thinking = '';
  for (const block of lastAssistant.content) {
    if (block.type === 'thinking') thinking += block.thinking;
  }
  if (thinking.length < 500) return messages;

  // Run heuristic
  if (!isThinkingSuspicious(thinking)) return messages;

  // TODO: replace heuristic with trained classifier
  // const score = await classify(thinking);
  // if (score < threshold) return messages;

  lastNudgeTurn = turnCounter;

  // Build summary of recent tool calls for context
  const recentTools = [];
  for (const msg of messages.slice(-20)) {
    if (!Array.isArray(msg.content)) continue;
    for (const block of msg.content) {
      if (block.type === 'tool_use') {
        const detail = block.input?.command || block.input?.file_path || '';
        recentTools.push(`${block.name}: ${String(detail).slice(-60)}`);
      }
    }
  }

  // Option A: Just inject a nudge (no pruning)
  const nudge = {
    role: 'user',
    content: [{ type: 'text', text:
      `[CONTEXT MONITOR — turn ${turnCounter}]\n\n` +
      `Your recent thinking shows signs of repeated reasoning patterns. ` +
      `You may be going in circles.\n\n` +
      `Recent tool calls:\n  ${recentTools.slice(-8).join('\n  ')}\n\n` +
      `Review your last few turns critically:\n` +
      `- Are you retrying the same approach with minor variations?\n` +
      `- Are you investigating the same files/functions repeatedly?\n` +
      `- Has your hypothesis changed or are you stuck on the same one?\n\n` +
      `If you are going in circles, try a fundamentally different strategy.`
    }],
  };

  // Option B: Prune last N turns and inject summary (more aggressive)
  // const pruneCount = parseInt(process.env.STUCK_PRUNE_TURNS || '3');
  // return pruneAndInject(messages, pruneCount, nudge);

  // For now, just append the nudge
  return [...messages, nudge];
}

function isThinkingSuspicious(text) {
  // Repeated 20-char substrings appearing 3+ times
  const seen = {};
  for (let i = 0; i < text.length - 20; i += 10) {
    const sub = text.substring(i, i + 20);
    seen[sub] = (seen[sub] || 0) + 1;
    if (seen[sub] >= 3) return true;
  }

  // Circle keywords
  const matches = text.match(
    /\b(try again|let me try|another approach|actually,|wait,|hmm|let me reconsider|that didn't work|same error|still failing)\b/gi
  );
  if (matches && matches.length >= 5) return true;

  // High word overlap between halves
  if (text.length > 2000) {
    const half = Math.floor(text.length / 2);
    const words1 = new Set(text.slice(0, half).toLowerCase().split(/\s+/).filter(w => w.length > 4));
    const words2 = text.slice(half).toLowerCase().split(/\s+/).filter(w => w.length > 4);
    let overlap = 0;
    for (const w of words2) if (words1.has(w)) overlap++;
    if (words2.length > 0 && overlap / words2.length > 0.6) return true;
  }

  return false;
}
```

---

## Why proxy > patches > plugins

| Aspect | Patches (v1-v5) | Plugins | Proxy |
|---|---|---|---|
| Survives updates | ✗ re-patch on every version | ✓ | ✓ |
| Can modify messages | ✓ (direct array access) | ✗ (inject only) | ✓ (full request body) |
| Can read thinking | ✓ (patch stream handler) | ✗ (not in hook input) | ✓ (response stream) |
| Can prune context | ✓ | ✗ | ✓ |
| Can compact outputs | ✓ | ✗ | ✓ |
| Works with vanilla CC | ✗ | partial | ✓ |
| A/B testing | awkward (two binaries) | awkward | trivial (env var) |
| Auth compatibility | n/a (same process) | n/a | needs testing (OAuth passthrough) |
| Model-agnostic | ✗ (CC-specific) | ✗ (CC-specific) | ✓ (any API-compatible client) |

---

## Phases

### Phase 0: Proxy skeleton + auto-compact (1 session)

| Step | Description |
|---|---|
| 1 | Build minimal proxy that forwards requests/responses transparently |
| 2 | Verify Claude Code works through it (ANTHROPIC_BASE_URL) |
| 3 | Verify OAuth auth passes through correctly |
| 4 | Add auto-compact on outgoing requests |
| 5 | Test on GCC bug: stock vs proxy, compare token counts |

### Phase 1: Generate training dataset (2-3 sessions)

| Step | Description |
|---|---|
| 6 | Log all thinking blocks passing through the proxy |
| 7 | Spawn sub-agents on hard tasks, collect reasoning traces |
| 8 | Generate 20 prompts per round, target 15-20 rounds = 300-400 examples |
| 9 | Label each trace: stuck vs productive |
| 10 | Adapt prompts between rounds to target failure modes |

### Phase 2: Train stuck classifier (1 session)

| Step | Description |
|---|---|
| 11 | Prepare data: window traces into 500-1000 token chunks |
| 12 | Train Option A: TF-IDF + Logistic Regression |
| 13 | Evaluate: target >85% recall, >70% precision |
| 14 | If needed, train Option B: small neural classifier |
| 15 | Export model, wire into proxy |

### Phase 3: End-to-end testing (1-2 sessions)

| Step | Description |
|---|---|
| 16 | Test stuck detection + nudge injection on GCC bug |
| 17 | Test pruning mode (more aggressive) |
| 18 | Tune threshold, cooldown, prune depth |
| 19 | Run parallel trials: stock vs proxy, compare results |
| 20 | Collect data for the writeup |

---

## Key reference: Meta-Harness (Lee et al., 2026)

**Paper:** "Meta-Harness: End-to-End Optimization of Model Harnesses" — Yoonho Lee, Roshen Nair, Qizheng Zhang (Stanford), Kangwook Lee (KRAFTON), Omar Khattab (MIT), Chelsea Finn (Stanford). arXiv:2603.28052

Key findings that inform this project:

1. **Raw traces beat summaries.** Ablation: raw execution traces produced dramatically better results than LLM-generated summaries (56.7% vs 38.7%). **Lesson:** save full thinking blocks in logs, don't compress. Train classifier on real text.

2. **The corrective message should prompt causal reasoning.** Most effective behavior: reading prior traces to "identify confounded edits, isolate likely causal changes." **Lesson:** the nudge asks what hypothesis was being tested and what evidence was found, not just "try something different."

3. **Minimal structure, maximum agent autonomy.** Simple outer loop, diagnosis left to agent. **Lesson:** inject a nudge, let the model decide what to do differently. Don't hard-code the fix.

4. **Harness changes are highest-leverage.** Auto-evolved harness beat best human-designed one by 7.7 points using 4x fewer tokens. **Lesson:** the proxy IS a harness improvement. Even a small one has outsized impact.

---

## Previous findings (from patching experiments)

These inform the proxy design:

- **Only compact Bash outputs.** Truncating Read/Edit/Write/Grep/Glob outputs causes the model to re-read files, costing more than it saves. (v1 finding: 8 unnecessary re-reads, 86% slower)
- **Model never uses novel tool parameters.** Added `ephemeral:false` to Bash — model never set it. (v3-v4 finding)
- **Model never calls Rewind.** Even with system prompt, CLAUDE.md, and direct nudges. (v5 finding)
- **Heuristic triggers correctly.** Substring repetition + keyword counting detected spiraling at turns 30 and 75 in a GCC debugging session. (trial 4 finding)
- **Variance dominates.** Same task, same model: 219s to 1731s range. Need many trials for significance.
- **GCC bug is a good test.** PR 123310 (wrong aggregate copy, -1U vs -1 in tree-ssa-sccvn.cc). Model sometimes finds correct fix, sometimes converges on plausible-but-wrong tree-dfa.cc fix. Both paths involve 50-200 tool calls over 15-30 minutes.
