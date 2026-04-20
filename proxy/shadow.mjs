/**
 * Shadow-summary + rewind middleware for the Anthropic API proxy.
 *
 * Each /v1/messages request that arrives from the agent (Claude Code,
 * or any API-compatible client) triggers a parallel "shadow" call to a
 * cheap model (default: Haiku). The shadow evaluates whether the
 * session has hit a verified milestone or is stuck in a loop and
 * returns a structured YES/NO + SUMMARY.
 *
 * On shadow YES (with cooldown clear), the proxy rewrites the agent's
 * outgoing messages[] to:
 *   1. keep the first user message (the original goal),
 *   2. drop everything between the goal and the most-recent
 *      assistant/user pair (which carries the verification evidence),
 *   3. splice a synthetic user message containing the shadow's summary.
 *
 * Safety valves:
 *   - The first user message (goal) is always preserved.
 *   - The most-recent assistant+user pair is preserved (concrete recent
 *     evidence that the agent can cross-check against the summary).
 *   - A cooldown prevents repeated rewinds within a few turns.
 *   - Any shadow error is fail-open: original messages forwarded
 *     unchanged. The agent request never fails because of the shadow.
 *
 * Environment:
 *   SHADOW_ENABLED           Master switch (default: 0)
 *   SHADOW_MODEL             Model for the shadow call
 *                            (default: claude-haiku-4-5-20251001)
 *   SHADOW_COOLDOWN          Turns to skip after a rewind (default: 3)
 *   SHADOW_MIN_MESSAGES      Min messages[] length before firing
 *                            (default: 6 — skip trivial sessions)
 *   SHADOW_MAX_TOKENS        max_tokens for the shadow response
 *                            (default: 400)
 *   SHADOW_LOG_SUMMARIES     Log full shadow summaries in JSONL
 *                            (default: 1)
 */

import { createHash } from 'node:crypto'
import { fetchUpstream } from './upstream.mjs'

const SHADOW_MODEL = process.env.SHADOW_MODEL || 'claude-haiku-4-5-20251001'
const SHADOW_COOLDOWN = parseIntEnv('SHADOW_COOLDOWN', 3)
const SHADOW_MIN_MESSAGES = parseIntEnv('SHADOW_MIN_MESSAGES', 6)
const SHADOW_MAX_TOKENS = parseIntEnv('SHADOW_MAX_TOKENS', 400)
const SHADOW_LOG_SUMMARIES = process.env.SHADOW_LOG_SUMMARIES !== '0'
// Hard deadline on the shadow call. The fail-open path kicks in on
// timeout; without this we inherit upstream.mjs's retry budget (can
// stretch to minutes on a bad upstream) and stall the agent request.
const SHADOW_TIMEOUT_MS = parseIntEnv('SHADOW_TIMEOUT_MS', 10000)
// Cap to prevent unbounded growth of the session state map across a
// long-running proxy. On overflow we evict in insertion order.
const SHADOW_MAX_SESSIONS = parseIntEnv('SHADOW_MAX_SESSIONS', 1000)

function parseIntEnv(key, fallback) {
  const v = parseInt(process.env[key] || '', 10)
  return Number.isFinite(v) && v >= 0 ? v : fallback
}

// Per-session state. Keyed by the hash of the first user message,
// which is stable within a conversation and distinct across tasks.
const sessions = new Map() // sessionKey -> { cooldownRemaining, rewindCount }

// ── SHADOW_QUERY — used as the system prompt for the shadow call ──────────
// The shadow's `messages[]` is the agent's history (up to the current
// turn); its `system` is this directive. That pattern avoids the
// role-alternation ambiguity of appending SHADOW_QUERY as a user turn.
export const SHADOW_QUERY = (
  "You are in REFLECTION mode, NOT action mode. Do NOT emit any tool_use " +
  "blocks. Your entire response must be plain text using exactly these " +
  "three labeled fields:\n\n" +
  "SHOULD_CHECKPOINT: YES if the most recent tool_result just provided " +
  "concrete evidence of a verified outcome. Two firing cases:\n" +
  "  - A change was just made AND verified by an observable signal (a " +
  "    test passing, a build succeeding, expected program output, a " +
  "    required artifact present, or equivalent confirmation). If the " +
  "    original task has multiple independent goals, the verified " +
  "    completion of ANY ONE of them fires YES — do not wait for the " +
  "    whole task to finish.\n" +
  "  - An approach was proven wrong — a concrete failure means the " +
  "    current direction cannot work and the session must change " +
  "    strategy. This includes being stuck: if you see the EXACT same " +
  "    tool call (same name AND same arguments) repeated across recent " +
  "    turns AND the tool_results are empty, errors, or unchanged, " +
  "    that counts as proven wrong — fire YES.\n" +
  "    Legitimate iterative progress (different edits, reads of " +
  "    different files, tests showing different pass/fail results each " +
  "    run) is NOT stuck — do not mis-classify it.\n" +
  "A file write, read, edit, or search is an ACTION, not an outcome — " +
  "those cannot justify YES on their own. Only the tool_result that " +
  "VERIFIES the change counts.\n" +
  "NO only if: the most recent change has not yet been verified, OR no " +
  "change has happened yet. Do NOT answer NO merely because OTHER " +
  "pending work remains elsewhere in the task. Focus on whether the " +
  "LAST change just got verified.\n" +
  "REASON: <one-line explanation, <=20 words, citing the specific evidence>\n" +
  "SUMMARY: <2-3 sentences that MUST include concrete references the " +
  "agent will need to continue: specific file paths, function names, " +
  "test names, or other identifiers. Cover: (1) what was verified " +
  "completed and WHERE, (2) what work remains in the original task " +
  "and WHERE, (3) the next concrete step. Do NOT write vague " +
  "references like 'the function' — use 'path/to/file.ext::symbol' or " +
  "equivalent locators.>"
)

// ── parsing ───────────────────────────────────────────────────────────────

const _RE_SHOULD = /SHOULD_CHECKPOINT:\s*(YES|NO)\b/i
const _RE_REASON = /REASON:\s*(.+)/
const _RE_SUMMARY = /SUMMARY:\s*([\s\S]+)/

export function parseShadow(text) {
  if (typeof text !== 'string' || !text) {
    return { should: null, reason: null, summary: null, raw: text ?? '' }
  }
  const mS = _RE_SHOULD.exec(text)
  const mR = _RE_REASON.exec(text)
  const mSum = _RE_SUMMARY.exec(text)
  return {
    should: mS ? mS[1].toUpperCase() : null,
    reason: mR ? mR[1].trim().split(/\r?\n/)[0] : null,
    summary: mSum ? mSum[1].trim() : null,
    raw: text,
  }
}

// ── rewind ────────────────────────────────────────────────────────────────

/**
 * Rewrite messages[] to keep:
 *   - the first user message (goal)
 *   - the most recent assistant + user pair (recent evidence)
 *   - a newly spliced synthetic user message carrying the summary
 *
 * Returns { messages, elided } where `elided` is the number of messages
 * dropped. If the input is too short to safely rewind, returns the
 * input unchanged with elided=0.
 *
 * Pairing rules (matching the Anthropic API's validation):
 *   - An assistant message that emits `tool_use` blocks MUST be
 *     followed by a user message containing the corresponding
 *     `tool_result` blocks. We preserve the pair together.
 *   - If the last message is a user `tool_result`, it goes AFTER the
 *     summary splice so the summary sits before the recent evidence,
 *     not after it.
 */
export function applyShadowRewind(messages, summary) {
  if (!Array.isArray(messages) || messages.length < 4 || !summary) {
    return { messages, elided: 0 }
  }

  // Find the most recent user message carrying a tool_result. That
  // message + its preceding assistant message (which emitted the
  // tool_use) are the "concrete recent evidence" we preserve. If
  // there's no tool_result in the history, there's nothing concrete
  // to fall back on — skip the rewind.
  let lastToolResultIdx = -1
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i]
    if (m?.role === 'user' && Array.isArray(m.content) &&
        m.content.some((b) => b?.type === 'tool_result')) {
      lastToolResultIdx = i
      break
    }
  }
  if (lastToolResultIdx < 0) {
    return { messages, elided: 0 }
  }

  // The paired assistant message is the one immediately before. It
  // MUST exist and be role=assistant for the API's pairing rule to
  // hold; otherwise we'd leave a dangling tool_result.
  const pairedAssistantIdx = lastToolResultIdx - 1
  if (pairedAssistantIdx < 1 || messages[pairedAssistantIdx]?.role !== 'assistant') {
    return { messages, elided: 0 }
  }

  // Cross-check: every tool_use.id in the preserved assistant must have
  // a matching tool_result.tool_use_id in the preserved user message.
  // The Anthropic API rejects unmatched tool_use/result, so if the
  // shape is unusual (e.g., multi-block assistant paired with partial
  // user result) we bail out rather than produce an invalid request.
  if (!toolUseIdsMatch(messages[pairedAssistantIdx], messages[lastToolResultIdx])) {
    return { messages, elided: 0 }
  }

  // Preserved: [goal, ..., pairedAssistantIdx, lastToolResultIdx, ...tail].
  // We also preserve anything after lastToolResultIdx (in case the
  // client appended more messages after the tool_result — rare but
  // possible if the agent emitted a follow-up without a new tool).
  const goal = messages[0]
  const tail = messages.slice(pairedAssistantIdx)
  if (tail.length >= messages.length - 1) {
    // Only the goal would be elided; not worth it.
    return { messages, elided: 0 }
  }

  const syntheticUser = {
    role: 'user',
    content: [{
      type: 'text',
      text: (
        `[Checkpoint note from your prior reasoning — prior exploration elided]\n` +
        `${summary}\n\n` +
        `The next messages show the most-recent verification evidence. ` +
        `Continue with any remaining work from the original task.`
      ),
    }],
  }

  const out = [goal, syntheticUser, ...tail]
  // `elided` counts original messages dropped, not net length change
  // (since we add one synthetic message). If zero, return unchanged.
  const preservedOriginals = 1 + tail.length // goal + tail
  const elided = messages.length - preservedOriginals
  return { messages: elided > 0 ? out : messages, elided: Math.max(0, elided) }
}

/**
 * Verify the API's tool_use/tool_result pairing invariant: every
 * tool_use.id emitted by the assistant must appear as a tool_result
 * in the following user message. Returns true iff the pair is valid
 * (or iff there are no tool_use blocks — nothing to match).
 */
function toolUseIdsMatch(assistantMsg, userMsg) {
  const assistantIds = new Set()
  if (Array.isArray(assistantMsg?.content)) {
    for (const b of assistantMsg.content) {
      if (b?.type === 'tool_use' && typeof b.id === 'string') {
        assistantIds.add(b.id)
      }
    }
  }
  if (assistantIds.size === 0) return true
  if (!Array.isArray(userMsg?.content)) return false
  const userIds = new Set()
  for (const b of userMsg.content) {
    if (b?.type === 'tool_result' && typeof b.tool_use_id === 'string') {
      userIds.add(b.tool_use_id)
    }
  }
  for (const id of assistantIds) {
    if (!userIds.has(id)) return false
  }
  return true
}

// ── shadow call ───────────────────────────────────────────────────────────

/**
 * Build the shadow request body: same messages[] as agent, but:
 *   - model = SHADOW_MODEL
 *   - system = SHADOW_QUERY, marked with cache_control so the
 *     (stable) system prompt is cached across every shadow call
 *   - second-to-last message's final content block is marked
 *     cache_control:ephemeral. Rationale: turn-N's last message ==
 *     turn-(N+1)'s second-to-last. Placing the breakpoint there
 *     keeps the cache anchor at a stable position in the prefix,
 *     which maximizes prefix-hit rate across consecutive turns.
 *     If the message's content is a string we normalize to a text
 *     block so the cache marker has a place to live.
 *   - tools omitted — the shadow must not take action
 */
function buildShadowBody(messages) {
  const out = messages.map((m) => cloneMessage(m))
  // Cache anchor on the messages prefix. Prefer second-to-last for
  // cross-turn hit rate; fall back to the last message for short
  // sessions where there's no second-to-last.
  const anchorIdx = out.length >= 2 ? out.length - 2 : out.length - 1
  attachCacheControl(out, anchorIdx)
  return {
    model: SHADOW_MODEL,
    max_tokens: SHADOW_MAX_TOKENS,
    // System as an array of blocks so we can attach cache_control. The
    // SHADOW_QUERY is immutable, giving us a guaranteed cache anchor
    // regardless of how the messages prefix churns.
    system: [{ type: 'text', text: SHADOW_QUERY, cache_control: { type: 'ephemeral' } }],
    messages: out,
  }
}

/**
 * Normalize the message at `idx` so its last content block carries
 * `cache_control:ephemeral`. Handles three shapes:
 *   - content: Array<block>           → mark last block
 *   - content: string                 → promote to [{type:'text',text,cache_control}]
 *   - anything else                   → no-op (shouldn't happen for
 *                                        well-formed API bodies)
 */
function attachCacheControl(messages, idx) {
  if (idx < 0 || idx >= messages.length) return
  const m = messages[idx]
  if (!m) return
  if (typeof m.content === 'string') {
    m.content = [{
      type: 'text',
      text: m.content,
      cache_control: { type: 'ephemeral' },
    }]
    return
  }
  if (Array.isArray(m.content) && m.content.length > 0) {
    const blocks = [...m.content]
    const tail = blocks[blocks.length - 1]
    if (tail && typeof tail === 'object') {
      blocks[blocks.length - 1] = {
        ...tail,
        cache_control: { type: 'ephemeral' },
      }
      m.content = blocks
    }
  }
}

function cloneMessage(m) {
  if (!m || typeof m !== 'object') return m
  return {
    ...m,
    content: Array.isArray(m.content)
      ? m.content.map((b) => (b && typeof b === 'object' ? { ...b } : b))
      : m.content,
  }
}

/**
 * Execute the shadow call against the upstream API. Returns the parsed
 * shadow response or throws on upstream error.
 *
 * `clientHeaders` is the set of headers from the original agent request,
 * which carry the auth token — we pass a filtered subset through.
 */
async function callShadow({ messages, upstream, clientHeaders, log }) {
  const body = buildShadowBody(messages)
  const headers = buildShadowHeaders(clientHeaders)
  // Don't set content-length: undici's fetch computes it from the
  // string body and overrides any caller-supplied value. Setting it
  // here is a footgun for future edits.

  const res = await fetchUpstream(
    upstream + '/v1/messages',
    {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      redirect: 'follow',
    },
    log,
  )

  if (res.status !== 200) {
    const errText = await res.text().catch(() => '')
    throw new Error(`shadow upstream ${res.status}: ${errText.slice(0, 200)}`)
  }

  const data = await res.json()
  const text = extractResponseText(data)
  return {
    parsed: parseShadow(text),
    usage: data?.usage ?? null,
  }
}

/**
 * Keep only headers needed for a fresh /v1/messages call: auth,
 * versioning, and content-type. Stripping everything else avoids
 * accidentally forwarding a cached content-length / streaming flag from
 * the client that doesn't apply to our side-call.
 */
const _SHADOW_PASSTHROUGH_HEADERS = new Set([
  'authorization',
  'x-api-key',
  'anthropic-version',
  'anthropic-beta',
  'user-agent',
])

function buildShadowHeaders(clientHeaders) {
  const out = {
    'content-type': 'application/json',
    'accept': 'application/json',
    // Ensure the shadow is non-streaming — we need the full JSON.
    'accept-encoding': 'identity',
  }
  if (!clientHeaders) return out
  for (const [k, v] of Object.entries(clientHeaders)) {
    if (_SHADOW_PASSTHROUGH_HEADERS.has(k.toLowerCase())) {
      out[k] = v
    }
  }
  return out
}

function extractResponseText(data) {
  if (!data || typeof data !== 'object') return ''
  const content = data.content
  if (!Array.isArray(content)) return ''
  return content
    .filter((b) => b?.type === 'text' && typeof b.text === 'string')
    .map((b) => b.text)
    .join('\n')
}

// ── session state ─────────────────────────────────────────────────────────

/**
 * Stable session key from the first user message's content. Claude Code
 * restarts each task with a fresh conversation, so the initial user
 * prompt uniquely identifies the task within a day.
 */
export function deriveSessionKey(messages) {
  if (!Array.isArray(messages) || messages.length === 0) return 'empty'
  const first = messages[0]
  if (!first || first.role !== 'user') return 'nongoal'
  const text = stringifyContent(first.content)
  return createHash('sha1').update(text).digest('hex').slice(0, 16)
}

function stringifyContent(content) {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content
    .map((b) => {
      if (typeof b === 'string') return b
      if (b?.type === 'text') return b.text ?? ''
      if (b?.type === 'tool_result') return JSON.stringify(b.content ?? '')
      return JSON.stringify(b)
    })
    .join('\n')
}

function getState(key) {
  let s = sessions.get(key)
  if (!s) {
    s = { cooldownRemaining: 0, rewindCount: 0 }
    sessions.set(key, s)
    // Simple LRU-style eviction in insertion order. Map preserves
    // insertion order, so the first key is the oldest.
    if (sessions.size > SHADOW_MAX_SESSIONS) {
      const evictCount = Math.max(1, Math.floor(SHADOW_MAX_SESSIONS / 10))
      const keys = sessions.keys()
      for (let i = 0; i < evictCount; i++) {
        const k = keys.next().value
        if (k == null) break
        sessions.delete(k)
      }
    }
  }
  return s
}

// Visible for testing.
export function __resetSessions() {
  sessions.clear()
}

/**
 * Wrap the shadow call in a hard deadline. Without this, upstream
 * retries can compound into minutes of blocking before fail-open.
 * Exported for direct unit testing (the internal call site uses the
 * module-level default timeout via `SHADOW_TIMEOUT_MS`).
 */
export async function callShadowWithTimeout(args, timeoutMs = SHADOW_TIMEOUT_MS) {
  const { callShadowFn, ...rest } = args
  const call = callShadowFn(rest)
  let timer
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(
      () => reject(new Error(`shadow timeout after ${timeoutMs}ms`)),
      timeoutMs,
    )
  })
  try {
    return await Promise.race([call, timeout])
  } finally {
    if (timer) clearTimeout(timer)
  }
}

// ── main entry point ──────────────────────────────────────────────────────

/**
 * Run the shadow middleware on an agent request. On success, returns
 * {messages, applied, summary?, elided?, reason?}:
 *   - messages: possibly-rewritten messages[] to forward upstream
 *   - applied: true iff rewind was applied
 *   - summary: the spliced summary (when applied=true)
 *   - elided: number of messages dropped (when applied=true)
 *   - reason: short tag indicating why we did/didn't fire
 *
 * On any error, returns the original messages with applied=false. The
 * proxy forwards the agent request unchanged — the shadow never blocks
 * the agent's work.
 */
export async function shadowAndMaybeRewind({
  messages,
  upstream,
  clientHeaders,
  log,
  callShadowFn = callShadow, // injected for testability
}) {
  const fallback = { messages, applied: false }
  if (!Array.isArray(messages) || messages.length < SHADOW_MIN_MESSAGES) {
    return { ...fallback, reason: 'too_short' }
  }

  const key = deriveSessionKey(messages)
  const state = getState(key)

  if (state.cooldownRemaining > 0) {
    state.cooldownRemaining -= 1
    log?.('shadow_cooldown', { sessionKey: key, remaining: state.cooldownRemaining })
    return { ...fallback, reason: 'cooldown' }
  }

  // Reserve a cooldown slot BEFORE awaiting to prevent two concurrent
  // requests for the same session from both firing the shadow and
  // both applying rewind. If the call fails or returns NO we restore.
  //
  // IMPORTANT: the check above and the write below must remain
  // synchronous — no `await` between them — or two concurrent calls
  // could both pass the gate before either reserves. Current code is
  // safe; future refactors must preserve this invariant.
  const savedCooldown = state.cooldownRemaining
  state.cooldownRemaining = 1

  let shadow
  try {
    shadow = await callShadowWithTimeout({
      messages, upstream, clientHeaders, log, callShadowFn,
    })
  } catch (e) {
    state.cooldownRemaining = savedCooldown
    log?.('shadow_error', { sessionKey: key, error: e.message })
    return { ...fallback, reason: 'shadow_error' }
  }

  const { parsed, usage } = shadow
  log?.('shadow_response', {
    sessionKey: key,
    should: parsed.should,
    reason: parsed.reason,
    summary: SHADOW_LOG_SUMMARIES ? parsed.summary : undefined,
    usage,
  })

  if (parsed.should !== 'YES') {
    state.cooldownRemaining = savedCooldown
    return { ...fallback, reason: 'shadow_no' }
  }
  if (!parsed.summary) {
    state.cooldownRemaining = savedCooldown
    return { ...fallback, reason: 'no_summary' }
  }

  const { messages: rewritten, elided } = applyShadowRewind(messages, parsed.summary)
  if (elided <= 0) {
    state.cooldownRemaining = savedCooldown
    return { ...fallback, reason: 'nothing_to_elide' }
  }

  state.cooldownRemaining = SHADOW_COOLDOWN
  state.rewindCount += 1

  log?.('shadow_rewind_applied', {
    sessionKey: key,
    elided,
    rewindCount: state.rewindCount,
    summary: parsed.summary,
  })

  return {
    messages: rewritten,
    applied: true,
    summary: parsed.summary,
    elided,
    reason: 'fired',
  }
}

/**
 * Diagnostic for the /stats endpoint.
 */
export function getShadowStats() {
  const byKey = {}
  for (const [k, v] of sessions.entries()) {
    byKey[k] = { cooldownRemaining: v.cooldownRemaining, rewindCount: v.rewindCount }
  }
  return {
    model: SHADOW_MODEL,
    cooldown: SHADOW_COOLDOWN,
    min_messages: SHADOW_MIN_MESSAGES,
    timeout_ms: SHADOW_TIMEOUT_MS,
    session_count: sessions.size,
    max_sessions: SHADOW_MAX_SESSIONS,
    sessions: byKey,
  }
}
