/**
 * Agent-initiated rewind-with-summary.
 *
 * When the agent calls the MCP tool `summarize_and_forget(summary)`,
 * this module rewrites subsequent requests to elide the failed branch
 * and splice the agent-authored summary in its place.
 *
 * Semantics:
 * - Walk messages[] and find each `summarize_and_forget` tool_use
 *   (in the assistant role, nested in content[] blocks).
 * - Each such call becomes an "anchor": everything between it and
 *   the previous anchor (or the initial user prompt) is eligible
 *   for elision.
 * - Elide whole turns only — an assistant message with tool_use and
 *   its paired user message with tool_result are removed as a unit
 *   to keep the API from rejecting orphaned references.
 * - Splice a synthetic user text message at the elision point:
 *     "Previous exploration summarized by agent: {summary}"
 * - The `summarize_and_forget` tool_use/result pair stays visible so
 *   the agent can see its call succeeded.
 *
 * The first user message (the original prompt) is never elided.
 */

const TOOL_NAME_SUFFIX = 'checkpoint_progress'

function isSummarizeToolUse(block) {
  if (!block || block.type !== 'tool_use') return false
  const name = block.name ?? ''
  // MCP tools arrive as `mcp__<server>__<tool>`; we match by suffix so
  // the server can be renamed without touching this module.
  return name === TOOL_NAME_SUFFIX || name.endsWith(`__${TOOL_NAME_SUFFIX}`)
}

function extractSummaryText(input) {
  // Structured schema: progress_type + finding + evidence + next_direction.
  // Fallback to legacy `summary` field for older checkpoints still in history.
  if (input?.finding || input?.evidence || input?.next_direction) {
    const label = input.progress_type === 'approach_eliminated'
      ? 'Approach eliminated'
      : 'Milestone'
    const parts = [`${label}: ${input.finding ?? ''}`]
    if (input.evidence) parts.push(`Evidence: ${input.evidence}`)
    if (input.next_direction) parts.push(`Next: ${input.next_direction}`)
    return parts.join('. ')
  }
  return input?.summary ?? ''
}

function findSummarizeInMessage(msg) {
  if (msg?.role !== 'assistant' || !Array.isArray(msg.content)) return null
  for (const b of msg.content) {
    if (isSummarizeToolUse(b)) {
      return { summary: extractSummaryText(b.input), id: b.id }
    }
  }
  return null
}

/**
 * Locate all summarize_and_forget calls in the messages array.
 * Returns an ordered list of anchor records, each with the assistant
 * message index carrying the tool_use and the summary string.
 */
function findAnchors(messages) {
  const anchors = []
  for (let i = 0; i < messages.length; i++) {
    const hit = findSummarizeInMessage(messages[i])
    if (hit) anchors.push({ assistantIdx: i, summary: hit.summary })
  }
  return anchors
}

/**
 * Compute the "preserved" index range around an anchor. We keep:
 *   - the assistant message carrying the summarize tool_use, and
 *   - the immediately-following user message (which must carry the
 *     tool_result for that tool_use; the API requires the pair).
 */
function anchorPair(messages, assistantIdx) {
  const end = assistantIdx
  // The next user message MUST carry the tool_result for the
  // summarize_and_forget call — Claude Code always emits it
  // immediately after the assistant turn.
  const next = assistantIdx + 1
  const hasPair = next < messages.length && messages[next]?.role === 'user'
  return { start: end, end: hasPair ? next : end }
}

/**
 * Apply all pending rewinds. Returns { messages, elisions } where
 * `elisions` is an array of { anchor, bytesSaved, turnsElided, summary }.
 *
 * We rebuild the array from scratch rather than mutate, to keep the
 * caller safe and preserve the input for comparison.
 */
export function applyRewind(messages) {
  if (!Array.isArray(messages) || messages.length === 0) {
    return { messages, elisions: [] }
  }

  const anchors = findAnchors(messages)
  if (anchors.length === 0) return { messages, elisions: [] }

  const elisions = []
  const out = []

  // The original prompt is always the very first message. Keep it.
  out.push(messages[0])

  // `cursor` walks through the input array. We've emitted index 0.
  let cursor = 1

  for (const anchor of anchors) {
    const pair = anchorPair(messages, anchor.assistantIdx)

    // Elide the slice (cursor .. pair.start-1) — replace with a single
    // synthetic user message carrying the agent's summary. This is the
    // whole point: drop the failed exploration, keep the lesson.
    const elideFrom = cursor
    const elideTo = pair.start // exclusive
    if (elideTo > elideFrom) {
      const elidedSlice = messages.slice(elideFrom, elideTo)
      const bytesSaved = JSON.stringify(elidedSlice).length
      out.push({
        role: 'user',
        content: [{
          type: 'text',
          text: `[Previous exploration elided — agent summary: ${anchor.summary}]`,
        }],
      })
      elisions.push({
        anchorIdx: anchor.assistantIdx,
        turnsElided: elideTo - elideFrom,
        bytesSaved,
        summary: anchor.summary,
      })
    }

    // Preserve the anchor pair itself (the tool_use + its tool_result).
    for (let i = pair.start; i <= pair.end; i++) out.push(messages[i])
    cursor = pair.end + 1
  }

  // Everything after the last anchor passes through unchanged.
  for (let i = cursor; i < messages.length; i++) out.push(messages[i])

  return { messages: out, elisions }
}
