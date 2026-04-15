/**
 * Pure message-processing helpers for Claude Code API request bodies.
 *
 * Extracted from the main request handler so they can be unit-tested without
 * starting the HTTP server.
 */

/**
 * Extract text from a tool_result content block.
 * Multi-block results are joined with newlines to match nlile.py's "\n".join()
 *
 * @param {Array|string} content  tool_result content (array of blocks or raw string)
 * @returns {string}
 */
export function extractResultText(content) {
  if (Array.isArray(content)) {
    return content
      .filter((b) => b.type === 'text')
      .map((b) => b.text ?? '')
      .join('\n')
  }
  return String(content ?? '')
}

/**
 * Extract every tool call and its output from the full message history.
 *
 * Emission order matches src/pipeline/parsers/nlile.py `parse_session`:
 * a single walk over messages, staging tool_use blocks in `pending` and
 * emitting them when their matching tool_result is seen (in tool_result
 * order, not tool_use order). Orphan tool_uses — ones without a matching
 * tool_result — are flushed at the end with empty output. This parity is
 * critical because the LR classifier was trained on Python-parsed
 * features; any reordering would shift history-dependent values.
 *
 * @param {Array} messages  full messages array from request body
 * @returns {{ toolName: string, input: object, output: string }[]}
 */
export function extractAllToolCalls(messages) {
  const pending = new Map()
  const steps = []
  for (const msg of messages) {
    if (!Array.isArray(msg.content)) continue
    for (const block of msg.content) {
      if (!block || typeof block !== 'object') continue
      const btype = block.type
      if (btype === 'tool_use') {
        pending.set(block.id, {
          toolName: block.name,
          input: block.input ?? {},
          output: '',
        })
      } else if (btype === 'tool_result') {
        const tid = block.tool_use_id
        if (pending.has(tid)) {
          const tc = pending.get(tid)
          pending.delete(tid)
          tc.output = extractResultText(block.content)
          steps.push(tc)
        }
      }
    }
  }
  // Flush orphan tool_uses (no matching tool_result) at the end.
  for (const tc of pending.values()) steps.push(tc)
  return steps
}

/**
 * Extract tool calls only from the most recent assistant turn.
 * Used on subsequent API calls — only the new turn needs processing.
 *
 * @param {Array} messages  full messages array from request body
 * @returns {{ toolName: string, input: object, output: string }[]}
 */
export function extractLastTurnToolCalls(messages) {
  const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant')
  if (!lastAssistant || !Array.isArray(lastAssistant.content)) return []

  // Precompute result map in one pass — O(N) vs O(N·M) inner loop
  const resultMap = new Map()
  for (const msg of messages) {
    if (msg.role !== 'user' || !Array.isArray(msg.content)) continue
    for (const b of msg.content) {
      if (b.type === 'tool_result') {
        resultMap.set(b.tool_use_id, extractResultText(b.content))
      }
    }
  }

  const toolCalls = []
  for (const block of lastAssistant.content) {
    if (block.type !== 'tool_use') continue
    toolCalls.push({
      toolName: block.name,
      input: block.input ?? {},
      output: resultMap.get(block.id) ?? '',
    })
  }
  return toolCalls
}

/**
 * Derive a stable session key from the message history.
 * Uses the first 200 chars of the first user message — matches the original proxy.
 *
 * @param {Array} messages
 * @returns {string}
 */
export function getSessionKey(messages) {
  for (const msg of messages) {
    if (msg.role === 'user') {
      const text = Array.isArray(msg.content)
        ? msg.content.map((b) => b.text ?? '').join('')
        : String(msg.content)
      return text.slice(0, 200) || '__default__'
    }
  }
  return '__default__'
}

/**
 * Build a list of recent tool call summaries for nudge text (last 8 calls).
 *
 * @param {Array} messages
 * @returns {string[]}
 */
export function recentToolSummary(messages) {
  const tools = []
  for (const msg of messages.slice(-20)) {
    if (!Array.isArray(msg.content)) continue
    for (const block of msg.content) {
      if (block.type === 'tool_use') {
        const detail =
          block.input?.command ??
          block.input?.file_path ??
          block.input?.pattern ??
          block.input?.description ??
          block.input?.prompt ??
          ''
        tools.push(`${block.name}: ${String(detail).slice(0, 60)}`)
      }
    }
  }
  return tools.slice(-8)
}
