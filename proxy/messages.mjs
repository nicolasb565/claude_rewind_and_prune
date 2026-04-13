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
 * Extract every tool call and its output from the full message history,
 * in assistant-turn order. Used on the first API call to build ring buffer
 * state from existing context.
 *
 * @param {Array} messages  full messages array from request body
 * @returns {{ toolName: string, input: object, output: string }[]}
 */
export function extractAllToolCalls(messages) {
  // First pass: collect all tool_use blocks keyed by id
  const pending = new Map()
  for (const msg of messages) {
    if (msg.role === 'assistant' && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === 'tool_use') {
          pending.set(block.id, { toolName: block.name, input: block.input ?? {}, output: '' })
        }
      }
    }
  }

  // Second pass: match tool_results to their tool_use entries
  for (const msg of messages) {
    if (msg.role === 'user' && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === 'tool_result' && pending.has(block.tool_use_id)) {
          pending.get(block.tool_use_id).output = extractResultText(block.content)
        }
      }
    }
  }

  // Third pass: emit in original assistant-turn order
  const ordered = []
  for (const msg of messages) {
    if (msg.role === 'assistant' && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === 'tool_use' && pending.has(block.id)) {
          ordered.push(pending.get(block.id))
        }
      }
    }
  }
  return ordered
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
