/**
 * Auto-compact: truncate stale Bash tool outputs in the messages array.
 * Only Bash outputs are compacted — Read/Edit/Write/Grep/Glob are never touched.
 *
 * Idempotent: compacted blocks carry a `[COMPACTED` prefix so repeated
 * calls (each request re-sends full history) skip already-compacted blocks.
 */

const defaults = {
  staleTurns: 2,
  keepFirst: 30,
  keepLast: 10,
  minLines: 50,
}

export function getConfig() {
  return {
    staleTurns: parseInt(process.env.COMPACT_STALE_TURNS || defaults.staleTurns, 10),
    keepFirst: parseInt(process.env.COMPACT_KEEP_FIRST || defaults.keepFirst, 10),
    keepLast: parseInt(process.env.COMPACT_KEEP_LAST || defaults.keepLast, 10),
    minLines: parseInt(process.env.COMPACT_MIN_LINES || defaults.minLines, 10),
  }
}

export function compact(messages, log) {
  const cfg = getConfig()

  // Build tool_use_id → tool_name map from assistant messages
  const toolNames = new Map()
  for (const msg of messages) {
    if (msg.role !== 'assistant' || !Array.isArray(msg.content)) continue
    for (const block of msg.content) {
      if (block.type === 'tool_use' && block.id && block.name) {
        toolNames.set(block.id, block.name)
      }
    }
  }

  // Count assistant turns for staleness tracking
  let turnCount = 0
  const turnAt = new Map()
  for (let i = 0; i < messages.length; i++) {
    if (messages[i].role === 'assistant') turnCount++
    turnAt.set(i, turnCount)
  }

  let totalSaved = 0
  let compactCount = 0

  const result = messages.map((msg, i) => {
    if (msg.role !== 'user' || !Array.isArray(msg.content)) return msg
    if (turnCount - (turnAt.get(i) || 0) < cfg.staleTurns) return msg

    let modified = false
    const newContent = msg.content.map((block) => {
      if (block.type !== 'tool_result') return block

      // Only compact Bash outputs
      const name = toolNames.get(block.tool_use_id) || ''
      if (name !== 'Bash') return block

      // Check if already compacted — handle all possible content shapes.
      // Claude Code may re-wrap our string content as [{type:"text", text:"..."}]
      const rawContent = block.content
      if (typeof rawContent === 'string' && rawContent.startsWith('[COMPACTED')) return block
      if (Array.isArray(rawContent)) {
        const firstText = rawContent.find((b) => b.type === 'text')
        if (firstText?.text?.startsWith('[COMPACTED')) return block
      }

      // Extract text content
      const text =
        typeof rawContent === 'string'
          ? rawContent
          : Array.isArray(rawContent)
            ? rawContent
                .filter((b) => b.type === 'text')
                .map((b) => b.text)
                .join('\n')
            : ''

      if (!text) return block

      const lines = text.split('\n')
      if (lines.length < cfg.minLines) return block

      const truncated = [
        ...lines.slice(0, cfg.keepFirst),
        `\n[... ${lines.length - cfg.keepFirst - cfg.keepLast} lines compacted ...]\n`,
        ...lines.slice(-cfg.keepLast),
      ].join('\n')

      const compacted = `[COMPACTED — ${lines.length} lines → ${cfg.keepFirst + cfg.keepLast}]\n${truncated}`
      const savedChars = text.length - compacted.length

      if (savedChars <= 0) return block

      modified = true
      compactCount++
      totalSaved += Math.round(savedChars / 4)

      log?.('compact', {
        toolName: name,
        toolUseId: block.tool_use_id,
        originalLines: lines.length,
        compactedLines: cfg.keepFirst + cfg.keepLast,
        tokensSavedEstimate: Math.round(savedChars / 4),
      })

      return { ...block, content: compacted }
    })

    return modified ? { ...msg, content: newContent } : msg
  })

  if (compactCount > 0) {
    log?.('compact_summary', {
      compactCount,
      totalTokensSaved: totalSaved,
    })
  }

  return result
}
