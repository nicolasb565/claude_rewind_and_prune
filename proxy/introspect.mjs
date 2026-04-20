/**
 * Request-body introspection — summarize what a client is asking the
 * Anthropic API for, without dumping the whole (multi-MB) body.
 *
 * We care about: which beta features the client opted into, whether it
 * set native `context_management`, message/tool/system sizes, and the
 * model. Enough to tell us if we're duplicating work Anthropic already
 * does at the API layer.
 */

export function summarizeRequest({ body, headers }) {
  const out = {
    model: body?.model ?? null,
    stream: body?.stream === true,
    max_tokens: body?.max_tokens ?? null,
    betas: parseBetas(headers),
    message_count: Array.isArray(body?.messages) ? body.messages.length : 0,
    tool_count: Array.isArray(body?.tools) ? body.tools.length : 0,
    system_size_chars: systemSize(body?.system),
    body_size_chars: null, // filled by caller (pre-stringify is free)
  }

  if (body?.context_management) {
    out.context_management = summarizeContextManagement(body.context_management)
  }

  // Inventory tool_result sizes and staleness — this is what Claude Code's
  // native microcompaction *would* target. If we see lots of big ones
  // surviving in the request, Claude Code isn't pruning them.
  const tri = tallyToolResults(body?.messages)
  if (tri) out.tool_results = tri

  return out
}

function parseBetas(headers) {
  if (!headers) return []
  const v = headers['anthropic-beta'] ?? headers['Anthropic-Beta']
  if (typeof v !== 'string') return []
  return v.split(',').map((s) => s.trim()).filter(Boolean)
}

function systemSize(system) {
  if (typeof system === 'string') return system.length
  if (Array.isArray(system)) {
    return system.reduce((n, b) => n + (b?.text?.length ?? 0), 0)
  }
  return 0
}

function summarizeContextManagement(cm) {
  // Capture just the names of strategies in use plus their knobs — we
  // want to know "is Claude Code opted in" and at what thresholds.
  if (!cm || typeof cm !== 'object') return null
  const edits = Array.isArray(cm.edits) ? cm.edits : []
  return {
    edit_count: edits.length,
    strategies: edits.map((e) => ({
      type: e.type,
      trigger: e.trigger ?? null,
      keep: e.keep ?? null,
      clear_at_least: e.clear_at_least ?? null,
      exclude_tools: e.exclude_tools ?? null,
    })),
  }
}

function tallyToolResults(messages) {
  if (!Array.isArray(messages)) return null
  const sizes = []
  let offloadedMarkers = 0
  for (const m of messages) {
    if (m.role !== 'user' || !Array.isArray(m.content)) continue
    for (const b of m.content) {
      if (b.type !== 'tool_result') continue
      const c = b.content
      const text = typeof c === 'string'
        ? c
        : Array.isArray(c)
          ? c.filter((x) => x?.type === 'text').map((x) => x.text ?? '').join('\n')
          : ''
      sizes.push(text.length)
      // Claude Code's disk-offload leaves a short preview referencing a
      // path. Detect heuristically — the preview is small AND mentions a
      // file path hint. We report the count so we can see native
      // microcompaction in action.
      if (text.length < 4096 && /\bsaved to [\w/.-]+|offloaded|\[truncated\]/i.test(text)) {
        offloadedMarkers++
      }
    }
  }
  if (!sizes.length) return { count: 0 }
  sizes.sort((a, b) => a - b)
  return {
    count: sizes.length,
    total_chars: sizes.reduce((a, b) => a + b, 0),
    max_chars: sizes[sizes.length - 1],
    p50_chars: sizes[Math.floor(sizes.length / 2)],
    p90_chars: sizes[Math.floor(sizes.length * 0.9)],
    offloaded_markers: offloadedMarkers,
  }
}
