/**
 * Inject Anthropic's native `clear_tool_uses_20250919` strategy into
 * outgoing requests so the API clears old tool_results on its side —
 * with cache-aware batching via `clear_at_least`.
 *
 * Claude Code already opts into the `context-management-2025-06-27`
 * beta and ships `clear_thinking_20251015` by default, but does NOT
 * enable tool-result clearing. We add it.
 *
 * We preserve any existing edits (notably `clear_thinking_20251015`)
 * and never duplicate our own edit if it is already present.
 *
 * Read/Grep/Glob outputs are excluded because clearing them caused
 * regressions in prior experiments.
 */

const EDIT_TYPE = 'clear_tool_uses_20250919'
const DEFAULT_EXCLUDE = ['Read', 'Grep', 'Glob']

export function getConfig() {
  return {
    trigger_input_tokens: intEnv('INJECT_TRIGGER_INPUT_TOKENS', 30000),
    keep_tool_uses: intEnv('INJECT_KEEP_TOOL_USES', 3),
    clear_at_least_input_tokens: intEnv('INJECT_CLEAR_AT_LEAST', 5000),
    exclude_tools: (process.env.INJECT_EXCLUDE_TOOLS || DEFAULT_EXCLUDE.join(','))
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean),
  }
}

function intEnv(key, fallback) {
  const v = parseInt(process.env[key] || '', 10)
  return Number.isFinite(v) && v > 0 ? v : fallback
}

/**
 * Build the edit entry we inject. The API expects each edit to carry
 * a `type` plus strategy-specific knobs. See Anthropic context-editing
 * docs for the shape.
 */
function buildEdit(cfg) {
  return {
    type: EDIT_TYPE,
    trigger: { type: 'input_tokens', value: cfg.trigger_input_tokens },
    keep: { type: 'tool_uses', value: cfg.keep_tool_uses },
    clear_at_least: {
      type: 'input_tokens',
      value: cfg.clear_at_least_input_tokens,
    },
    exclude_tools: cfg.exclude_tools,
  }
}

/**
 * Merge the clear_tool_uses edit into `body.context_management.edits`.
 * Mutates and returns the body for convenience.
 *
 * If the client already has our edit type we leave it alone — the
 * client's config wins. We only fill the gap.
 */
export function injectClearToolUses(body, cfg = getConfig()) {
  if (!body || typeof body !== 'object') return body

  const cm = body.context_management ?? {}
  const edits = Array.isArray(cm.edits) ? [...cm.edits] : []

  if (edits.some((e) => e?.type === EDIT_TYPE)) {
    return body
  }

  edits.push(buildEdit(cfg))
  body.context_management = { ...cm, edits }
  return body
}
