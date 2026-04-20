/**
 * Extract the Anthropic `usage` object from a response body.
 *
 * Handles three shapes the upstream may emit:
 *   1. Non-streaming JSON response: { ..., "usage": { ... } }
 *   2. SSE streaming: `event: message_start\ndata: {"type":"message_start",
 *      "message":{..., "usage": {...}}}`
 *   3. SSE streaming message_delta: `event: message_delta\ndata: {"type":
 *      "message_delta","usage": {...}}` — cumulative update, not prefixed
 *      by a "message" wrapper.
 *
 * We take whichever appears first. The message_start usage has all the
 * cache fields we need (input_tokens, cache_creation_input_tokens,
 * cache_read_input_tokens, output_tokens). That is enough for a cache
 * hit/miss analysis; message_delta only refreshes output_tokens.
 */

export function extractUsage(text) {
  if (!text || typeof text !== 'string') return null

  // SSE: find the first `data: {...}` payload that carries a usage object.
  // We scan line-by-line so we don't match `usage` occurring inside a
  // later event by accident.
  for (const line of text.split('\n')) {
    if (!line.startsWith('data: ')) continue
    const payload = line.slice(6).trim()
    if (!payload || payload === '[DONE]') continue
    let obj
    try { obj = JSON.parse(payload) } catch { continue }
    const usage = obj?.message?.usage ?? obj?.usage
    if (usage && typeof usage === 'object') return usage
  }

  // Non-streaming: try parsing the whole body as JSON.
  try {
    const obj = JSON.parse(text)
    if (obj?.usage && typeof obj.usage === 'object') return obj.usage
  } catch { /* not a complete JSON — ignore */ }

  return null
}
