#!/usr/bin/env node
/**
 * HTTP proxy for Claude Code context hygiene experiments.
 *
 * Intercepts Anthropic API requests and either rewrites `messages[]`
 * client-side or injects a native `context_management` strategy so the
 * API can clear history on its side. No modifications to Claude Code.
 *
 * Usage:
 *   node proxy/proxy.mjs &
 *   ANTHROPIC_BASE_URL=http://localhost:8080 claude "your prompt"
 *
 * Environment:
 *   PROXY_PORT                listen port (default: 8080)
 *   PROXY_UPSTREAM            upstream API (default: https://api.anthropic.com)
 *   COMPACT_ENABLED           client-side Bash tool_result truncation (default: 0)
 *   INJECT_CLEAR_TOOL_USES    inject clear_tool_uses_20250919 into requests (default: 0)
 *   REWIND_ENABLED            apply agent-initiated summarize_and_forget elisions
 *                             on outgoing requests (default: 0)
 */

import { createServer } from 'node:http'
import { log, logRequest } from './log.mjs'
import { fetchUpstream, getStats } from './upstream.mjs'
import { extractUsage } from './usage.mjs'
import { summarizeRequest } from './introspect.mjs'
import { injectClearToolUses, getConfig as getInjectConfig } from './inject.mjs'
import { applyRewind } from './rewind.mjs'

const PORT = parseInt(process.env.PROXY_PORT || '8080', 10)
const UPSTREAM = process.env.PROXY_UPSTREAM || 'https://api.anthropic.com'
const COMPACT_ENABLED = process.env.COMPACT_ENABLED === '1'
const INJECT_CLEAR_TOOL_USES = process.env.INJECT_CLEAR_TOOL_USES === '1'
const REWIND_ENABLED = process.env.REWIND_ENABLED === '1'

let compact = null
if (COMPACT_ENABLED) {
  const mod = await import('./compact.mjs')
  compact = mod.compact
}

log('proxy_start', {
  port: PORT,
  upstream: UPSTREAM,
  compactEnabled: COMPACT_ENABLED,
  injectClearToolUses: INJECT_CLEAR_TOOL_USES,
  injectConfig: INJECT_CLEAR_TOOL_USES ? getInjectConfig() : null,
  rewindEnabled: REWIND_ENABLED,
  ...getStats(),
})

const server = createServer(async (req, res) => {
  const chunks = []
  for await (const chunk of req) chunks.push(chunk)
  let body = Buffer.concat(chunks)

  if (req.url === '/stats' && req.method === 'GET') {
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify(getStats()))
    return
  }

  const isMessages = req.url?.startsWith('/v1/messages') && req.method === 'POST'

  if (isMessages) {
    try {
      const parsed = JSON.parse(body.toString())
      const originalCount = parsed.messages?.length ?? 0

      const summary = summarizeRequest({ body: parsed, headers: req.headers })
      summary.body_size_chars = body.length
      log('request_summary', { url: req.url, ...summary })

      if (REWIND_ENABLED && Array.isArray(parsed.messages)) {
        const { messages: rewritten, elisions } = applyRewind(parsed.messages)
        if (elisions.length > 0) {
          parsed.messages = rewritten
          for (const e of elisions) {
            log('rewind_applied', {
              url: req.url,
              anchorIdx: e.anchorIdx,
              turnsElided: e.turnsElided,
              bytesSaved: e.bytesSaved,
              summary: e.summary,
            })
          }
        }
      }

      if (COMPACT_ENABLED && compact && Array.isArray(parsed.messages)) {
        parsed.messages = compact(parsed.messages, log)
      }

      if (INJECT_CLEAR_TOOL_USES) {
        const beforeLen = Array.isArray(parsed.context_management?.edits)
          ? parsed.context_management.edits.length : 0
        injectClearToolUses(parsed)
        const afterLen = parsed.context_management.edits.length
        if (afterLen > beforeLen) {
          log('inject_clear_tool_uses', {
            url: req.url,
            edit_count_before: beforeLen,
            edit_count_after: afterLen,
          })
        }
      }

      body = Buffer.from(JSON.stringify(parsed))
      logRequest(req.method, req.url, originalCount)
    } catch (e) {
      log('parse_error', { error: e.message, url: req.url })
    }
  }

  const headers = {}
  for (const [k, v] of Object.entries(req.headers)) {
    if (k === 'host') continue
    headers[k] = k === 'content-length' ? body.length : v
  }

  try {
    headers['accept-encoding'] = 'identity'
    const upstreamRes = await fetchUpstream(
      UPSTREAM + req.url,
      {
        method: req.method,
        headers,
        body: req.method === 'POST' || req.method === 'PUT' ? body : undefined,
        redirect: 'follow',
      },
      log,
    )

    const resHeaders = {}
    upstreamRes.headers.forEach((v, k) => {
      if (k === 'transfer-encoding' && v === 'chunked') return
      resHeaders[k] = v
    })

    if (upstreamRes.status !== 200) {
      log('upstream_non200', { status: upstreamRes.status, url: req.url })
    }

    res.writeHead(upstreamRes.status, resHeaders)

    if (upstreamRes.body) {
      const reader = upstreamRes.body.getReader()
      // Tee up to 32 KB into a buffer so we can parse the Anthropic
      // usage block (it always lives in the first SSE event or at the
      // root of a non-streaming response). Beyond that we pass through
      // without copying.
      const teeLimit = 32 * 1024
      let teeBuf = ''
      const decoder = new TextDecoder('utf-8', { fatal: false })
      try {
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          res.write(value)
          if (teeBuf.length < teeLimit) {
            teeBuf += decoder.decode(value, { stream: true })
            if (teeBuf.length > teeLimit) teeBuf = teeBuf.slice(0, teeLimit)
          }
        }
      } catch (e) {
        log('stream_error', { error: e.message })
      } finally {
        reader.releaseLock()
      }

      if (isMessages) {
        const usage = extractUsage(teeBuf)
        if (usage) {
          log('cache_stats', {
            url: req.url,
            input_tokens: usage.input_tokens ?? null,
            cache_creation_input_tokens: usage.cache_creation_input_tokens ?? null,
            cache_read_input_tokens: usage.cache_read_input_tokens ?? null,
            output_tokens: usage.output_tokens ?? null,
          })
        }
      }
    }

    res.end()
  } catch (e) {
    log('upstream_error', { error: e.message, url: req.url })
    res.writeHead(502, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ error: 'upstream_error', message: e.message }))
  }
})

server.listen(PORT, () => {
  const actualPort = server.address().port
  log('proxy_listening', { port: actualPort })
  process.stderr.write(`[proxy] Listening on :${actualPort}\n`)
})
