#!/usr/bin/env node
/**
 * v5 per-step MLP proxy for Claude Code.
 *
 * Intercepts Anthropic API requests, scores each tool call through the stuck
 * detector, and injects a corrective nudge when the MLP score exceeds the
 * threshold. No modifications to Claude Code required.
 *
 * Usage:
 *   node proxy/proxy.mjs &
 *   ANTHROPIC_BASE_URL=http://localhost:8080 claude "your prompt"
 *
 * Environment:
 *   PROXY_PORT         listen port (default: 8080)
 *   PROXY_UPSTREAM     upstream API (default: https://api.anthropic.com)
 *   STUCK_ENABLED      enable stuck detection (default: 1)
 *   COMPACT_ENABLED    enable Bash output compaction (default: 0)
 */

import { createServer } from 'node:http'
import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { loadMLP } from './mlp.mjs'
import { SessionDetector } from './detector.mjs'
import { NudgeController } from './nudge.mjs'
import {
  extractAllToolCalls,
  extractLastTurnToolCalls,
  getSessionKey,
  recentToolSummary,
} from './messages.mjs'
import { log, logRequest } from './log.mjs'
import { fetchUpstream, getStats } from './upstream.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))

const PORT = parseInt(process.env.PROXY_PORT || '8080', 10)
const UPSTREAM = process.env.PROXY_UPSTREAM || 'https://api.anthropic.com'
const COMPACT_ENABLED = process.env.COMPACT_ENABLED === '1'
const STUCK_ENABLED = process.env.STUCK_ENABLED !== '0'

const mlp = loadMLP(resolve(__dirname, 'stuck_weights.json'))
const { threshold } = JSON.parse(readFileSync(resolve(__dirname, 'stuck_config.json'), 'utf8'))

let compact = null
if (COMPACT_ENABLED) {
  const mod = await import('./compact.mjs')
  compact = mod.compact
}

log('proxy_start', {
  port: PORT,
  upstream: UPSTREAM,
  compactEnabled: COMPACT_ENABLED,
  stuckEnabled: STUCK_ENABLED,
  classifier: 'mlp_v5',
  threshold,
  ...getStats(),
})

// ── Session state ──────────────────────────────────────────────────────────────

/** Map<sessionKey, { detector, nudge, turnCounter, initialized }> */
const sessions = new Map()

function getSession(key) {
  if (!sessions.has(key)) {
    sessions.set(key, {
      detector: new SessionDetector(mlp),
      nudge: new NudgeController({ threshold }),
      turnCounter: 0,
      initialized: false,
    })
  }
  return sessions.get(key)
}

// ── Core detection logic ───────────────────────────────────────────────────────

function pruneIfStuck(messages) {
  const key = getSessionKey(messages)
  const session = getSession(key)
  session.turnCounter++

  const prevInitialized = session.initialized
  const toolCalls = session.initialized
    ? extractLastTurnToolCalls(messages)
    : extractAllToolCalls(messages)
  session.initialized = true

  let lastScore = 0
  try {
    for (const tc of toolCalls) {
      lastScore = session.detector.addStep(tc.toolName, tc.input, tc.output)
    }
  } catch (e) {
    // Rebuild detector so partial ring state doesn't corrupt future calls;
    // restore initialized so the full-scan path runs again next call
    session.initialized = prevInitialized
    session.detector = new SessionDetector(mlp)
    throw e
  }

  log('mlp_score', {
    turn: session.turnCounter,
    score: +lastScore.toFixed(4),
    threshold,
    stuck: lastScore >= threshold,
  })

  const recentTools = recentToolSummary(messages)
  const { fire, level, text } = session.nudge.update(lastScore, session.turnCounter, recentTools)

  if (!fire) return messages

  log('nudge_injected', {
    turn: session.turnCounter,
    score: lastScore,
    nudgeLevel: level,
    recentTools: recentTools.slice(-5),
  })

  return [...messages, { role: 'user', content: [{ type: 'text', text }] }]
}

// ── HTTP server ────────────────────────────────────────────────────────────────

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

      if (COMPACT_ENABLED && compact && Array.isArray(parsed.messages)) {
        parsed.messages = compact(parsed.messages, log)
      }

      if (STUCK_ENABLED && Array.isArray(parsed.messages)) {
        parsed.messages = pruneIfStuck(parsed.messages)
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
      try {
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          res.write(value)
        }
      } catch (e) {
        log('stream_error', { error: e.message })
      } finally {
        reader.releaseLock()
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
  log('proxy_listening', { port: PORT })
  process.stderr.write(`[proxy] Listening on :${PORT} — MLP v5, threshold=${threshold}\n`)
})
