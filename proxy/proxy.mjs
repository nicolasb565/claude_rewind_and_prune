#!/usr/bin/env node
/**
 * LR + three-tier filter proxy for Claude Code.
 *
 * Intercepts Anthropic API requests, scores each tool call through the
 * LR stuck detector + three-tier filter (soft/medium/hard), and injects a
 * corrective nudge when the state machine advances a tier. No modifications
 * to Claude Code required.
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
import { loadLR } from './lr.mjs'
import { LRSessionDetector } from './lr_detector.mjs'
import { TieredNudgeController, DEFAULT_TIERED_CONFIG } from './tiered_filter.mjs'
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

const lr = loadLR(resolve(__dirname, 'lr_weights.json'))

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
  classifier: 'lr_content_v1',
  filterTiers: DEFAULT_TIERED_CONFIG,
  ...getStats(),
})

// ── Session state ──────────────────────────────────────────────────────────────

/** Map<sessionKey, { detector, nudge, turnCounter, initialized }> */
const sessions = new Map()

function getSession(key) {
  if (!sessions.has(key)) {
    sessions.set(key, {
      detector: new LRSessionDetector(lr),
      nudge: new TieredNudgeController(),
      turnCounter: 0,
      initialized: false,
    })
  }
  return sessions.get(key)
}

// ── Core detection logic ───────────────────────────────────────────────────────

function pruneIfStuck(messages) {
  const key = getSessionKey(messages)
  // Short, stable digest of the session key for log attribution.
  // Full 200-char key is too noisy to read; first 64 chars is enough to
  // uniquely identify a benchmark task and match it to tasks/<id>/task.md.
  const sessionKeyPrefix = key.slice(0, 64)
  const session = getSession(key)
  session.turnCounter++

  const prevInitialized = session.initialized
  const toolCalls = session.initialized
    ? extractLastTurnToolCalls(messages)
    : extractAllToolCalls(messages)
  session.initialized = true

  // Text-only turns produce no tool calls. Preserve detector + nudge state
  // across them — a turn with no actions is neither stuck nor productive
  // signal, just absence of evidence.
  if (toolCalls.length === 0) return messages

  let lastResult = null
  try {
    for (const tc of toolCalls) {
      lastResult = session.detector.addStep(tc.toolName, tc.input, tc.output)
    }
  } catch (e) {
    // Rebuild BOTH detector and nudge so partial state can't corrupt
    // future calls. Restore `initialized` so the next call re-scans the
    // full message history into the new detector.
    session.initialized = prevInitialized
    session.detector = new LRSessionDetector(lr)
    session.nudge = new TieredNudgeController()
    throw e
  }

  const { score: lastScore, filters, aggregates } = lastResult

  log('lr_score', {
    sessionKeyPrefix,
    turn: session.turnCounter,
    score: +lastScore.toFixed(4),
    soft: filters.soft,
    medium: filters.medium,
    hard: filters.hard,
    mean2: aggregates.soft != null ? +aggregates.soft.toFixed(4) : null,
    med4: aggregates.medium != null ? +aggregates.medium.toFixed(4) : null,
    med9: aggregates.hard != null ? +aggregates.hard.toFixed(4) : null,
  })

  const recentTools = recentToolSummary(messages)
  const { fire, level, text } = session.nudge.update(
    filters, lastScore, session.turnCounter, recentTools,
  )

  if (!fire) return messages

  log('nudge_injected', {
    sessionKeyPrefix,
    turn: session.turnCounter,
    score: +lastScore.toFixed(4),
    nudgeLevel: level,
    tier: ['soft', 'medium', 'hard'][level],
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
  const actualPort = server.address().port
  log('proxy_listening', { port: actualPort })
  const t = DEFAULT_TIERED_CONFIG
  process.stderr.write(
    `[proxy] Listening on :${actualPort} — LR + tiered `
    + `(soft=mean${t.soft.n}@${t.soft.threshold} `
    + `medium=median${t.medium.n}@${t.medium.threshold} `
    + `hard=median${t.hard.n}@${t.hard.threshold})\n`,
  )
})
