#!/usr/bin/env node
/**
 * End-to-end HTTP smoke test for the context-hygiene proxy.
 *
 * Spins up:
 *   - a fake upstream server that always returns a canned 200
 *   - the real proxy with PROXY_UPSTREAM pointed at the fake and
 *     COMPACT_ENABLED=1
 * Then sends a /v1/messages POST with a stale 200-line Bash tool_result
 * and checks that the proxy:
 *   - returns the canned 200
 *   - truncates the stale Bash block before forwarding upstream
 *   - emits a `compact` event to LOG_DIR
 *
 * Exits non-zero on failure. Run via: node proxy/test/smoke_http.mjs
 */
import { createServer } from 'node:http'
import { spawn } from 'node:child_process'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const REPO = resolve(__dirname, '..', '..')

let fakeUpstream = null
let proxy = null
let proxyOut = ''

function fail(msg) {
  console.error('SMOKE FAIL:', msg)
  if (proxy) proxy.kill()
  if (fakeUpstream) fakeUpstream.close()
  process.exit(1)
}

async function main() {
  // ── Fake upstream ─────────────────────────────────────────────────────
  const upstreamReceived = []
  fakeUpstream = createServer((req, res) => {
    const chunks = []
    req.on('data', (c) => chunks.push(c))
    req.on('end', () => {
      upstreamReceived.push({
        method: req.method,
        url: req.url,
        body: Buffer.concat(chunks).toString('utf8'),
      })
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify({
        id: 'msg_smoke',
        type: 'message',
        role: 'assistant',
        content: [{ type: 'text', text: 'hello' }],
        stop_reason: 'end_turn',
        usage: {
          input_tokens: 42,
          cache_creation_input_tokens: 10,
          cache_read_input_tokens: 500,
          output_tokens: 5,
        },
      }))
    })
  })
  await new Promise((r) => fakeUpstream.listen(0, '127.0.0.1', r))
  const upstreamPort = fakeUpstream.address().port

  // ── Proxy with both primitives on ─────────────────────────────────────
  const LOG_DIR = '/tmp/compact_proxy_smoke_logs'
  proxy = spawn('node', ['proxy/proxy.mjs'], {
    cwd: REPO,
    env: {
      ...process.env,
      PROXY_PORT: '0',
      PROXY_UPSTREAM: `http://127.0.0.1:${upstreamPort}`,
      LOG_DIR,
      COMPACT_ENABLED: '1',
      INJECT_CLEAR_TOOL_USES: '1',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  let proxyPort = null
  await new Promise((resolveReady) => {
    const timer = setTimeout(() => {
      fail(`proxy did not announce listening port within 5s\nstderr:\n${proxyOut}`)
    }, 5000)
    proxy.stderr.on('data', (chunk) => {
      proxyOut += chunk.toString('utf8')
      const m = proxyOut.match(/Listening on :(\d+)/)
      if (m) {
        proxyPort = parseInt(m[1], 10)
        clearTimeout(timer)
        resolveReady()
      }
    })
    proxy.stdout.on('data', (chunk) => {
      proxyOut += '[stdout] ' + chunk.toString('utf8')
    })
    proxy.on('error', (e) => fail(`proxy spawn error: ${e.message}`))
    proxy.on('exit', (code) => {
      if (proxyPort === null) fail(`proxy exited before announcing (code=${code})\n${proxyOut}`)
    })
  })

  // ── Synthetic request: one stale 200-line Bash result + trailing turns ─
  const bigOutput = Array.from({ length: 200 }, (_, i) => `line ${i}`).join('\n')
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'build the project' }] },
    {
      role: 'assistant',
      content: [{ type: 'tool_use', id: 'smoke_1', name: 'Bash', input: { command: 'make' } }],
    },
    {
      role: 'user',
      content: [{
        type: 'tool_result',
        tool_use_id: 'smoke_1',
        content: [{ type: 'text', text: bigOutput }],
      }],
    },
    // Two trailing turns push the bash turn past the default staleTurns=2
    { role: 'assistant', content: [{ type: 'text', text: 'looking at output' }] },
    { role: 'user', content: [{ type: 'text', text: 'continue' }] },
    { role: 'assistant', content: [{ type: 'text', text: 'next step' }] },
    { role: 'user', content: [{ type: 'text', text: 'keep going' }] },
  ]
  const body = JSON.stringify({
    model: 'claude-sonnet-4-6',
    max_tokens: 1024,
    messages,
  })

  const res = await fetch(`http://127.0.0.1:${proxyPort}/v1/messages`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body,
  })
  if (res.status !== 200) fail(`proxy returned status ${res.status}`)
  const respBody = await res.text()
  if (!respBody.includes('msg_smoke')) fail(`response missing canned payload: ${respBody}`)

  // ── Verify upstream received the compacted body ───────────────────────
  if (upstreamReceived.length !== 1) {
    fail(`expected exactly 1 upstream request, got ${upstreamReceived.length}`)
  }
  const fwd = JSON.parse(upstreamReceived[0].body)
  if (!Array.isArray(fwd.messages)) fail('forwarded body has no messages array')
  const toolResult = fwd.messages
    .flatMap((m) => (Array.isArray(m.content) ? m.content : []))
    .find((b) => b?.type === 'tool_result')
  if (!toolResult) fail('forwarded body has no tool_result block')
  const text = typeof toolResult.content === 'string'
    ? toolResult.content
    : toolResult.content?.[0]?.text ?? ''
  if (!text.startsWith('[COMPACTED')) {
    fail(`tool_result was not compacted: ${text.slice(0, 80)}`)
  }
  if (text.length >= bigOutput.length) {
    fail(`compacted body not smaller (orig=${bigOutput.length} got=${text.length})`)
  }

  // ── Verify clear_tool_uses strategy was injected ──────────────────────
  const edits = fwd?.context_management?.edits ?? []
  const tu = edits.find((e) => e?.type === 'clear_tool_uses_20250919')
  if (!tu) fail(`clear_tool_uses strategy not injected; edits=${JSON.stringify(edits)}`)
  if (!Array.isArray(tu.exclude_tools) || !tu.exclude_tools.includes('Read')) {
    fail(`injected edit missing Read exclusion: ${JSON.stringify(tu)}`)
  }

  // ── Verify proxy logged a compact event ───────────────────────────────
  await new Promise((r) => setTimeout(r, 200))
  const fs = await import('node:fs/promises')
  const logEntries = await fs.readdir(LOG_DIR).catch(() => [])
  const evFile = logEntries.find((f) => f.startsWith('events-'))
  if (!evFile) fail(`no events log under ${LOG_DIR} (found: ${logEntries.join(',')})`)
  const log = await fs.readFile(`${LOG_DIR}/${evFile}`, 'utf8')
  const types = new Set()
  let compactEv = null
  let cacheStatsEv = null
  let injectEv = null
  for (const line of log.split('\n').filter(Boolean)) {
    try {
      const ev = JSON.parse(line)
      types.add(ev.type)
      if (ev.type === 'compact') compactEv = ev
      if (ev.type === 'cache_stats') cacheStatsEv = ev
      if (ev.type === 'inject_clear_tool_uses') injectEv = ev
    } catch { /* ignore */ }
  }
  if (!types.has('proxy_start')) fail('no proxy_start log entry')
  if (!types.has('compact')) fail(`no compact log entry (types: ${[...types].join(',')})`)
  if (compactEv.originalLines !== 200) {
    fail(`compact event originalLines=${compactEv.originalLines}, expected 200`)
  }
  if (!cacheStatsEv) fail(`no cache_stats log entry (types: ${[...types].join(',')})`)
  if (cacheStatsEv.input_tokens !== 42) fail(`cache_stats.input_tokens=${cacheStatsEv.input_tokens}, expected 42`)
  if (cacheStatsEv.cache_read_input_tokens !== 500) {
    fail(`cache_stats.cache_read_input_tokens=${cacheStatsEv.cache_read_input_tokens}, expected 500`)
  }
  if (!injectEv) fail(`no inject_clear_tool_uses log entry`)
  if (injectEv.edit_count_after !== injectEv.edit_count_before + 1) {
    fail(`inject event should add exactly 1 edit, got ${injectEv.edit_count_after - injectEv.edit_count_before}`)
  }

  console.log('SMOKE PASS')
  console.log(`  proxy port: ${proxyPort}`)
  console.log(`  upstream forwarded ${fwd.messages.length} messages`)
  console.log(`  tool_result: ${bigOutput.length} → ${text.length} chars`)
  console.log(`  log types: ${[...types].sort().join(', ')}`)

  proxy.kill()
  fakeUpstream.close()
}

main().catch((e) => fail(e.stack || e.message))
