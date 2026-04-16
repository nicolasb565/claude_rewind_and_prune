#!/usr/bin/env node
/**
 * End-to-end HTTP smoke test for the LR proxy.
 *
 * Spins up:
 *   - a fake upstream server that always returns a canned 200
 *   - the real proxy with PROXY_UPSTREAM pointed at the fake
 * Then sends a synthetic /v1/messages POST containing a tiny tool-call
 * history and checks:
 *   - the proxy returned 200
 *   - the proxy logged an lr_score event
 *   - the proxy did NOT crash on the LR + filter + nudge wiring
 *
 * Exits non-zero on any failure; intended to be run via:
 *   node proxy/test/smoke_http.mjs
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
  // ── Fake upstream that returns a canned 200 ───────────────────────────
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
      }))
    })
  })
  await new Promise((r) => fakeUpstream.listen(0, '127.0.0.1', r))
  const upstreamPort = fakeUpstream.address().port

  // ── Spawn the proxy with PROXY_UPSTREAM = fake ────────────────────────
  proxy = spawn('node', ['proxy/proxy.mjs'], {
    cwd: REPO,
    env: {
      ...process.env,
      PROXY_PORT: '0', // 0 means assign random port
      PROXY_UPSTREAM: `http://127.0.0.1:${upstreamPort}`,
      LOG_DIR: '/tmp/lr_proxy_smoke_logs',
      STUCK_ENABLED: '1',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  let proxyPort = null
  await new Promise((resolveReady) => {
    let timer = setTimeout(() => {
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

  // ── Synthetic request body: 5 repeated stuck-shaped tool calls ────────
  // The repetition will drive the LR signal up so we can verify the
  // pipeline wiring works (NOT necessarily that the nudge fires — that's
  // covered by the unit tests).
  const messages = []
  for (let i = 0; i < 6; i++) {
    messages.push({
      role: 'assistant',
      content: [{
        type: 'tool_use',
        id: `tool_${i}`,
        name: 'Bash',
        input: { command: 'gcc -O2 -c src/main.c' },
      }],
    })
    messages.push({
      role: 'user',
      content: [{
        type: 'tool_result',
        tool_use_id: `tool_${i}`,
        content: [{
          type: 'text',
          text: 'src/main.c:42: error: undefined reference to foo',
        }],
      }],
    })
  }
  // Add a final user-text turn so the request looks like a real Claude
  // Code request shape
  messages.unshift({ role: 'user', content: [{ type: 'text', text: 'fix the build' }] })

  const body = JSON.stringify({
    model: 'claude-sonnet-4-6',
    max_tokens: 1024,
    messages,
  })

  // ── Send the request to the proxy ─────────────────────────────────────
  const res = await fetch(`http://127.0.0.1:${proxyPort}/v1/messages`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body,
  })
  if (res.status !== 200) fail(`proxy returned status ${res.status}`)
  const respBody = await res.text()
  if (!respBody.includes('msg_smoke')) {
    fail(`response did not include canned upstream payload, got: ${respBody}`)
  }

  // ── Verify upstream got the (possibly nudge-augmented) request ────────
  if (upstreamReceived.length !== 1) {
    fail(`expected exactly 1 upstream request, got ${upstreamReceived.length}`)
  }
  const fwd = JSON.parse(upstreamReceived[0].body)
  if (!Array.isArray(fwd.messages)) fail('forwarded body has no messages array')
  // The proxy MAY have appended a nudge user message — accept ≥ original.
  if (fwd.messages.length < messages.length) {
    fail(`forwarded ${fwd.messages.length} messages, expected ≥ ${messages.length}`)
  }

  // ── Verify proxy logged an lr_score event for this request ────────────
  // Wait briefly for the log to flush
  await new Promise((r) => setTimeout(r, 200))
  const fs = await import('node:fs/promises')
  const logEntries = await fs.readdir('/tmp/lr_proxy_smoke_logs').catch(() => [])
  const evFile = logEntries.find((f) => f.startsWith('events-'))
  if (!evFile) fail(`no events log under /tmp/lr_proxy_smoke_logs (found: ${logEntries.join(',')})`)
  const log = await fs.readFile(`/tmp/lr_proxy_smoke_logs/${evFile}`, 'utf8')
  const lines = log.split('\n').filter(Boolean)
  const types = new Set()
  let lastLrScore = null
  for (const line of lines) {
    try {
      const ev = JSON.parse(line)
      types.add(ev.type)
      if (ev.type === 'lr_score') lastLrScore = ev
    } catch {
      // ignore malformed lines
    }
  }
  if (!types.has('proxy_start')) fail('no proxy_start log entry')
  if (!types.has('lr_score')) fail(`no lr_score log entry (saw types: ${[...types].join(',')})`)
  if (lastLrScore.score == null) fail(`lr_score has no score field: ${JSON.stringify(lastLrScore)}`)
  if (lastLrScore.score < 0 || lastLrScore.score > 1) {
    fail(`lr_score out of [0,1]: ${lastLrScore.score}`)
  }

  console.log('SMOKE PASS')
  console.log(`  proxy port: ${proxyPort}`)
  console.log(`  upstream received: 1 request, forwarded ${fwd.messages.length} messages`)
  console.log(`  log types: ${[...types].sort().join(', ')}`)
  console.log(`  last lr_score: score=${lastLrScore.score} `
    + `medium=${lastLrScore.medium} hard=${lastLrScore.hard}`)
  console.log(`  filter aggregates: med4=${lastLrScore.med4} med9=${lastLrScore.med9}`)

  proxy.kill()
  fakeUpstream.close()
}

main().catch((e) => fail(e.stack || e.message))
