#!/usr/bin/env node
/**
 * End-to-end smoke for REWIND_ENABLED.
 *
 * Sends a synthetic /v1/messages POST whose history contains a
 * `checkpoint_progress` tool_use. Verifies the proxy:
 *   - elides the earlier turns on the way upstream,
 *   - splices in a synthetic user summary message,
 *   - emits a `rewind_applied` event.
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
  const upstreamReceived = []
  fakeUpstream = createServer((req, res) => {
    const chunks = []
    req.on('data', (c) => chunks.push(c))
    req.on('end', () => {
      upstreamReceived.push({ body: Buffer.concat(chunks).toString('utf8') })
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify({
        id: 'msg_smoke',
        type: 'message',
        role: 'assistant',
        content: [{ type: 'text', text: 'hi' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 1 },
      }))
    })
  })
  await new Promise((r) => fakeUpstream.listen(0, '127.0.0.1', r))
  const upstreamPort = fakeUpstream.address().port

  const LOG_DIR = '/tmp/rewind_proxy_smoke_logs'
  proxy = spawn('node', ['proxy/proxy.mjs'], {
    cwd: REPO,
    env: {
      ...process.env,
      PROXY_PORT: '0',
      PROXY_UPSTREAM: `http://127.0.0.1:${upstreamPort}`,
      LOG_DIR,
      REWIND_ENABLED: '1',
      COMPACT_ENABLED: '0',
      INJECT_CLEAR_TOOL_USES: '0',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  let proxyPort = null
  await new Promise((done) => {
    const timer = setTimeout(() => {
      fail(`proxy did not announce listening port within 5s\n${proxyOut}`)
    }, 5000)
    proxy.stderr.on('data', (c) => {
      proxyOut += c.toString('utf8')
      const m = proxyOut.match(/Listening on :(\d+)/)
      if (m) { proxyPort = parseInt(m[1], 10); clearTimeout(timer); done() }
    })
    proxy.stdout.on('data', (c) => (proxyOut += '[stdout] ' + c.toString('utf8')))
    proxy.on('exit', (code) => {
      if (proxyPort == null) fail(`proxy exited early (code=${code})\n${proxyOut}`)
    })
  })

  // ── Synthetic history: prompt + 2 failed tool calls + summarize + 1 more ──
  const messages = [
    { role: 'user', content: [{ type: 'text', text: 'fix the bug' }] },
    { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'Bash', input: { command: 'make' } }] },
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't1', content: [{ type: 'text', text: 'fail' }] }] },
    { role: 'assistant', content: [{ type: 'tool_use', id: 't2', name: 'Bash', input: { command: 'cmake' } }] },
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't2', content: [{ type: 'text', text: 'fail2' }] }] },
    {
      role: 'assistant',
      content: [{
        type: 'tool_use', id: 's1',
        name: 'mcp__bookmarks__checkpoint_progress',
        input: { summary: 'Build is broken, not my bug path. Focus on source changes.' },
      }],
    },
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: 's1', content: [{ type: 'text', text: 'ack' }] }] },
    { role: 'assistant', content: [{ type: 'tool_use', id: 't3', name: 'Read', input: { path: 'main.c' } }] },
    { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't3', content: [{ type: 'text', text: 'src' }] }] },
  ]

  const res = await fetch(`http://127.0.0.1:${proxyPort}/v1/messages`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'claude-opus-4-7', max_tokens: 1024, messages }),
  })
  if (res.status !== 200) fail(`proxy returned status ${res.status}`)

  // ── Upstream body should be shorter: elided 2 Bash pairs, synth msg added ──
  if (upstreamReceived.length !== 1) fail(`expected 1 upstream req, got ${upstreamReceived.length}`)
  const fwd = JSON.parse(upstreamReceived[0].body)
  if (!Array.isArray(fwd.messages)) fail('forwarded body missing messages')

  // Expected shape: [prompt, synth_summary, summarize_use, summarize_result, read_use, read_result]
  if (fwd.messages.length !== 6) {
    fail(`expected 6 forwarded messages (prompt + synth + summarize-pair + read-pair), got ${fwd.messages.length}`)
  }

  const synth = fwd.messages[1]
  if (synth.role !== 'user' || !synth.content?.[0]?.text?.includes('Build is broken')) {
    fail(`synthetic summary not spliced in correctly: ${JSON.stringify(synth)}`)
  }

  // Pairing preserved
  const useIds = new Set()
  const resultIds = new Set()
  for (const m of fwd.messages) {
    if (!Array.isArray(m.content)) continue
    for (const b of m.content) {
      if (b.type === 'tool_use') useIds.add(b.id)
      if (b.type === 'tool_result') resultIds.add(b.tool_use_id)
    }
  }
  for (const id of useIds) if (!resultIds.has(id)) fail(`orphan tool_use: ${id}`)

  // ── Log check ──
  await new Promise((r) => setTimeout(r, 200))
  const fs = await import('node:fs/promises')
  const entries = await fs.readdir(LOG_DIR).catch(() => [])
  const evFile = entries.find((f) => f.startsWith('events-'))
  if (!evFile) fail(`no events log in ${LOG_DIR}`)
  const log = await fs.readFile(`${LOG_DIR}/${evFile}`, 'utf8')
  let rewind = null
  for (const line of log.split('\n').filter(Boolean)) {
    try {
      const ev = JSON.parse(line)
      if (ev.type === 'rewind_applied') rewind = ev
    } catch { /* ignore */ }
  }
  if (!rewind) fail(`no rewind_applied event`)
  if (rewind.turnsElided !== 4) fail(`turnsElided=${rewind.turnsElided}, expected 4`)
  if (!rewind.summary.includes('Build is broken')) fail(`summary missing in rewind event`)

  console.log('SMOKE PASS')
  console.log(`  forwarded ${fwd.messages.length} messages (was ${messages.length})`)
  console.log(`  elided ${rewind.turnsElided} turns, ${rewind.bytesSaved} bytes saved`)

  proxy.kill()
  fakeUpstream.close()
}

main().catch((e) => fail(e.stack || e.message))
