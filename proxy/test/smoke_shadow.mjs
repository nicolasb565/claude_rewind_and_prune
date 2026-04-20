#!/usr/bin/env node
/**
 * End-to-end smoke test for the shadow middleware: stand up a fake
 * upstream that answers /v1/messages with a canned shadow response,
 * point the proxy at it, POST a request through the proxy, and verify:
 *   - The shadow call was made (visible in the fake upstream's request
 *     log) with the SHADOW_QUERY as the system prompt and a different
 *     model than the agent request.
 *   - When shadow returns YES + summary, the agent request forwarded
 *     upstream has a rewritten messages[] with the synthetic summary
 *     message present and the old exploration dropped.
 *   - When shadow returns NO, the agent request is forwarded unchanged.
 *
 * Run:
 *   node proxy/test/smoke_shadow.mjs
 */

import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { createServer } from 'node:http'
import assert from 'node:assert/strict'

const AGENT_MODEL = 'claude-sonnet-4-6'
const SHADOW_MODEL = 'claude-haiku-4-5-20251001'

function upstreamResponseText(text) {
  // Non-streaming JSON response body like the real API returns.
  return JSON.stringify({
    id: 'msg_test',
    type: 'message',
    role: 'assistant',
    model: SHADOW_MODEL,
    content: [{ type: 'text', text }],
    stop_reason: 'end_turn',
    usage: { input_tokens: 100, output_tokens: 50 },
  })
}

function agentResponseText(text) {
  return JSON.stringify({
    id: 'msg_agent',
    type: 'message',
    role: 'assistant',
    model: AGENT_MODEL,
    content: [{ type: 'text', text }],
    stop_reason: 'end_turn',
    usage: { input_tokens: 200, output_tokens: 30 },
  })
}

// Fake upstream. Routes the shadow call (identified by the SHADOW_QUERY
// in `system`) to `shadowResponseText`, everything else to the agent
// stub. Records all requests for assertion.
async function startFakeUpstream({ shadowResponseText }) {
  const received = []
  const server = createServer(async (req, res) => {
    const chunks = []
    for await (const c of req) chunks.push(c)
    const body = Buffer.concat(chunks).toString()
    let parsed = null
    try { parsed = JSON.parse(body) } catch {}
    received.push({ url: req.url, headers: req.headers, body: parsed })

    const systemStr = extractSystemText(parsed?.system)
    const isShadow = systemStr.startsWith('You are in REFLECTION mode')

    const text = isShadow ? shadowResponseText : 'agent reply'
    const bodyOut = isShadow ? upstreamResponseText(text) : agentResponseText(text)
    res.writeHead(200, {
      'content-type': 'application/json',
      'content-length': Buffer.byteLength(bodyOut),
    })
    res.end(bodyOut)
  })
  server.listen(0)
  await once(server, 'listening')
  return {
    url: `http://127.0.0.1:${server.address().port}`,
    received,
    close: () => new Promise((r) => server.close(r)),
  }
}

async function startProxy({ upstream, env = {} }) {
  const child = spawn(
    process.execPath,
    ['proxy/proxy.mjs'],
    {
      cwd: process.cwd(),
      env: {
        ...process.env,
        PROXY_PORT: '0', // server.listen(0) picks an ephemeral port
        PROXY_UPSTREAM: upstream,
        SHADOW_ENABLED: '1',
        ...env,
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  )
  let port = null
  const stderrLines = []
  child.stderr.on('data', (d) => {
    const s = d.toString()
    stderrLines.push(s)
    const m = /Listening on :(\d+)/.exec(s)
    if (m) port = parseInt(m[1], 10)
  })

  // Wait up to 3s for the proxy to report its port.
  const deadline = Date.now() + 3000
  while (!port && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 50))
  }
  if (!port) {
    child.kill('SIGKILL')
    throw new Error('proxy did not start:\n' + stderrLines.join(''))
  }
  return {
    port,
    url: `http://127.0.0.1:${port}`,
    stop: () =>
      new Promise((resolve) => {
        child.on('exit', resolve)
        child.kill('SIGTERM')
      }),
    stderrLines,
  }
}

function buildAgentRequest() {
  return {
    model: AGENT_MODEL,
    max_tokens: 1024,
    messages: [
      { role: 'user', content: [{ type: 'text', text: 'Fix the failing test in src/foo.py' }] },
      { role: 'assistant', content: [{ type: 'tool_use', id: 'a1', name: 'Bash', input: { command: 'pytest' } }] },
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'a1', content: 'FAIL' }] },
      { role: 'assistant', content: [{ type: 'tool_use', id: 'a2', name: 'Read', input: { file_path: 'src/foo.py' } }] },
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'a2', content: 'def foo():' }] },
      { role: 'assistant', content: [{ type: 'tool_use', id: 'a3', name: 'Edit', input: { file_path: 'src/foo.py' } }] },
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'a3', content: 'Wrote src/foo.py' }] },
    ],
  }
}

async function postThroughProxy(proxyUrl, body) {
  const res = await fetch(proxyUrl + '/v1/messages', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-api-key': 'test-key',
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify(body),
  })
  const text = await res.text()
  return { status: res.status, text }
}

function extractSystemText(system) {
  if (typeof system === 'string') return system
  if (Array.isArray(system)) {
    return system.map((b) => (typeof b === 'string' ? b : b?.text ?? '')).join('\n')
  }
  return ''
}

function findShadowRequest(received) {
  return received.find((r) => extractSystemText(r.body?.system).startsWith('You are in REFLECTION mode'))
}

function findAgentRequest(received) {
  return received.find((r) => r.body?.model === AGENT_MODEL)
}

// ── scenario 1: shadow returns YES, rewind applies ────────────────────────

async function testShadowFiresRewind() {
  const shadowText = [
    'SHOULD_CHECKPOINT: YES',
    'REASON: test output confirms src/foo.py::foo now returns correct value',
    'SUMMARY: Fixed src/foo.py::foo to handle empty inputs. tests/test_foo.py now passes. No further work.',
  ].join('\n')
  const upstream = await startFakeUpstream({ shadowResponseText: shadowText })
  const proxy = await startProxy({ upstream: upstream.url })
  try {
    const body = buildAgentRequest()
    const res = await postThroughProxy(proxy.url, body)
    assert.equal(res.status, 200)

    const shadowReq = findShadowRequest(upstream.received)
    assert.ok(shadowReq, 'shadow request should have been sent')
    assert.equal(shadowReq.body.model, SHADOW_MODEL)
    assert.ok(shadowReq.body.messages.length >= 6, 'shadow should carry the session history')

    const agentReq = findAgentRequest(upstream.received)
    assert.ok(agentReq, 'agent request should have been forwarded')
    const agentMsgs = agentReq.body.messages
    // With rewind: [goal, synthetic, pairedAssistant, lastToolResult] = 4 messages.
    assert.equal(agentMsgs.length, 4, `expected 4 messages after rewind, got ${agentMsgs.length}`)
    const synthetic = agentMsgs[1]
    assert.equal(synthetic.role, 'user')
    assert.match(
      synthetic.content[0].text,
      /Checkpoint note.*Fixed src\/foo\.py::foo/s,
      'synthetic message should contain the shadow summary',
    )
    console.log('  ✓ scenario 1 passed: shadow YES → rewind applied, messages shortened from 7 → 4')
  } finally {
    await proxy.stop()
    await upstream.close()
  }
}

// ── scenario 2: shadow returns NO, no rewind ──────────────────────────────

async function testShadowNoDoesNotRewind() {
  const shadowText = [
    'SHOULD_CHECKPOINT: NO',
    'REASON: change applied but tests have not been re-run yet',
    'SUMMARY: n/a',
  ].join('\n')
  const upstream = await startFakeUpstream({ shadowResponseText: shadowText })
  const proxy = await startProxy({ upstream: upstream.url })
  try {
    const body = buildAgentRequest()
    const res = await postThroughProxy(proxy.url, body)
    assert.equal(res.status, 200)

    const agentReq = findAgentRequest(upstream.received)
    assert.ok(agentReq)
    assert.deepEqual(
      agentReq.body.messages,
      body.messages,
      'messages[] should be byte-identical on shadow NO',
    )
    console.log('  ✓ scenario 2 passed: shadow NO → messages forwarded unchanged')
  } finally {
    await proxy.stop()
    await upstream.close()
  }
}

// ── scenario 3: shadow upstream errors → fail-open ────────────────────────

async function testShadowErrorIsFailOpen() {
  // Fake upstream that returns 500 for the shadow path but success for agent.
  const received = []
  const server = createServer(async (req, res) => {
    const chunks = []
    for await (const c of req) chunks.push(c)
    const body = Buffer.concat(chunks).toString()
    let parsed = null
    try { parsed = JSON.parse(body) } catch {}
    received.push({ url: req.url, body: parsed })
    const systemStr = extractSystemText(parsed?.system)
    const isShadow = systemStr.startsWith('You are in REFLECTION mode')
    if (isShadow) {
      const err = JSON.stringify({ type: 'error', error: { message: 'simulated 500' } })
      res.writeHead(500, { 'content-type': 'application/json' })
      res.end(err)
    } else {
      const b = agentResponseText('ok')
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(b)
    }
  })
  server.listen(0)
  await once(server, 'listening')
  const upstreamUrl = `http://127.0.0.1:${server.address().port}`

  // Tighten retry settings so the 500 fails fast (default MAX_RETRIES=8
  // would add ~2 min of backoff before giving up).
  const proxy = await startProxy({
    upstream: upstreamUrl,
    env: { PROXY_MAX_RETRIES: '0', PROXY_BASE_DELAY_MS: '1' },
  })
  try {
    const body = buildAgentRequest()
    const res = await postThroughProxy(proxy.url, body)
    assert.equal(res.status, 200, 'agent request should still succeed despite shadow error')
    const agent = received.find((r) => r.body?.model === AGENT_MODEL)
    assert.ok(agent, 'agent request should have been forwarded')
    assert.equal(agent.body.messages.length, body.messages.length,
      'messages[] should be unchanged when shadow errors')
    console.log('  ✓ scenario 3 passed: shadow 500 → agent request still forwarded, unchanged')
  } finally {
    await proxy.stop()
    await new Promise((r) => server.close(r))
  }
}

async function main() {
  console.log('=== proxy/test/smoke_shadow.mjs ===')
  await testShadowFiresRewind()
  await testShadowNoDoesNotRewind()
  await testShadowErrorIsFailOpen()
  console.log('=== all smoke scenarios passed ===')
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
