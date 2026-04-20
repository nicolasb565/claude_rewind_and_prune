import { describe, test, beforeEach } from 'node:test'
import assert from 'node:assert/strict'
import {
  parseShadow,
  applyShadowRewind,
  deriveSessionKey,
  shadowAndMaybeRewind,
  callShadowWithTimeout,
  __resetSessions,
} from '../shadow.mjs'

// ── helpers ───────────────────────────────────────────────────────────────

function userText(text) {
  return { role: 'user', content: [{ type: 'text', text }] }
}
function assistantText(text) {
  return { role: 'assistant', content: [{ type: 'text', text }] }
}
function assistantToolUse(name, id, input = {}) {
  return { role: 'assistant', content: [{ type: 'tool_use', id, name, input }] }
}
function userToolResult(toolUseId, text) {
  return {
    role: 'user',
    content: [{ type: 'tool_result', tool_use_id: toolUseId, content: [{ type: 'text', text }] }],
  }
}

beforeEach(() => __resetSessions())

// ── parseShadow ───────────────────────────────────────────────────────────

describe('parseShadow', () => {
  test('parses YES with all fields', () => {
    const text = [
      'SHOULD_CHECKPOINT: YES',
      'REASON: Test just passed after edit',
      'SUMMARY: Fixed src/foo.py::bar to handle empty input. tests/test_foo.py now passes.',
    ].join('\n')
    const r = parseShadow(text)
    assert.equal(r.should, 'YES')
    assert.equal(r.reason, 'Test just passed after edit')
    assert.match(r.summary, /^Fixed src\/foo\.py::bar/)
  })

  test('parses NO', () => {
    const r = parseShadow('SHOULD_CHECKPOINT: NO\nREASON: still exploring\nSUMMARY: n/a')
    assert.equal(r.should, 'NO')
    assert.equal(r.reason, 'still exploring')
  })

  test('case-insensitive YES/NO', () => {
    assert.equal(parseShadow('SHOULD_CHECKPOINT: yes').should, 'YES')
    assert.equal(parseShadow('SHOULD_CHECKPOINT: no').should, 'NO')
  })

  test('empty/malformed input', () => {
    assert.deepEqual(parseShadow(''), { should: null, reason: null, summary: null, raw: '' })
    assert.deepEqual(parseShadow(null), { should: null, reason: null, summary: null, raw: '' })
    const r = parseShadow('garbage text with nothing structured')
    assert.equal(r.should, null)
  })

  test('SUMMARY captures multi-line content', () => {
    const text = [
      'SHOULD_CHECKPOINT: YES',
      'REASON: ok',
      'SUMMARY: line one',
      'line two of summary',
    ].join('\n')
    const r = parseShadow(text)
    assert.match(r.summary, /line one/)
    assert.match(r.summary, /line two/)
  })
})

// ── applyShadowRewind ─────────────────────────────────────────────────────

describe('applyShadowRewind', () => {
  test('no-op when messages too short', () => {
    const msgs = [userText('goal'), assistantText('hi'), userText('ok')]
    const r = applyShadowRewind(msgs, 'summary')
    assert.equal(r.elided, 0)
    assert.equal(r.messages, msgs)
  })

  test('no-op when summary missing', () => {
    const msgs = [userText('goal'), assistantToolUse('Bash', 'a1'), userToolResult('a1', 'out'),
      assistantToolUse('Read', 'a2'), userToolResult('a2', 'file contents')]
    const r = applyShadowRewind(msgs, '')
    assert.equal(r.elided, 0)
  })

  test('preserves goal + most-recent pair, splices summary', () => {
    const msgs = [
      userText('Fix the failing tests'),                                 // 0 goal
      assistantToolUse('Bash', 'a1', { command: 'pytest' }),             // 1
      userToolResult('a1', 'FAIL'),                                      // 2
      assistantToolUse('Read', 'a2', { file_path: 'x.py' }),             // 3
      userToolResult('a2', 'def x():'),                                  // 4
      assistantToolUse('Edit', 'a3'),                                    // 5 ← most recent assistant
      userToolResult('a3', 'Wrote x.py'),                                // 6
    ]
    const { messages: out, elided } = applyShadowRewind(msgs, 'Fixed x.py::x')
    // Expected: [goal, synthetic, a3, tool_result(a3)]
    assert.equal(out.length, 4)
    assert.equal(out[0], msgs[0], 'goal preserved by ref')
    assert.equal(out[1].role, 'user')
    assert.match(out[1].content[0].text, /Fixed x\.py::x/)
    assert.match(out[1].content[0].text, /Checkpoint note/)
    assert.equal(out[2], msgs[5], 'most-recent assistant preserved by ref')
    assert.equal(out[3], msgs[6], 'most-recent tool_result preserved by ref')
    // 4 original messages dropped (idx 1..4); synthetic added.
    assert.equal(elided, 4)
  })

  test('no-op when there is no tool_result (no concrete evidence)', () => {
    const msgs = [userText('hi'), assistantText('replied'), userText('x'), assistantText('y')]
    const r = applyShadowRewind(msgs, 'summary')
    assert.equal(r.elided, 0)
  })

  test('no-op when recent pair is already adjacent to goal', () => {
    const msgs = [
      userText('goal'),
      assistantToolUse('Bash', 'a1'),
      userToolResult('a1', 'done'),
      assistantText('wrap-up text'),
    ]
    const r = applyShadowRewind(msgs, 'summary')
    // Most recent tool_result is idx 2; paired assistant is idx 1; goal is 0.
    // pairedAssistantIdx=1 → no elision possible, skip.
    assert.equal(r.elided, 0)
  })

  test('no-op when tool_use ids do not match tool_result ids (invalid pair)', () => {
    // Multi-block assistant with 2 tool_use, user returns only 1 tool_result.
    // This SHOULD be rejected (would create an invalid API request).
    const msgs = [
      userText('goal'),
      assistantToolUse('Bash', 'a1'),
      userToolResult('a1', 'out'),
      {
        role: 'assistant',
        content: [
          { type: 'tool_use', id: 'a2', name: 'Read', input: { file_path: 'x' } },
          { type: 'tool_use', id: 'a3', name: 'Read', input: { file_path: 'y' } },
        ],
      },
      // Only one matching tool_result — a3 is orphaned.
      userToolResult('a2', 'contents of x'),
    ]
    const r = applyShadowRewind(msgs, 'summary')
    assert.equal(r.elided, 0, 'should bail out when pairing invariant would break')
  })

  test('accepts a valid multi-tool_use pair', () => {
    const msgs = [
      userText('goal'),
      assistantToolUse('Bash', 'a1'),
      userToolResult('a1', 'out'),
      assistantToolUse('Read', 'a2'),
      userToolResult('a2', 'filler'),
      {
        role: 'assistant',
        content: [
          { type: 'tool_use', id: 'b1', name: 'Read', input: {} },
          { type: 'tool_use', id: 'b2', name: 'Grep', input: {} },
        ],
      },
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'b1', content: 'r1' },
          { type: 'tool_result', tool_use_id: 'b2', content: 'r2' },
        ],
      },
    ]
    const r = applyShadowRewind(msgs, 'ok summary')
    assert.ok(r.elided > 0, 'should elide the pre-pair messages')
    // Preserved tail: [multi-tool_use assistant, matching user]
    assert.equal(r.messages[r.messages.length - 1].content.length, 2)
  })

  test('still works when messages end with a trailing assistant text after last tool_result', () => {
    const msgs = [
      userText('goal'),
      assistantToolUse('Bash', 'a1'),
      userToolResult('a1', 'FAIL'),
      assistantToolUse('Read', 'a2'),
      userToolResult('a2', 'file contents'),    // last tool_result (idx 4)
      assistantText('thinking about next step'), // trailing assistant (idx 5)
    ]
    const { messages: out, elided } = applyShadowRewind(msgs, 'Fixed x.py::bar')
    // Preserve goal + paired assistant (idx 3) + tool_result (idx 4) + trailing (idx 5).
    // Elide idx 1,2.
    assert.equal(elided, 2)
    assert.equal(out[0], msgs[0])
    assert.equal(out[2], msgs[3])
    assert.equal(out[3], msgs[4])
    assert.equal(out[4], msgs[5])
  })
})

// ── deriveSessionKey ──────────────────────────────────────────────────────

describe('deriveSessionKey', () => {
  test('stable across calls on same goal', () => {
    const a = deriveSessionKey([userText('fix the bug')])
    const b = deriveSessionKey([userText('fix the bug')])
    assert.equal(a, b)
  })

  test('different goals produce different keys', () => {
    const a = deriveSessionKey([userText('task A')])
    const b = deriveSessionKey([userText('task B')])
    assert.notEqual(a, b)
  })

  test('handles string content', () => {
    const key = deriveSessionKey([{ role: 'user', content: 'plain string goal' }])
    assert.equal(typeof key, 'string')
    assert.ok(key.length > 0)
  })

  test('returns sentinel for empty/malformed', () => {
    assert.equal(deriveSessionKey([]), 'empty')
    assert.equal(deriveSessionKey([{ role: 'assistant', content: [] }]), 'nongoal')
  })
})

// ── session map eviction ──────────────────────────────────────────────────

describe('session map eviction', () => {
  test('evicts oldest entries when size exceeds SHADOW_MAX_SESSIONS', async () => {
    // The default cap is 1000; we can drive overflow by firing many
    // distinct sessions. Use a stubbed shadow that always returns NO
    // so no rewind fires but state entries are still created.
    const shadow = makeFakeShadow({ should: 'NO', summary: null })
    for (let i = 0; i < 1100; i++) {
      await shadowAndMaybeRewind({
        messages: buildSession({ turns: 3 }).map((m, idx) =>
          idx === 0 ? userText(`goal #${i}`) : m),
        upstream: 'x', clientHeaders: {}, log: () => {}, callShadowFn: shadow,
      })
    }
    // Can't inspect map size directly from here without exposing more
    // internals — but if eviction is broken we'd leak memory. A
    // minimal assertion: the mechanism didn't throw, and a fresh
    // invocation after overflow still works.
    const r = await shadowAndMaybeRewind({
      messages: buildSession({ turns: 3 }).map((m, idx) =>
        idx === 0 ? userText('post-eviction goal') : m),
      upstream: 'x', clientHeaders: {}, log: () => {}, callShadowFn: shadow,
    })
    assert.equal(r.applied, false)
    assert.equal(r.reason, 'shadow_no')
  })
})

// ── shadowAndMaybeRewind (with injected shadow) ───────────────────────────

function makeFakeShadow(parsed, usage = { input_tokens: 100, output_tokens: 50 }) {
  return async () => ({ parsed, usage })
}

function buildSession({ turns = 3 } = {}) {
  const msgs = [userText('Debug the failing test tests/test_x.py')]
  for (let i = 0; i < turns; i++) {
    msgs.push(assistantToolUse('Bash', `a${i}`, { command: `echo ${i}` }))
    msgs.push(userToolResult(`a${i}`, `output ${i}`))
  }
  return msgs
}

describe('callShadowWithTimeout', () => {
  test('resolves when call completes in time', async () => {
    const result = await callShadowWithTimeout(
      { callShadowFn: async () => ({ parsed: { should: 'NO' }, usage: null }) },
      1000,
    )
    assert.deepEqual(result, { parsed: { should: 'NO' }, usage: null })
  })

  test('rejects with timeout error when call hangs', async () => {
    // Never-resolving promise simulates a hung upstream.
    const neverResolves = () => new Promise(() => {})
    await assert.rejects(
      () => callShadowWithTimeout({ callShadowFn: neverResolves }, 50),
      /shadow timeout after 50ms/,
    )
  })

  test('clears timer when call wins the race', async () => {
    // Smoke-level check: many calls with short timeouts shouldn't
    // accumulate timers (node would keep process alive if they did).
    for (let i = 0; i < 50; i++) {
      await callShadowWithTimeout(
        { callShadowFn: async () => ({ parsed: null, usage: null }) },
        100,
      )
    }
    // Just completing without exit-timer leaks is the assertion.
  })
})

describe('shadowAndMaybeRewind', () => {
  const noLog = () => {}

  test('skips when messages[] is too short', async () => {
    const msgs = [userText('goal'), assistantText('hi')]
    const r = await shadowAndMaybeRewind({
      messages: msgs,
      upstream: 'https://api.test',
      clientHeaders: {},
      log: noLog,
      callShadowFn: makeFakeShadow({ should: 'YES', summary: 'x' }),
    })
    assert.equal(r.applied, false)
    assert.equal(r.reason, 'too_short')
  })

  test('fires rewind on YES + summary', async () => {
    const msgs = buildSession({ turns: 3 }) // 7 messages
    const shadow = makeFakeShadow({
      should: 'YES',
      reason: 'test passed',
      summary: 'Fixed src/foo.py::bar',
    })
    const r = await shadowAndMaybeRewind({
      messages: msgs,
      upstream: 'https://api.test',
      clientHeaders: {},
      log: noLog,
      callShadowFn: shadow,
    })
    assert.equal(r.applied, true)
    assert.equal(r.reason, 'fired')
    assert.ok(r.elided > 0)
    assert.match(r.summary, /Fixed src\/foo\.py::bar/)
    assert.ok(r.messages.length < msgs.length)
  })

  test('does not fire on NO', async () => {
    const msgs = buildSession({ turns: 3 })
    const shadow = makeFakeShadow({ should: 'NO', reason: 'still working', summary: null })
    const r = await shadowAndMaybeRewind({
      messages: msgs, upstream: 'x', clientHeaders: {}, log: noLog, callShadowFn: shadow,
    })
    assert.equal(r.applied, false)
    assert.equal(r.reason, 'shadow_no')
  })

  test('cooldown prevents second fire within window', async () => {
    const msgs = buildSession({ turns: 3 })
    const shadow = makeFakeShadow({ should: 'YES', summary: 'ok' })
    const first = await shadowAndMaybeRewind({
      messages: msgs, upstream: 'x', clientHeaders: {}, log: noLog, callShadowFn: shadow,
    })
    assert.equal(first.applied, true)

    // Next call with same session key should be cooldown'd
    const second = await shadowAndMaybeRewind({
      messages: msgs, upstream: 'x', clientHeaders: {}, log: noLog, callShadowFn: shadow,
    })
    assert.equal(second.applied, false)
    assert.equal(second.reason, 'cooldown')
  })

  test('fail-open on shadow error', async () => {
    const msgs = buildSession({ turns: 3 })
    const shadow = async () => { throw new Error('boom') }
    const r = await shadowAndMaybeRewind({
      messages: msgs, upstream: 'x', clientHeaders: {}, log: noLog, callShadowFn: shadow,
    })
    assert.equal(r.applied, false)
    assert.equal(r.reason, 'shadow_error')
    assert.equal(r.messages, msgs, 'original messages returned unchanged')
  })

  test('timeout-shaped error from shadow is fail-open', async () => {
    const msgs = buildSession({ turns: 3 })
    const shadow = async () => { throw new Error('shadow timeout after 100ms') }
    const r = await shadowAndMaybeRewind({
      messages: msgs, upstream: 'x', clientHeaders: {}, log: noLog, callShadowFn: shadow,
    })
    assert.equal(r.applied, false)
    assert.equal(r.reason, 'shadow_error')
    assert.equal(r.messages, msgs)
  })

  test('malformed shadow response (missing content array) does not throw', async () => {
    const msgs = buildSession({ turns: 3 })
    // parsed.should=null, parsed.summary=null
    const shadow = makeFakeShadow({ should: null, summary: null, reason: null, raw: '' })
    const r = await shadowAndMaybeRewind({
      messages: msgs, upstream: 'x', clientHeaders: {}, log: noLog, callShadowFn: shadow,
    })
    assert.equal(r.applied, false)
    assert.equal(r.reason, 'shadow_no')
  })

  test('handles YES with missing summary as no-fire', async () => {
    const msgs = buildSession({ turns: 3 })
    const shadow = makeFakeShadow({ should: 'YES', summary: null })
    const r = await shadowAndMaybeRewind({
      messages: msgs, upstream: 'x', clientHeaders: {}, log: noLog, callShadowFn: shadow,
    })
    assert.equal(r.applied, false)
    assert.equal(r.reason, 'no_summary')
  })
})
