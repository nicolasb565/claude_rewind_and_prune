import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { compact } from '../compact.mjs'

// ── helpers ───────────────────────────────────────────────────────────────────

/** Generate N lines of bash-looking output. */
function lines(n, prefix = 'line') {
  return Array.from({ length: n }, (_, i) => `${prefix} ${i}`).join('\n')
}

/**
 * Build a conversation with N bash turns, each producing `output`. An extra
 * trailing pair of assistant+user turns (no tool calls) pushes prior bash
 * turns into the "stale" range without adding another tool_result to compact.
 */
function conversation({ turns = 3, output = lines(100), toolName = 'Bash', trailingTurns = 2 } = {}) {
  const msgs = []
  for (let i = 0; i < turns; i++) {
    const id = `t${i}`
    msgs.push({
      role: 'assistant',
      content: [{ type: 'tool_use', id, name: toolName, input: { command: `cmd-${i}` } }],
    })
    msgs.push({
      role: 'user',
      content: [
        { type: 'tool_result', tool_use_id: id, content: [{ type: 'text', text: output }] },
      ],
    })
  }
  for (let i = 0; i < trailingTurns; i++) {
    msgs.push({ role: 'assistant', content: [{ type: 'text', text: `thinking ${i}` }] })
    msgs.push({ role: 'user', content: [{ type: 'text', text: `next ${i}` }] })
  }
  return msgs
}

function firstText(block) {
  const c = block.content
  if (typeof c === 'string') return c
  if (Array.isArray(c)) return c.find((b) => b.type === 'text')?.text ?? ''
  return ''
}

function toolResults(messages) {
  const out = []
  for (const m of messages) {
    if (m.role !== 'user' || !Array.isArray(m.content)) continue
    for (const b of m.content) if (b.type === 'tool_result') out.push(b)
  }
  return out
}

// ── behavior ─────────────────────────────────────────────────────────────────

describe('compact', () => {
  test('truncates stale bash outputs beyond minLines', () => {
    const msgs = conversation({ turns: 1, output: lines(200), trailingTurns: 2 })
    const out = compact(msgs)
    const tr = toolResults(out)[0]
    const text = firstText(tr)
    assert.ok(text.startsWith('[COMPACTED'), 'expected compacted prefix')
    assert.ok(text.includes('lines compacted'), 'expected elision marker')
    assert.ok(text.length < lines(200).length, 'expected shrinkage')
  })

  test('leaves recent bash output intact (within staleTurns)', () => {
    // No trailing turns → the single bash turn IS the current turn.
    const msgs = conversation({ turns: 1, output: lines(200), trailingTurns: 0 })
    const out = compact(msgs)
    const text = firstText(toolResults(out)[0])
    assert.ok(!text.startsWith('[COMPACTED'))
    assert.equal(text.split('\n').length, 200)
  })

  test('skips outputs below minLines', () => {
    const msgs = conversation({ turns: 1, output: lines(10), trailingTurns: 2 })
    const out = compact(msgs)
    const text = firstText(toolResults(out)[0])
    assert.ok(!text.startsWith('[COMPACTED'))
  })

  test('does not touch Read/Edit/Grep tool_results', () => {
    for (const name of ['Read', 'Edit', 'Write', 'Grep', 'Glob']) {
      const msgs = conversation({ turns: 1, output: lines(200), toolName: name, trailingTurns: 2 })
      const out = compact(msgs)
      const text = firstText(toolResults(out)[0])
      assert.ok(!text.startsWith('[COMPACTED'), `${name} should be untouched`)
    }
  })

  test('preserves tool_use ↔ tool_result pairing', () => {
    const msgs = conversation({ turns: 3, output: lines(200), trailingTurns: 2 })
    const out = compact(msgs)

    const useIds = []
    for (const m of out) {
      if (m.role !== 'assistant' || !Array.isArray(m.content)) continue
      for (const b of m.content) if (b.type === 'tool_use') useIds.push(b.id)
    }
    const resultIds = toolResults(out).map((b) => b.tool_use_id)
    assert.deepEqual(resultIds, useIds, 'every tool_use still has a matching tool_result')
  })

  test('is idempotent — second pass makes no further changes', () => {
    const msgs = conversation({ turns: 2, output: lines(200), trailingTurns: 2 })
    const once = compact(msgs)
    const twice = compact(once)
    assert.deepEqual(twice, once)
  })

  test('handles string-form tool_result content', () => {
    const msgs = [
      {
        role: 'assistant',
        content: [{ type: 'tool_use', id: 'a', name: 'Bash', input: { command: 'ls' } }],
      },
      {
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 'a', content: lines(200) }],
      },
      { role: 'assistant', content: [{ type: 'text', text: '...' }] },
      { role: 'user', content: [{ type: 'text', text: '...' }] },
      { role: 'assistant', content: [{ type: 'text', text: '...' }] },
      { role: 'user', content: [{ type: 'text', text: '...' }] },
    ]
    const out = compact(msgs)
    const text = firstText(toolResults(out)[0])
    assert.ok(text.startsWith('[COMPACTED'))
  })

  test('does not mutate the input messages array', () => {
    const msgs = conversation({ turns: 1, output: lines(200), trailingTurns: 2 })
    const snapshot = JSON.parse(JSON.stringify(msgs))
    compact(msgs)
    assert.deepEqual(msgs, snapshot)
  })

  test('logs a compact event per block and a summary', () => {
    const events = []
    const msgs = conversation({ turns: 2, output: lines(200), trailingTurns: 2 })
    compact(msgs, (name, payload) => events.push({ name, payload }))

    const perBlock = events.filter((e) => e.name === 'compact')
    const summary = events.filter((e) => e.name === 'compact_summary')
    assert.equal(perBlock.length, 2)
    assert.equal(summary.length, 1)
    assert.equal(summary[0].payload.compactCount, 2)
    assert.ok(summary[0].payload.totalTokensSaved > 0)
  })
})
