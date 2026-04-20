import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { applyRewind } from '../rewind.mjs'

// ── helpers ───────────────────────────────────────────────────────────────

function userText(text) {
  return { role: 'user', content: [{ type: 'text', text }] }
}
function assistantToolUse(name, id, input = {}) {
  return { role: 'assistant', content: [{ type: 'tool_use', id, name, input }] }
}
function assistantText(text) {
  return { role: 'assistant', content: [{ type: 'text', text }] }
}
function userToolResult(toolUseId, text) {
  return {
    role: 'user',
    content: [{ type: 'tool_result', tool_use_id: toolUseId, content: [{ type: 'text', text }] }],
  }
}
function summarize(id, summary) {
  // Legacy-shaped input (single `summary` field) — still supported for
  // backward compat with any pre-schema history.
  return assistantToolUse('mcp__bookmarks__checkpoint_progress', id, { summary })
}
function checkpoint(id, { finding, evidence, next_direction, progress_type = 'milestone_achieved' }) {
  return assistantToolUse('mcp__bookmarks__checkpoint_progress', id, {
    progress_type, finding, evidence, next_direction,
  })
}
function summarizeResult(id) {
  return userToolResult(id, 'Checkpoint saved.')
}

// ── behavior ──────────────────────────────────────────────────────────────

describe('applyRewind', () => {
  test('no-op when no checkpoint_progress in history', () => {
    const msgs = [
      userText('fix the bug'),
      assistantToolUse('Bash', 't1', { command: 'ls' }),
      userToolResult('t1', 'file.c'),
    ]
    const { messages, elisions } = applyRewind(msgs)
    assert.deepEqual(messages, msgs)
    assert.equal(elisions.length, 0)
  })

  test('elides turns between initial prompt and single summarize call', () => {
    const msgs = [
      userText('fix the bug'),
      assistantToolUse('Bash', 't1', { command: 'make' }),
      userToolResult('t1', 'error log'),
      assistantToolUse('Read', 't2', { path: 'foo.c' }),
      userToolResult('t2', 'contents'),
      summarize('s1', 'Tried rebuild, wrong path'),
      summarizeResult('s1'),
      assistantToolUse('Bash', 't3', { command: 'grep' }),
      userToolResult('t3', 'match'),
    ]
    const { messages, elisions } = applyRewind(msgs)

    // Original prompt preserved
    assert.equal(messages[0], msgs[0])
    // Synthetic summary msg inserted
    assert.equal(messages[1].role, 'user')
    const summaryText = messages[1].content[0].text
    assert.ok(summaryText.includes('Tried rebuild, wrong path'))
    // checkpoint_progress pair preserved
    assert.equal(messages[2], msgs[5])
    assert.equal(messages[3], msgs[6])
    // Post-anchor turns passed through
    assert.equal(messages[4], msgs[7])
    assert.equal(messages[5], msgs[8])
    // Total length shrunk
    assert.ok(messages.length < msgs.length)
    // One elision recorded
    assert.equal(elisions.length, 1)
    assert.equal(elisions[0].turnsElided, 4)
  })

  test('handles multiple summarize calls cumulatively', () => {
    const msgs = [
      userText('debug issue'),
      assistantToolUse('Bash', 'a1', { command: 'first' }),
      userToolResult('a1', 'r1'),
      summarize('s1', 'First attempt failed'),
      summarizeResult('s1'),
      assistantToolUse('Bash', 'a2', { command: 'second' }),
      userToolResult('a2', 'r2'),
      summarize('s2', 'Second attempt also failed'),
      summarizeResult('s2'),
      assistantToolUse('Bash', 'a3', { command: 'third' }),
      userToolResult('a3', 'r3'),
    ]
    const { messages, elisions } = applyRewind(msgs)

    assert.equal(elisions.length, 2)
    // First elision covers [1..2] (2 turns)
    assert.equal(elisions[0].turnsElided, 2)
    assert.equal(elisions[0].summary, 'First attempt failed')
    // Second elision covers [5..6] (2 turns) — the ones after s1 pair
    assert.equal(elisions[1].turnsElided, 2)
    assert.equal(elisions[1].summary, 'Second attempt also failed')
    // Shape: prompt, synth1, s1u, s1r, synth2, s2u, s2r, a3u, a3r
    assert.equal(messages.length, 9)
  })

  test('preserves tool_use ↔ tool_result pairing', () => {
    const msgs = [
      userText('start'),
      assistantToolUse('Bash', 'x', { command: 'c' }),
      userToolResult('x', 'r'),
      assistantToolUse('Read', 'y', { path: 'p' }),
      userToolResult('y', 'r2'),
      summarize('s1', 'dead end'),
      summarizeResult('s1'),
    ]
    const { messages } = applyRewind(msgs)

    // Every tool_use id should have a matching tool_result id.
    const useIds = new Set()
    const resultIds = new Set()
    for (const m of messages) {
      if (!Array.isArray(m.content)) continue
      for (const b of m.content) {
        if (b.type === 'tool_use') useIds.add(b.id)
        if (b.type === 'tool_result') resultIds.add(b.tool_use_id)
      }
    }
    assert.deepEqual([...useIds].sort(), [...resultIds].sort(),
      'every tool_use must have paired tool_result and vice versa')
  })

  test('does nothing unsafe when summarize is the first assistant turn', () => {
    // Edge case: agent summarizes before any other work. Nothing to elide.
    const msgs = [
      userText('start'),
      summarize('s1', 'nothing to say'),
      summarizeResult('s1'),
      assistantToolUse('Bash', 'b', { command: 'ls' }),
      userToolResult('b', 'ok'),
    ]
    const { messages, elisions } = applyRewind(msgs)
    // No elision: elideTo - elideFrom = 1 - 1 = 0.
    assert.equal(elisions.length, 0)
    // Messages should pass through unchanged.
    assert.deepEqual(messages, msgs)
  })

  test('idempotent: applying again is a no-op on already-rewritten output', () => {
    const msgs = [
      userText('start'),
      assistantToolUse('Bash', 't1', { command: 'x' }),
      userToolResult('t1', 'out'),
      summarize('s1', 'done'),
      summarizeResult('s1'),
    ]
    const once = applyRewind(msgs).messages
    const twice = applyRewind(once).messages
    assert.deepEqual(twice, once)
  })

  test('does not mutate the input array', () => {
    const msgs = [
      userText('start'),
      assistantToolUse('Bash', 't1', { command: 'x' }),
      userToolResult('t1', 'out'),
      summarize('s1', 'dead end'),
      summarizeResult('s1'),
    ]
    const snapshot = JSON.parse(JSON.stringify(msgs))
    applyRewind(msgs)
    assert.deepEqual(msgs, snapshot)
  })

  test('returns bytes saved in elision metadata', () => {
    const msgs = [
      userText('start'),
      assistantToolUse('Bash', 'b', { command: 'x'.repeat(1000) }),
      userToolResult('b', 'y'.repeat(5000)),
      summarize('s1', 'dead end'),
      summarizeResult('s1'),
    ]
    const { elisions } = applyRewind(msgs)
    assert.equal(elisions.length, 1)
    assert.ok(elisions[0].bytesSaved > 5000,
      `expected >5000 bytes saved, got ${elisions[0].bytesSaved}`)
  })

  test('uses structured fields for synthetic summary', () => {
    const msgs = [
      userText('start'),
      assistantToolUse('Bash', 't1', { command: 'x' }),
      userToolResult('t1', 'out'),
      checkpoint('s1', {
        progress_type: 'approach_eliminated',
        finding: 'X is not the cause',
        evidence: 'Reverted X, bug still reproduces with same output',
        next_direction: 'Investigate Y next',
      }),
      summarizeResult('s1'),
    ]
    const { messages, elisions } = applyRewind(msgs)
    const synthText = messages[1].content[0].text
    assert.ok(synthText.includes('Approach eliminated'))
    assert.ok(synthText.includes('X is not the cause'))
    assert.ok(synthText.includes('Reverted X'))
    assert.ok(synthText.includes('Investigate Y next'))
    assert.equal(elisions[0].summary, synthText.replace(/^\[Previous exploration elided — agent summary: /, '').replace(/]$/, ''))
  })

  test('matches both bare and mcp-prefixed tool names', () => {
    const mkWithName = (name) => [
      userText('start'),
      assistantToolUse('Bash', 't1', { command: 'x' }),
      userToolResult('t1', 'out'),
      { role: 'assistant', content: [{ type: 'tool_use', id: 's', name, input: { summary: 'x' } }] },
      userToolResult('s', 'ok'),
    ]
    const r1 = applyRewind(mkWithName('checkpoint_progress'))
    const r2 = applyRewind(mkWithName('mcp__bookmarks__checkpoint_progress'))
    assert.equal(r1.elisions.length, 1)
    assert.equal(r2.elisions.length, 1)
  })
})
