import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import {
  extractResultText,
  extractAllToolCalls,
  extractLastTurnToolCalls,
  getSessionKey,
  recentToolSummary,
} from '../messages.mjs'

// ── helpers ───────────────────────────────────────────────────────────────────

/** Build a minimal messages array with one tool_use / tool_result pair. */
function oneToolConversation({ toolName = 'Bash', input = { command: 'ls' }, output = 'ok' } = {}) {
  return [
    {
      role: 'assistant',
      content: [{ type: 'tool_use', id: 't1', name: toolName, input }],
    },
    {
      role: 'user',
      content: [
        { type: 'tool_result', tool_use_id: 't1', content: [{ type: 'text', text: output }] },
      ],
    },
  ]
}

// ── extractResultText ─────────────────────────────────────────────────────────

describe('extractResultText', () => {
  test('returns string content unchanged', () => {
    assert.equal(extractResultText('hello'), 'hello')
  })

  test('returns empty string for null/undefined', () => {
    assert.equal(extractResultText(null), '')
    assert.equal(extractResultText(undefined), '')
  })

  test('single text block returns its text', () => {
    assert.equal(extractResultText([{ type: 'text', text: 'hello' }]), 'hello')
  })

  test('multi-block content is joined with newlines', () => {
    const result = extractResultText([
      { type: 'text', text: 'line1' },
      { type: 'text', text: 'line2' },
    ])
    assert.equal(result, 'line1\nline2')
  })

  test('non-text blocks are filtered out', () => {
    const result = extractResultText([
      { type: 'image', source: '...' },
      { type: 'text', text: 'only this' },
    ])
    assert.equal(result, 'only this')
  })
})

// ── extractAllToolCalls ───────────────────────────────────────────────────────

describe('extractAllToolCalls', () => {
  test('extracts single tool call with its output', () => {
    const calls = extractAllToolCalls(oneToolConversation())
    assert.equal(calls.length, 1)
    assert.equal(calls[0].toolName, 'Bash')
    assert.equal(calls[0].input.command, 'ls')
    assert.equal(calls[0].output, 'ok')
  })

  test('returns empty array for messages with no tool calls', () => {
    const messages = [
      { role: 'user', content: [{ type: 'text', text: 'hello' }] },
      { role: 'assistant', content: [{ type: 'text', text: 'hi' }] },
    ]
    assert.deepEqual(extractAllToolCalls(messages), [])
  })

  test('emits in tool_result order (matches Python parse_session)', () => {
    // Python's src/pipeline/parsers/nlile.py:parse_session is single-pass
    // and appends each step the moment its tool_result is encountered.
    // The LR classifier was trained on Python-parsed features, so JS must
    // emit in the same order or the rolling history features drift.
    const messages = [
      {
        role: 'assistant',
        content: [
          { type: 'tool_use', id: 't1', name: 'Read', input: { file_path: 'a.py' } },
          { type: 'tool_use', id: 't2', name: 'Bash', input: { command: 'ls' } },
        ],
      },
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 't2', content: [{ type: 'text', text: 'files' }] },
          { type: 'tool_result', tool_use_id: 't1', content: [{ type: 'text', text: 'content' }] },
        ],
      },
    ]
    const calls = extractAllToolCalls(messages)
    assert.equal(calls.length, 2)
    assert.equal(calls[0].toolName, 'Bash') // result order: t2 first
    assert.equal(calls[1].toolName, 'Read')
  })

  test('multi-block tool_result output is joined with newlines', () => {
    const messages = [
      { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'Bash', input: {} }] },
      {
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 't1',
            content: [
              { type: 'text', text: 'stdout' },
              { type: 'text', text: 'stderr' },
            ],
          },
        ],
      },
    ]
    const calls = extractAllToolCalls(messages)
    assert.equal(calls[0].output, 'stdout\nstderr')
  })

  test('tool call with no matching result gets empty output', () => {
    const messages = [
      { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'Bash', input: {} }] },
      { role: 'user', content: [{ type: 'text', text: 'no result here' }] },
    ]
    const calls = extractAllToolCalls(messages)
    assert.equal(calls[0].output, '')
  })
})

// ── extractLastTurnToolCalls ──────────────────────────────────────────────────

describe('extractLastTurnToolCalls', () => {
  test('returns only tools from the last assistant turn', () => {
    const messages = [
      {
        role: 'assistant',
        content: [{ type: 'tool_use', id: 't1', name: 'Read', input: { file_path: 'old.py' } }],
      },
      {
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 't1', content: 'old result' }],
      },
      {
        role: 'assistant',
        content: [{ type: 'tool_use', id: 't2', name: 'Bash', input: { command: 'pwd' } }],
      },
      {
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 't2', content: 'new result' }],
      },
    ]
    const calls = extractLastTurnToolCalls(messages)
    assert.equal(calls.length, 1)
    assert.equal(calls[0].toolName, 'Bash')
    assert.equal(calls[0].output, 'new result')
  })

  test('returns empty array when no assistant turn exists', () => {
    const messages = [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }]
    assert.deepEqual(extractLastTurnToolCalls(messages), [])
  })

  test('multi-block tool_result output is joined with newlines', () => {
    const messages = [
      { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'Bash', input: {} }] },
      {
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 't1',
            content: [
              { type: 'text', text: 'line1' },
              { type: 'text', text: 'line2' },
            ],
          },
        ],
      },
    ]
    const calls = extractLastTurnToolCalls(messages)
    assert.equal(calls[0].output, 'line1\nline2')
  })
})

// ── getSessionKey ─────────────────────────────────────────────────────────────

describe('getSessionKey', () => {
  test('returns first 200 chars of first user message', () => {
    const long = 'x'.repeat(300)
    const messages = [{ role: 'user', content: long }]
    assert.equal(getSessionKey(messages).length, 200)
  })

  test('returns __default__ for empty messages array', () => {
    assert.equal(getSessionKey([]), '__default__')
  })

  test('returns __default__ when first user message is empty', () => {
    const messages = [{ role: 'user', content: '' }]
    assert.equal(getSessionKey(messages), '__default__')
  })

  test('handles array content blocks', () => {
    const messages = [
      {
        role: 'user',
        content: [{ type: 'text', text: 'hello world' }],
      },
    ]
    assert.equal(getSessionKey(messages), 'hello world')
  })

  test('skips non-user messages', () => {
    const messages = [
      { role: 'assistant', content: 'ignored' },
      { role: 'user', content: 'first user' },
    ]
    assert.equal(getSessionKey(messages), 'first user')
  })

  test('strips <system-reminder> blocks', () => {
    const messages = [{
      role: 'user',
      content: [{
        type: 'text',
        text: '<system-reminder>boilerplate</system-reminder>real task prompt',
      }],
    }]
    assert.equal(getSessionKey(messages), 'real task prompt')
  })

  test('strips multi-line and multi-block system reminders', () => {
    const messages = [{
      role: 'user',
      content: [{
        type: 'text',
        text: '<system-reminder>\na\nb\n</system-reminder><system-reminder>c</system-reminder>  actual prompt  ',
      }],
    }]
    assert.equal(getSessionKey(messages), 'actual prompt')
  })

  test('skips purely-boilerplate user messages and uses the next', () => {
    const messages = [
      {
        role: 'user',
        content: [{ type: 'text', text: '<system-reminder>only boilerplate</system-reminder>' }],
      },
      {
        role: 'assistant',
        content: [{ type: 'text', text: 'ignored' }],
      },
      {
        role: 'user',
        content: [{ type: 'text', text: 'real task' }],
      },
    ]
    assert.equal(getSessionKey(messages), 'real task')
  })

  test('ignores non-text blocks like tool_result when computing the key', () => {
    const messages = [{
      role: 'user',
      content: [
        { type: 'tool_result', tool_use_id: 't1', content: 'noise' },
        { type: 'text', text: 'the real prompt' },
      ],
    }]
    assert.equal(getSessionKey(messages), 'the real prompt')
  })
})

// ── recentToolSummary ─────────────────────────────────────────────────────────

describe('recentToolSummary', () => {
  test('returns empty array for messages with no tool calls', () => {
    const messages = [{ role: 'user', content: [{ type: 'text', text: 'hello' }] }]
    assert.deepEqual(recentToolSummary(messages), [])
  })

  test('formats each entry as "ToolName: detail"', () => {
    const messages = [
      {
        role: 'assistant',
        content: [{ type: 'tool_use', id: 't1', name: 'Bash', input: { command: 'ls -la' } }],
      },
    ]
    const summary = recentToolSummary(messages)
    assert.equal(summary.length, 1)
    assert.ok(summary[0].startsWith('Bash: ls -la'))
  })

  test('truncates detail to 60 chars from the start', () => {
    const long = 'a'.repeat(100)
    const messages = [
      {
        role: 'assistant',
        content: [{ type: 'tool_use', id: 't1', name: 'Bash', input: { command: long } }],
      },
    ]
    const summary = recentToolSummary(messages)
    // "Bash: " + 60 chars
    assert.equal(summary[0], `Bash: ${'a'.repeat(60)}`)
  })

  test('caps output at 8 entries', () => {
    const content = Array.from({ length: 20 }, (_, i) => ({
      type: 'tool_use',
      id: `t${i}`,
      name: 'Bash',
      input: { command: `cmd${i}` },
    }))
    const messages = [{ role: 'assistant', content }]
    assert.equal(recentToolSummary(messages).length, 8)
  })
})
