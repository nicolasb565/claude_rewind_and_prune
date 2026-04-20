import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { injectClearToolUses, getConfig } from '../inject.mjs'

const TYPE = 'clear_tool_uses_20250919'

describe('injectClearToolUses', () => {
  test('adds edit to body with no context_management', () => {
    const body = { model: 'x' }
    injectClearToolUses(body)
    assert.equal(body.context_management.edits.length, 1)
    assert.equal(body.context_management.edits[0].type, TYPE)
  })

  test('preserves existing clear_thinking edit alongside', () => {
    const body = {
      context_management: {
        edits: [{ type: 'clear_thinking_20251015', keep: 'all' }],
      },
    }
    injectClearToolUses(body)
    const edits = body.context_management.edits
    assert.equal(edits.length, 2)
    assert.equal(edits[0].type, 'clear_thinking_20251015')
    assert.equal(edits[1].type, TYPE)
  })

  test('idempotent: does not duplicate if already present', () => {
    const body = {
      context_management: {
        edits: [{ type: TYPE, keep: { type: 'tool_uses', value: 99 } }],
      },
    }
    injectClearToolUses(body)
    assert.equal(body.context_management.edits.length, 1)
    // Client's existing config wins — we don't overwrite.
    assert.equal(body.context_management.edits[0].keep.value, 99)
  })

  test('excludes Read/Grep/Glob by default', () => {
    const body = {}
    injectClearToolUses(body)
    const excl = body.context_management.edits[0].exclude_tools
    assert.ok(excl.includes('Read'))
    assert.ok(excl.includes('Grep'))
    assert.ok(excl.includes('Glob'))
  })

  test('accepts custom config', () => {
    const body = {}
    injectClearToolUses(body, {
      trigger_input_tokens: 50000,
      keep_tool_uses: 5,
      clear_at_least_input_tokens: 10000,
      exclude_tools: ['Foo'],
    })
    const edit = body.context_management.edits[0]
    assert.equal(edit.trigger.value, 50000)
    assert.equal(edit.keep.value, 5)
    assert.equal(edit.clear_at_least.value, 10000)
    assert.deepEqual(edit.exclude_tools, ['Foo'])
  })

  test('handles null / non-object body safely', () => {
    assert.equal(injectClearToolUses(null), null)
    assert.equal(injectClearToolUses(undefined), undefined)
    assert.equal(injectClearToolUses('string'), 'string')
  })

  test('getConfig honors env vars', () => {
    const old = { ...process.env }
    try {
      process.env.INJECT_TRIGGER_INPUT_TOKENS = '99999'
      process.env.INJECT_KEEP_TOOL_USES = '7'
      process.env.INJECT_EXCLUDE_TOOLS = 'Read,Bash'
      const c = getConfig()
      assert.equal(c.trigger_input_tokens, 99999)
      assert.equal(c.keep_tool_uses, 7)
      assert.deepEqual(c.exclude_tools, ['Read', 'Bash'])
    } finally {
      for (const k of Object.keys(process.env)) {
        if (!(k in old)) delete process.env[k]
      }
      for (const [k, v] of Object.entries(old)) process.env[k] = v
    }
  })
})
