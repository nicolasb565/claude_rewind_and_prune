import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { summarizeRequest } from '../introspect.mjs'

describe('summarizeRequest', () => {
  test('reports model, stream, max_tokens', () => {
    const s = summarizeRequest({
      body: { model: 'claude-opus-4-7', stream: true, max_tokens: 4096 },
      headers: {},
    })
    assert.equal(s.model, 'claude-opus-4-7')
    assert.equal(s.stream, true)
    assert.equal(s.max_tokens, 4096)
  })

  test('parses anthropic-beta header into array', () => {
    const s = summarizeRequest({
      body: {},
      headers: { 'anthropic-beta': 'context-management-2025-06-27,cache_edits-2025-09-15' },
    })
    assert.deepEqual(s.betas, ['context-management-2025-06-27', 'cache_edits-2025-09-15'])
  })

  test('surfaces context_management strategies', () => {
    const s = summarizeRequest({
      body: {
        context_management: {
          edits: [
            { type: 'clear_tool_uses_20250919', trigger: { type: 'input_tokens', value: 30000 }, keep: { type: 'tool_uses', value: 3 }, clear_at_least: { type: 'input_tokens', value: 5000 } },
          ],
        },
      },
      headers: {},
    })
    assert.equal(s.context_management.edit_count, 1)
    assert.equal(s.context_management.strategies[0].type, 'clear_tool_uses_20250919')
    assert.equal(s.context_management.strategies[0].trigger.value, 30000)
  })

  test('omits context_management when absent', () => {
    const s = summarizeRequest({ body: {}, headers: {} })
    assert.equal(s.context_management, undefined)
  })

  test('measures system size (string and blocks)', () => {
    assert.equal(summarizeRequest({ body: { system: 'abc' }, headers: {} }).system_size_chars, 3)
    assert.equal(
      summarizeRequest({
        body: { system: [{ type: 'text', text: 'hello' }, { type: 'text', text: 'world!' }] },
        headers: {},
      }).system_size_chars,
      11,
    )
  })

  test('tallies tool_result sizes + detects disk-offload markers', () => {
    const body = {
      messages: [
        {
          role: 'user',
          content: [
            { type: 'tool_result', tool_use_id: 'a', content: 'x'.repeat(100) },
            { type: 'tool_result', tool_use_id: 'b', content: 'y'.repeat(10_000) },
            // Looks like a Claude Code offload preview
            { type: 'tool_result', tool_use_id: 'c', content: 'Large output saved to /tmp/output-abc.txt' },
          ],
        },
      ],
    }
    const s = summarizeRequest({ body, headers: {} })
    assert.equal(s.tool_results.count, 3)
    assert.equal(s.tool_results.max_chars, 10_000)
    assert.equal(s.tool_results.offloaded_markers, 1)
  })

  test('counts tools', () => {
    const s = summarizeRequest({
      body: { tools: [{ name: 'Bash' }, { name: 'Read' }, { name: 'Edit' }] },
      headers: {},
    })
    assert.equal(s.tool_count, 3)
  })
})
