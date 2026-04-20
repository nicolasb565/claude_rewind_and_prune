import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { extractUsage } from '../usage.mjs'

describe('extractUsage', () => {
  test('parses non-streaming JSON response', () => {
    const body = JSON.stringify({
      id: 'msg_1',
      type: 'message',
      usage: {
        input_tokens: 100,
        cache_creation_input_tokens: 50,
        cache_read_input_tokens: 1000,
        output_tokens: 20,
      },
    })
    const u = extractUsage(body)
    assert.equal(u.input_tokens, 100)
    assert.equal(u.cache_creation_input_tokens, 50)
    assert.equal(u.cache_read_input_tokens, 1000)
    assert.equal(u.output_tokens, 20)
  })

  test('parses SSE message_start event', () => {
    const sse =
      'event: message_start\n' +
      'data: {"type":"message_start","message":{"id":"msg_1","usage":' +
      '{"input_tokens":42,"cache_creation_input_tokens":10,' +
      '"cache_read_input_tokens":500,"output_tokens":1}}}\n\n' +
      'event: content_block_start\n' +
      'data: {"type":"content_block_start","index":0}\n\n'
    const u = extractUsage(sse)
    assert.equal(u.input_tokens, 42)
    assert.equal(u.cache_read_input_tokens, 500)
    assert.equal(u.cache_creation_input_tokens, 10)
  })

  test('parses SSE message_delta usage', () => {
    const sse =
      'event: message_delta\n' +
      'data: {"type":"message_delta","delta":{},"usage":{"output_tokens":15}}\n\n'
    const u = extractUsage(sse)
    assert.equal(u.output_tokens, 15)
  })

  test('returns null for empty / malformed input', () => {
    assert.equal(extractUsage(''), null)
    assert.equal(extractUsage(null), null)
    assert.equal(extractUsage('not json, not sse'), null)
    assert.equal(extractUsage('data: not-json\n\n'), null)
  })

  test('ignores data: [DONE] sentinels', () => {
    const sse = 'data: [DONE]\n\n'
    assert.equal(extractUsage(sse), null)
  })

  test('prefers first event with usage over later ones', () => {
    const sse =
      'event: message_start\n' +
      'data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}\n\n' +
      'event: message_delta\n' +
      'data: {"type":"message_delta","usage":{"output_tokens":99}}\n\n'
    const u = extractUsage(sse)
    assert.equal(u.input_tokens, 1)
  })
})
