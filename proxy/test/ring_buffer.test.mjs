import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { RingBuffer } from '../ring_buffer.mjs'

describe('RingBuffer', () => {
  test('initial buildInput is zero-padded except for curr', () => {
    const ring = new RingBuffer(5, 8)
    const curr = new Float32Array(8).fill(1)
    const inp = ring.buildInput(curr)

    assert.equal(inp.length, 53) // 8*(1+5)+5
    for (let i = 0; i < 8; i++) assert.equal(inp[i], 1, `curr[${i}]`)
    for (let i = 8; i < 53; i++) assert.equal(inp[i], 0, `history/scores[${i}]`)
  })

  test('after one push, T-1 slot matches pushed features', () => {
    const ring = new RingBuffer(5, 8)
    const step0 = new Float32Array(8).fill(0.5)
    ring.push(step0, 0.9)

    const inp = ring.buildInput(new Float32Array(8).fill(0))
    // T-1 slot: positions [8..16)
    for (let i = 0; i < 8; i++) assert.ok(Math.abs(inp[8 + i] - 0.5) < 1e-6, `T-1 feat[${i}]`)
    // score T-1: position 48
    assert.ok(Math.abs(inp[48] - 0.9) < 1e-6)
  })

  test('T-2 slot is correct after two pushes', () => {
    const ring = new RingBuffer(5, 8)
    ring.push(new Float32Array(8).fill(0.1), 0.1)
    ring.push(new Float32Array(8).fill(0.2), 0.2)

    const inp = ring.buildInput(new Float32Array(8).fill(0))
    // T-1: [8..16) = 0.2
    for (let i = 0; i < 8; i++) assert.ok(Math.abs(inp[8 + i] - 0.2) < 1e-6, `T-1[${i}]`)
    // T-2: [16..24) = 0.1
    for (let i = 0; i < 8; i++) assert.ok(Math.abs(inp[16 + i] - 0.1) < 1e-6, `T-2[${i}]`)
    // scores
    assert.ok(Math.abs(inp[48] - 0.2) < 1e-6)
    assert.ok(Math.abs(inp[49] - 0.1) < 1e-6)
  })

  test('T-3 slot holds step-0 features after 3 pushes', () => {
    const ring = new RingBuffer(5, 8)
    const step0 = new Float32Array(8).fill(0.7)
    ring.push(step0, 0)
    ring.push(new Float32Array(8).fill(0.1), 0)
    ring.push(new Float32Array(8).fill(0.2), 0)

    const inp = ring.buildInput(new Float32Array(8))
    // T-3 slot: positions [3*8..4*8) = [24..32)
    for (let i = 0; i < 8; i++) assert.ok(Math.abs(inp[24 + i] - 0.7) < 1e-6, `T-3[${i}]`)
  })

  test('oldest entry evicted after N+1 pushes', () => {
    const ring = new RingBuffer(3, 4) // depth=3, dim=4 → input length = 4*4+3 = 19
    // Push 4 steps: step0(value=1) gets evicted by step3(value=4)
    for (let v = 1; v <= 4; v++) ring.push(new Float32Array(4).fill(v), 0)

    const inp = ring.buildInput(new Float32Array(4))
    // T-3 slot: [3*4..4*4) = [12..16): should be step1 (value=2), step0 is evicted
    for (let i = 0; i < 4; i++) assert.equal(inp[12 + i], 2, `oldest[${i}]`)
  })

  test('scores are correctly tracked across multiple pushes', () => {
    const ring = new RingBuffer(5, 8)
    ring.push(new Float32Array(8), 0.3)
    ring.push(new Float32Array(8), 0.6)
    ring.push(new Float32Array(8), 0.9)

    const inp = ring.buildInput(new Float32Array(8))
    assert.ok(Math.abs(inp[48] - 0.9) < 1e-6) // score T-1
    assert.ok(Math.abs(inp[49] - 0.6) < 1e-6) // score T-2
    assert.ok(Math.abs(inp[50] - 0.3) < 1e-6) // score T-3
    assert.equal(inp[51], 0.0) // score T-4 (zero-padded)
    assert.equal(inp[52], 0.0) // score T-5 (zero-padded)
  })

  test('buildInput does not mutate the ring', () => {
    const ring = new RingBuffer(5, 8)
    ring.push(new Float32Array(8).fill(0.5), 0.7)
    ring.buildInput(new Float32Array(8).fill(1))
    ring.buildInput(new Float32Array(8).fill(2))
    // Ring state should be unchanged between calls
    const inp = ring.buildInput(new Float32Array(8).fill(0))
    assert.ok(Math.abs(inp[8] - 0.5) < 1e-6)
    assert.ok(Math.abs(inp[48] - 0.7) < 1e-6)
  })
})
