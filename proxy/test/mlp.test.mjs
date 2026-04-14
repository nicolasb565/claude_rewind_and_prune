import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { loadMLP } from '../mlp.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))
const WEIGHTS_PATH = resolve(__dirname, '../stuck_weights.json')
const VECTORS_PATH = resolve(__dirname, 'mlp_parity_vectors.json')

describe('MLP', () => {
  const mlp = loadMLP(WEIGHTS_PATH)

  test('output is in [0, 1] for zero input', () => {
    const score = mlp.forward(new Float32Array(42))
    assert.ok(score >= 0 && score <= 1, `Score ${score} out of [0,1]`)
  })

  test('output is in [0, 1] for all-ones input', () => {
    const score = mlp.forward(new Float32Array(42).fill(1))
    assert.ok(score >= 0 && score <= 1, `Score ${score} out of [0,1]`)
  })

  test('output is finite for random inputs', () => {
    for (let i = 0; i < 10; i++) {
      const input = new Float32Array(42).map(() => Math.random() * 2 - 1)
      assert.ok(Number.isFinite(mlp.forward(input)))
    }
  })

  test('different inputs produce different outputs', () => {
    const a = mlp.forward(new Float32Array(42).fill(0))
    const b = mlp.forward(new Float32Array(42).fill(1))
    assert.notEqual(a, b)
  })

  test('parity with Python reference vectors (max diff < 1e-5)', () => {
    const vectors = JSON.parse(readFileSync(VECTORS_PATH, 'utf8'))
    let maxDiff = 0

    for (const { input, score: expected } of vectors) {
      const actual = mlp.forward(new Float32Array(input))
      const diff = Math.abs(actual - expected)
      if (diff > maxDiff) maxDiff = diff
      assert.ok(
        diff < 1e-5,
        `Parity failure: expected=${expected.toFixed(8)} actual=${actual.toFixed(8)} diff=${diff.toExponential(3)}`,
      )
    }

    process.stderr.write(
      `  [parity] max diff: ${maxDiff.toExponential(3)} over ${vectors.length} vectors\n`,
    )
  })
})
