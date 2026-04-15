import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { LRClassifier, loadLR } from '../lr.mjs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const WEIGHTS = resolve(__dirname, '..', 'lr_weights.json')

describe('LRClassifier — toy weights', () => {
  // Tiny synthetic model: 2 features, w=[1, -1], b=0, identity normalization.
  // logit = z0 - z1; sigmoid(0) = 0.5; sigmoid(2) = 0.8808...
  const toy = new LRClassifier({
    features: ['a', 'b'],
    weights: [1.0, -1.0],
    intercept: 0.0,
    feature_mean: [0.0, 0.0],
    feature_std: [1.0, 1.0],
  })

  test('all-zero input → 0.5', () => {
    const p = toy.score([0, 0])
    assert.ok(Math.abs(p - 0.5) < 1e-12, `expected 0.5, got ${p}`)
  })

  test('positive logit → > 0.5', () => {
    const p = toy.score([1, 0])
    assert.ok(p > 0.5)
    assert.ok(Math.abs(p - 1 / (1 + Math.exp(-1))) < 1e-12)
  })

  test('negative logit → < 0.5', () => {
    const p = toy.score([0, 1])
    assert.ok(p < 0.5)
    assert.ok(Math.abs(p - 1 / (1 + Math.exp(1))) < 1e-12)
  })

  test('feature row length mismatch throws', () => {
    assert.throws(() => toy.score([1, 2, 3]), /length/)
  })

  test('normalization: mean=2 std=2 maps 4→z=1', () => {
    const m = new LRClassifier({
      features: ['x'],
      weights: [2.0],
      intercept: 0.0,
      feature_mean: [2.0],
      feature_std: [2.0],
    })
    // z = (4-2)/2 = 1, logit = 2*1 = 2, sigmoid(2) ≈ 0.88079
    const p = m.score([4])
    assert.ok(Math.abs(p - 1 / (1 + Math.exp(-2))) < 1e-12)
  })
})

describe('LRClassifier — production weights', () => {
  const lr = loadLR(WEIGHTS)

  test('loaded with 8 features', () => {
    assert.equal(lr.features.length, 8)
  })

  test('all-zero feature row produces a finite probability in [0,1]', () => {
    const z = new Float64Array(8)
    const p = lr.score(z)
    assert.ok(Number.isFinite(p))
    assert.ok(p >= 0 && p <= 1)
  })

  test('saturating positive features → high score', () => {
    // match_ratio_5=1, self_sim_max=1, repeat_no_error=1, cur_bash_and_match_ratio=1
    // with the rest zero, should produce a clearly stuck score (>0.5)
    const high = new Float64Array([1, 1, 1, 1, 0, 0, 0, 0])
    const p = lr.score(high)
    assert.ok(p > 0.5, `expected > 0.5 for stuck-shaped row, got ${p}`)
  })

  test('saturating negative features → low score', () => {
    // unique_err_sigs_6=1 (negative weight), new_token_ratio=1 (negative),
    // has_success_marker=1 (negative), err_volume_ratio=3 (negative)
    const low = new Float64Array([0, 0, 0, 0, 1, 1, 1, 3])
    const p = lr.score(low)
    assert.ok(p < 0.5, `expected < 0.5 for productive-shaped row, got ${p}`)
  })
})
