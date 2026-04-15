import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { TieredFilter, TieredNudgeController, DEFAULT_TIERED_CONFIG } from '../tiered_filter.mjs'

// ── TieredFilter ──────────────────────────────────────────────────────────

describe('TieredFilter — warmup and aggregation', () => {
  test('all filters are off until each buffer is full', () => {
    const f = new TieredFilter()
    // mean-of-2 needs 2 samples; median-of-4 needs 4; median-of-9 needs 9.
    // Push 1 sample: nothing should fire.
    let s = f.update(0.99)
    assert.equal(s.soft, false)
    assert.equal(s.medium, false)
    assert.equal(s.hard, false)
    assert.equal(s.aggregates.soft, null)
    // Push a second high sample: soft is now ready and >= 0.34
    s = f.update(0.99)
    assert.equal(s.soft, true)
    assert.equal(s.medium, false)
    assert.equal(s.hard, false)
    assert.ok(s.aggregates.soft >= 0.34)
  })

  test('mean-of-2 aggregate is exactly the arithmetic mean of last 2 scores', () => {
    const f = new TieredFilter()
    f.update(0.4)
    const s = f.update(0.2)
    assert.ok(Math.abs(s.aggregates.soft - 0.3) < 1e-12)
  })

  test('median-of-4 with [0.1, 0.7, 0.8, 0.6] = 0.65', () => {
    const f = new TieredFilter()
    f.update(0.1); f.update(0.7); f.update(0.8)
    const s = f.update(0.6)
    // Sorted: [0.1, 0.6, 0.7, 0.8] → median = (0.6+0.7)/2 = 0.65
    assert.ok(Math.abs(s.aggregates.medium - 0.65) < 1e-12)
    assert.equal(s.medium, true) // >= 0.645
  })

  test('a single low score in last 2 prevents soft from firing', () => {
    const f = new TieredFilter()
    f.update(0.9)
    const s = f.update(0.1)
    // mean = 0.5 → soft fires (>= 0.34)
    assert.equal(s.soft, true)
    // Now push another low score so the pair is [0.1, 0.05]
    const s2 = f.update(0.05)
    assert.equal(s2.soft, false) // mean 0.075 < 0.34
  })

  test('median ignores symmetric outliers; mean-of-2 does not', () => {
    const f = new TieredFilter()
    // Build a window where mean-of-2 would fire on the spike but
    // median-of-4 would not.
    f.update(0.1); f.update(0.1); f.update(0.1)
    const s = f.update(0.99)
    // mean2(0.1, 0.99) = 0.545, soft fires
    assert.equal(s.soft, true)
    // median4(0.1, 0.1, 0.1, 0.99) = 0.1, medium does NOT fire
    assert.equal(s.medium, false)
  })

  test('reset clears all buffers — filters go back to warmup', () => {
    const f = new TieredFilter()
    for (let i = 0; i < 10; i++) f.update(0.99)
    assert.equal(f.update(0.99).hard, true)
    f.reset()
    const s = f.update(0.99)
    // Fresh start: only 1 score in buffer, none ready
    assert.equal(s.soft, false)
    assert.equal(s.medium, false)
    assert.equal(s.hard, false)
  })

  test('config override is respected', () => {
    const f = new TieredFilter({
      soft:   { kind: 'mean',   n: 1, threshold: 0.5 },
      medium: { kind: 'mean',   n: 1, threshold: 0.7 },
      hard:   { kind: 'mean',   n: 1, threshold: 0.9 },
    })
    let s = f.update(0.95)
    assert.equal(s.soft, true)
    assert.equal(s.medium, true)
    assert.equal(s.hard, true)
    s = f.update(0.6)
    assert.equal(s.soft, true)
    assert.equal(s.medium, false)
    assert.equal(s.hard, false)
  })
})

// ── TieredNudgeController ────────────────────────────────────────────────

describe('TieredNudgeController — state machine', () => {
  test('starts at idle (-1)', () => {
    const c = new TieredNudgeController()
    assert.equal(c.level, -1)
  })

  test('soft-only filter advances level -1 → 0 with fire', () => {
    const c = new TieredNudgeController()
    const r = c.update({ soft: true, medium: false, hard: false }, 0.4, 1, [])
    assert.equal(r.fire, true)
    assert.equal(r.level, 0)
    assert.equal(r.prevLevel, -1)
    assert.ok(r.text.length > 0)
  })

  test('soft filter off resets to -1, no fire', () => {
    const c = new TieredNudgeController()
    c.update({ soft: true, medium: false, hard: false }, 0.4, 1, [])
    const r = c.update({ soft: false, medium: false, hard: false }, 0.1, 2, [])
    assert.equal(r.fire, false)
    assert.equal(r.level, -1)
    assert.equal(r.prevLevel, 0)
  })

  test('does NOT advance to medium without medium filter agreeing', () => {
    const c = new TieredNudgeController()
    c.update({ soft: true, medium: false, hard: false }, 0.4, 1, []) // → 0
    const r = c.update({ soft: true, medium: false, hard: false }, 0.4, 2, [])
    assert.equal(r.fire, false)
    assert.equal(r.level, 0)
  })

  test('medium filter advances level 0 → 1', () => {
    const c = new TieredNudgeController()
    c.update({ soft: true, medium: false, hard: false }, 0.4, 1, [])
    const r = c.update({ soft: true, medium: true, hard: false }, 0.7, 2, [])
    assert.equal(r.fire, true)
    assert.equal(r.level, 1)
  })

  test('hard filter advances level 1 → 2 only after medium fired', () => {
    const c = new TieredNudgeController()
    // Try to skip levels: soft+medium+hard all true at idle.
    const r = c.update({ soft: true, medium: true, hard: true }, 0.95, 1, [])
    // Still only one level advance per step.
    assert.equal(r.level, 0)
    assert.equal(r.fire, true)
    // Next step: advance to 1
    const r2 = c.update({ soft: true, medium: true, hard: true }, 0.95, 2, [])
    assert.equal(r2.level, 1)
    // Next step: advance to 2
    const r3 = c.update({ soft: true, medium: true, hard: true }, 0.95, 3, [])
    assert.equal(r3.level, 2)
    // Next step: stay at 2
    const r4 = c.update({ soft: true, medium: true, hard: true }, 0.95, 4, [])
    assert.equal(r4.level, 2)
    assert.equal(r4.fire, false)
  })

  test('reset on soft collapse clears even from hard', () => {
    const c = new TieredNudgeController()
    c.update({ soft: true, medium: true, hard: true }, 0.95, 1, [])
    c.update({ soft: true, medium: true, hard: true }, 0.95, 2, [])
    c.update({ soft: true, medium: true, hard: true }, 0.95, 3, [])
    assert.equal(c.level, 2)
    const r = c.update({ soft: false, medium: true, hard: true }, 0.05, 4, [])
    assert.equal(r.level, -1)
    assert.equal(r.fire, false)
  })

  test('after reset, escalation starts again from soft', () => {
    const c = new TieredNudgeController()
    c.update({ soft: true, medium: true, hard: true }, 0.95, 1, [])
    c.update({ soft: false, medium: false, hard: false }, 0.05, 2, [])
    const r = c.update({ soft: true, medium: true, hard: true }, 0.95, 3, [])
    assert.equal(r.level, 0) // soft, not medium
    assert.equal(r.fire, true)
  })

  test('nudge text contains the turn number and confidence percent', () => {
    const c = new TieredNudgeController()
    const r = c.update({ soft: true, medium: false, hard: false }, 0.42, 17, ['Bash: ls'])
    assert.match(r.text, /turn 17/)
    assert.match(r.text, /42/) // 42% from score 0.42
    assert.match(r.text, /Bash: ls/)
  })

  test('default tiered config matches the production parameters', () => {
    assert.equal(DEFAULT_TIERED_CONFIG.soft.kind, 'mean')
    assert.equal(DEFAULT_TIERED_CONFIG.soft.n, 2)
    assert.equal(DEFAULT_TIERED_CONFIG.soft.threshold, 0.34)
    assert.equal(DEFAULT_TIERED_CONFIG.medium.kind, 'median')
    assert.equal(DEFAULT_TIERED_CONFIG.medium.n, 4)
    assert.equal(DEFAULT_TIERED_CONFIG.medium.threshold, 0.645)
    assert.equal(DEFAULT_TIERED_CONFIG.hard.kind, 'median')
    assert.equal(DEFAULT_TIERED_CONFIG.hard.n, 9)
    assert.equal(DEFAULT_TIERED_CONFIG.hard.threshold, 0.605)
  })
})
