import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { TieredFilter, TieredNudgeController, DEFAULT_TIERED_CONFIG } from '../tiered_filter.mjs'

// ── TieredFilter ──────────────────────────────────────────────────────────

describe('TieredFilter — warmup and aggregation (2-tier)', () => {
  test('both filters are off until each buffer is full', () => {
    const f = new TieredFilter()
    // median-of-4 needs 4 samples, median-of-9 needs 9. One push → nothing.
    let s = f.update(0.99)
    assert.equal(s.medium, false)
    assert.equal(s.hard, false)
    assert.equal(s.aggregates.medium, null)
    assert.equal(s.aggregates.hard, null)
    // Push 3 more 0.99s: medium ready and firing, hard still warming.
    for (let i = 0; i < 3; i++) s = f.update(0.99)
    assert.equal(s.medium, true)
    assert.equal(s.hard, false)
  })

  test('median-of-4 with [0.1, 0.9, 0.9, 0.9] → 0.9 ≥ 0.85', () => {
    const f = new TieredFilter()
    f.update(0.1); f.update(0.9); f.update(0.9)
    const s = f.update(0.9)
    // Sorted: [0.1, 0.9, 0.9, 0.9] → median = (0.9 + 0.9) / 2 = 0.9
    assert.ok(Math.abs(s.aggregates.medium - 0.9) < 1e-12)
    assert.equal(s.medium, true)
  })

  test('median-of-4 with [0.1, 0.1, 0.9, 0.9] → 0.5 < 0.85', () => {
    const f = new TieredFilter()
    f.update(0.1); f.update(0.1); f.update(0.9)
    const s = f.update(0.9)
    // Sorted: [0.1, 0.1, 0.9, 0.9] → median = 0.5
    assert.ok(Math.abs(s.aggregates.medium - 0.5) < 1e-12)
    assert.equal(s.medium, false)
  })

  test('hard filter fires after 9 sustained high scores', () => {
    const f = new TieredFilter()
    let s
    for (let i = 0; i < 9; i++) s = f.update(0.95)
    assert.equal(s.medium, true)
    assert.equal(s.hard, true)
  })

  test('single low score breaks median-of-4 momentarily but not med-of-9', () => {
    const f = new TieredFilter()
    for (let i = 0; i < 9; i++) f.update(0.95)
    const s = f.update(0.05)
    // median-of-4 over [0.95, 0.95, 0.95, 0.05] = (0.95 + 0.95) / 2 = 0.95
    // still fires! median is robust to a single low value in N=4.
    assert.equal(s.medium, true)
    // But two consecutive low scores would drop the median.
    const s2 = f.update(0.05)
    assert.equal(s2.medium, false) // median of [0.95, 0.95, 0.05, 0.05] = 0.5
  })

  test('reset clears both buffers — filters go back to warmup', () => {
    const f = new TieredFilter()
    for (let i = 0; i < 10; i++) f.update(0.95)
    assert.equal(f.update(0.95).hard, true)
    f.reset()
    const s = f.update(0.95)
    assert.equal(s.medium, false)
    assert.equal(s.hard, false)
    assert.equal(s.aggregates.medium, null)
  })

  test('config override is respected', () => {
    const f = new TieredFilter({
      medium: { kind: 'mean', n: 1, threshold: 0.5 },
      hard:   { kind: 'mean', n: 1, threshold: 0.9 },
    })
    let s = f.update(0.95)
    assert.equal(s.medium, true)
    assert.equal(s.hard, true)
    s = f.update(0.6)
    assert.equal(s.medium, true)
    assert.equal(s.hard, false)
  })
})

// ── TieredNudgeController ────────────────────────────────────────────────

describe('TieredNudgeController — 2-tier state machine', () => {
  test('starts at idle (-1)', () => {
    const c = new TieredNudgeController()
    assert.equal(c.level, -1)
  })

  test('medium-only filter advances level -1 → 0 with fire', () => {
    const c = new TieredNudgeController()
    const r = c.update({ medium: true, hard: false }, 0.9, 1, [])
    assert.equal(r.fire, true)
    assert.equal(r.level, 0)
    assert.equal(r.prevLevel, -1)
    assert.ok(r.text.length > 0)
  })

  test('medium filter off resets to -1, no fire', () => {
    const c = new TieredNudgeController()
    c.update({ medium: true, hard: false }, 0.9, 1, [])
    const r = c.update({ medium: false, hard: false }, 0.1, 2, [])
    assert.equal(r.fire, false)
    assert.equal(r.level, -1)
    assert.equal(r.prevLevel, 0)
  })

  test('does NOT advance to hard without hard filter agreeing', () => {
    const c = new TieredNudgeController()
    c.update({ medium: true, hard: false }, 0.9, 1, [])
    const r = c.update({ medium: true, hard: false }, 0.9, 2, [])
    assert.equal(r.fire, false)
    assert.equal(r.level, 0)
  })

  test('hard filter advances level 0 → 1', () => {
    const c = new TieredNudgeController()
    c.update({ medium: true, hard: false }, 0.9, 1, [])
    const r = c.update({ medium: true, hard: true }, 0.95, 2, [])
    assert.equal(r.fire, true)
    assert.equal(r.level, 1)
  })

  test('one level advance per step even when both filters fire at once', () => {
    const c = new TieredNudgeController()
    // Initial state: both filters true, but we can only advance -1 → 0
    const r = c.update({ medium: true, hard: true }, 0.95, 1, [])
    assert.equal(r.level, 0)
    assert.equal(r.fire, true)
    // Next step: advance to 1
    const r2 = c.update({ medium: true, hard: true }, 0.95, 2, [])
    assert.equal(r2.level, 1)
    assert.equal(r2.fire, true)
    // Already at max; stay
    const r3 = c.update({ medium: true, hard: true }, 0.95, 3, [])
    assert.equal(r3.level, 1)
    assert.equal(r3.fire, false)
  })

  test('reset on medium collapse clears even from hard', () => {
    const c = new TieredNudgeController()
    c.update({ medium: true, hard: true }, 0.95, 1, [])
    c.update({ medium: true, hard: true }, 0.95, 2, [])
    assert.equal(c.level, 1)
    const r = c.update({ medium: false, hard: true }, 0.1, 3, [])
    assert.equal(r.level, -1)
    assert.equal(r.fire, false)
  })

  test('after reset, escalation starts again from medium', () => {
    const c = new TieredNudgeController()
    c.update({ medium: true, hard: true }, 0.95, 1, [])
    c.update({ medium: false, hard: false }, 0.1, 2, [])
    const r = c.update({ medium: true, hard: true }, 0.95, 3, [])
    assert.equal(r.level, 0)
    assert.equal(r.fire, true)
  })

  test('level 0 uses MEDIUM_NUDGE text (not soft)', () => {
    const c = new TieredNudgeController()
    const r = c.update({ medium: true, hard: false }, 0.9, 17, ['Bash: ls'])
    // The medium nudge text contains "repeated signal" as its unique marker.
    // Soft would say "may be going in circles".
    assert.match(r.text, /repeated signal/)
    assert.match(r.text, /turn 17/)
    assert.match(r.text, /90/) // 90% from score 0.9
    assert.match(r.text, /Bash: ls/)
  })

  test('level 1 uses HARD_NUDGE text (STOP directive)', () => {
    const c = new TieredNudgeController()
    c.update({ medium: true, hard: false }, 0.9, 1, [])
    const r = c.update({ medium: true, hard: true }, 0.95, 2, [])
    assert.match(r.text, /STOP/)
    assert.match(r.text, /escalated/)
  })

  test('default tiered config is medium=0.85 hard=0.85', () => {
    assert.equal(DEFAULT_TIERED_CONFIG.medium.kind, 'median')
    assert.equal(DEFAULT_TIERED_CONFIG.medium.n, 4)
    assert.equal(DEFAULT_TIERED_CONFIG.medium.threshold, 0.85)
    assert.equal(DEFAULT_TIERED_CONFIG.hard.kind, 'median')
    assert.equal(DEFAULT_TIERED_CONFIG.hard.n, 9)
    assert.equal(DEFAULT_TIERED_CONFIG.hard.threshold, 0.85)
    assert.equal('soft' in DEFAULT_TIERED_CONFIG, false)
  })
})
