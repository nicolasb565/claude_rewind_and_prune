import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { NudgeController } from '../nudge.mjs'

const THRESHOLD = 0.5

describe('NudgeController', () => {
  test('no fire when score is below threshold', () => {
    const n = new NudgeController({ threshold: THRESHOLD })
    assert.equal(n.update(0.3, 1).fire, false)
  })

  test('first detection above threshold is silent — no fire', () => {
    const n = new NudgeController({ threshold: THRESHOLD })
    const r = n.update(0.9, 1)
    assert.equal(r.fire, false) // level -1 absorbs first hit
    assert.equal(n._nudgeLevel, 0) // level bumped to 0
  })

  test('second detection fires at level 0', () => {
    const n = new NudgeController({ threshold: THRESHOLD, cooldowns: [1, 1, 1, 1] })
    n.update(0.9, 1) // silent → level 0
    const r = n.update(0.9, 2)
    assert.equal(r.fire, true)
    assert.equal(r.level, 0)
    assert.equal(n._nudgeLevel, 1)
  })

  test('level escalates 0 → 1 → 2', () => {
    const n = new NudgeController({ threshold: THRESHOLD, cooldowns: [1, 1, 1, 1] })
    n.update(0.9, 1) // silent → 0
    n.update(0.9, 2) // fire level 0 → 1
    n.update(0.9, 3) // fire level 1 → 2
    const r = n.update(0.9, 4)
    assert.equal(r.level, 2)
  })

  test('level caps at 2', () => {
    const n = new NudgeController({ threshold: THRESHOLD, cooldowns: [1, 1, 1, 1] })
    for (let i = 1; i <= 10; i++) n.update(0.9, i)
    assert.equal(n._nudgeLevel, 2)
  })

  test('no fire during cooldown window', () => {
    // cooldowns indexed by (level+1):
    //   level -1 (idx 0): 1 turn cooldown (fires next turn)
    //   level  0 (idx 1): 1 turn cooldown (fires next turn)
    //   level  1 (idx 2): 5 turn cooldown
    const n = new NudgeController({ threshold: THRESHOLD, cooldowns: [1, 1, 5, 8] })
    n.update(0.9, 1) // silent → level 0
    const r1 = n.update(0.9, 2) // fires level 0 → level 1, lastNudgeTurn=2
    assert.equal(r1.fire, true)
    // turns 3..6: (t - 2) = 1..4 < 5 → in cooldown
    for (let t = 3; t <= 6; t++) {
      assert.equal(n.update(0.9, t).fire, false, `should not fire at turn ${t}`)
    }
    // turn 7: 7 - 2 = 5 ≥ 5 → fires
    assert.equal(n.update(0.9, 7).fire, true)
  })

  test('score drop below reset threshold resets level to -1', () => {
    const n = new NudgeController({ threshold: THRESHOLD, cooldowns: [1, 1, 1, 1] })
    n.update(0.9, 1) // silent → 0
    n.update(0.9, 2) // fire → 1
    // Reset threshold = 0.5 * 0.94 = 0.47
    n.update(0.1, 3) // below reset threshold → reset to -1
    assert.equal(n._nudgeLevel, -1)
  })

  test('score drop below reset threshold resets level even during cooldown', () => {
    // cooldowns[1+1] = 8, so turns 3..9 are in cooldown after firing at turn 2
    const n = new NudgeController({ threshold: THRESHOLD, cooldowns: [1, 1, 8, 8] })
    n.update(0.9, 1) // silent → level 0
    n.update(0.9, 2) // fire → level 1, lastNudgeTurn=2
    // turn 3: cooldown window (3-2=1 < 8), but score is well below resetThreshold
    n.update(0.0, 3)
    assert.equal(n._nudgeLevel, -1) // must reset even inside cooldown
  })

  test('score between reset threshold and threshold does not reset level', () => {
    const n = new NudgeController({ threshold: THRESHOLD, cooldowns: [1, 1, 1, 1] })
    n.update(0.9, 1) // silent → 0
    n.update(0.9, 2) // fire → 1
    // Score in (0.47, 0.50): not stuck, but also not below reset threshold
    n.update(0.48, 3)
    assert.equal(n._nudgeLevel, 1) // level unchanged
  })

  test('nudge text contains confidence percentage', () => {
    const n = new NudgeController({ threshold: THRESHOLD, cooldowns: [1, 1, 1, 1] })
    n.update(0.9, 1)
    const { text } = n.update(0.9, 2)
    assert.ok(text.includes('90%'), `Expected "90%" in nudge text`)
  })

  test('nudge text contains turn number', () => {
    const n = new NudgeController({ threshold: THRESHOLD, cooldowns: [1, 1, 1, 1] })
    n.update(0.9, 1)
    const { text } = n.update(0.9, 42)
    assert.ok(text.includes('42'), `Expected turn number in nudge text`)
  })

  test('returned text is empty when not firing', () => {
    const n = new NudgeController({ threshold: THRESHOLD })
    assert.equal(n.update(0.1, 1).text, '')
  })

  test('level 0 nudge text differs from level 1', () => {
    const n = new NudgeController({ threshold: THRESHOLD, cooldowns: [1, 1, 1, 1] })
    n.update(0.9, 1) // silent
    const r0 = n.update(0.9, 2) // level 0
    const r1 = n.update(0.9, 3) // level 1
    assert.notEqual(r0.text, r1.text)
  })

  test('recent tools appear in nudge text', () => {
    const n = new NudgeController({ threshold: THRESHOLD, cooldowns: [1, 1, 1, 1] })
    n.update(0.9, 1)
    const { text } = n.update(0.9, 2, ['Bash: ls -la', 'Edit: src/foo.py'])
    assert.ok(text.includes('Bash: ls -la'))
  })
})
