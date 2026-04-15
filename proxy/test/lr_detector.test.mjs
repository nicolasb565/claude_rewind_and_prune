import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { LRSessionDetector } from '../lr_detector.mjs'
import { loadLR } from '../lr.mjs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const lr = loadLR(resolve(__dirname, '..', 'lr_weights.json'))

describe('LRSessionDetector', () => {
  test('addStep returns score, filters and aggregates', () => {
    const det = new LRSessionDetector(lr)
    const r = det.addStep('Bash', { command: 'ls' }, 'foo\nbar')
    assert.ok(typeof r.score === 'number')
    assert.ok(r.score >= 0 && r.score <= 1)
    assert.ok('soft' in r.filters)
    assert.ok('medium' in r.filters)
    assert.ok('hard' in r.filters)
    assert.ok('aggregates' in r)
  })

  test('first step: filters off (insufficient buffer)', () => {
    const det = new LRSessionDetector(lr)
    const r = det.addStep('Bash', { command: 'ls' }, 'foo')
    assert.equal(r.filters.soft, false)
    assert.equal(r.filters.medium, false)
    assert.equal(r.filters.hard, false)
    assert.equal(r.aggregates.soft, null)
  })

  test('repeated identical bash command escalates filter signal', () => {
    const det = new LRSessionDetector(lr)
    const cmd = { command: 'gcc -O2 -c src/main.c' }
    const out = 'src/main.c:42: error: undefined reference to foo'
    let r
    for (let i = 0; i < 9; i++) r = det.addStep('Bash', cmd, out)
    // After 9 repeated stuck steps, expect the soft filter to have latched on.
    // (We don't assert hard here because it depends on LR magnitude, which is
    // the operating point of the production weights; soft is the cleanest
    // bound to assert.)
    assert.ok(r.score > 0.3, `expected high score after 9 repeats, got ${r.score}`)
    assert.equal(r.filters.soft, true)
  })

  test('stepCount tracks calls', () => {
    const det = new LRSessionDetector(lr)
    det.addStep('Bash', { command: 'a' }, '')
    det.addStep('Bash', { command: 'b' }, '')
    assert.equal(det.stepCount, 2)
  })

  test('handles tool_name correctly so Grep ≠ Glob', () => {
    // If parseToolCall didn't include tool_name, both would map to action
    // key "search|search" and incorrectly match each other. The LR was
    // trained with these as distinct actions.
    const det = new LRSessionDetector(lr)
    det.addStep('Grep', { pattern: 'foo' }, 'no matches')
    const r = det.addStep('Glob', { pattern: '**/*.py' }, 'a.py\nb.py')
    // The Glob step should NOT see the Grep as a matching prior.
    assert.ok(r.score < 0.5, `expected low (productive) score, got ${r.score}`)
  })
})
