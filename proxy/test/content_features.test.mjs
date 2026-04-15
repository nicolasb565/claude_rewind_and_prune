import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  ContentFeatureExtractor,
  LR_FEATURE_NAMES,
  v9ActionKey,
  v9TargetFile,
  v9NormalizeToSet,
  hasErrorIndicators,
  stripSystemReminders,
} from '../content_features.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))
const REPO = resolve(__dirname, '..', '..')

// ── Pure helpers ──────────────────────────────────────────────────────────

describe('v9ActionKey', () => {
  test('non-bash uses (tool, tool_name)', () => {
    assert.equal(v9ActionKey({ tool: 'view', tool_name: 'Read', cmd: '' }), 'view|Read')
    assert.equal(v9ActionKey({ tool: 'search', tool_name: 'Grep', cmd: '' }), 'search|Grep')
    assert.equal(v9ActionKey({ tool: 'search', tool_name: 'Glob', cmd: '' }), 'search|Glob')
    // Glob and Grep must NOT collide — the LR was trained assuming they're
    // distinct actions for repetition detection.
    assert.notEqual(
      v9ActionKey({ tool: 'search', tool_name: 'Grep', cmd: '' }),
      v9ActionKey({ tool: 'search', tool_name: 'Glob', cmd: '' }),
    )
  })

  test('bash falls back to program name and optional subcommand', () => {
    assert.equal(v9ActionKey({ tool: 'bash', cmd: 'ls -la' }), 'bash|ls')
    assert.equal(v9ActionKey({ tool: 'bash', cmd: '/usr/bin/git status' }), 'bash|git|status')
    assert.equal(v9ActionKey({ tool: 'bash', cmd: 'git commit -m "x"' }), 'bash|git|commit')
    assert.equal(v9ActionKey({ tool: 'bash', cmd: 'cd /tmp && ls' }), 'bash|ls')
  })

  test('inline-script bash collapses to program name', () => {
    assert.equal(v9ActionKey({ tool: 'bash', cmd: 'python -c "print(1)"' }), 'bash|python')
    assert.equal(v9ActionKey({ tool: 'bash', cmd: 'node --eval "1+1"' }), 'bash|node')
  })
})

describe('v9TargetFile', () => {
  test('uses step.file when present', () => {
    assert.equal(v9TargetFile({ tool: 'edit', file: 'src/foo.py', cmd: '' }), 'src/foo.py')
  })

  test('extracts path token from bash command', () => {
    assert.equal(v9TargetFile({ tool: 'bash', cmd: 'gcc -O2 src/main.c -o main' }), 'src/main.c')
  })

  test('non-bash without file returns cmd', () => {
    assert.equal(v9TargetFile({ tool: 'search', cmd: 'foobar' }), 'foobar')
  })
})

describe('v9NormalizeToSet', () => {
  test('strips hex, timestamps, pids, /tmp paths', () => {
    const set = v9NormalizeToSet('addr 0xdeadbeef\n2024-01-01T12:00:00 ok\npid=12345\n/tmp/abc/def')
    // Each line should be normalized so equivalent lines collapse.
    assert.ok(set.has('addr 0xADDR'))
    assert.ok(set.has('TIMESTAMP ok'))
    assert.ok(set.has('pid=PID'))
    assert.ok(set.has('/tmp/TMPFILE'))
  })

  test('does NOT squash long integers (different from err-line normalization)', () => {
    const set = v9NormalizeToSet('count: 12345')
    assert.ok(set.has('count: 12345'))
  })
})

describe('hasErrorIndicators', () => {
  test('detects common error words', () => {
    assert.equal(hasErrorIndicators('foo bar error: something'), true)
    assert.equal(hasErrorIndicators('Traceback (most recent call last):'), true)
    assert.equal(hasErrorIndicators('TypeError: cannot do thing'), true)
  })

  test('clean output has no error indicator', () => {
    assert.equal(hasErrorIndicators('hello world\n42'), false)
  })

  test('only scans first 2000 chars (matches Python behavior)', () => {
    const padding = 'x'.repeat(2100)
    assert.equal(hasErrorIndicators(padding + 'error'), false)
    assert.equal(hasErrorIndicators('error' + padding), true)
  })
})

describe('stripSystemReminders', () => {
  test('removes <system-reminder>...</system-reminder> blocks', () => {
    const t = 'before<system-reminder>internal</system-reminder>after'
    assert.equal(stripSystemReminders(t), 'beforeafter')
  })

  test('handles multi-line reminders and multiple blocks', () => {
    const t = 'a<system-reminder>x\ny\nz</system-reminder>b<system-reminder>q</system-reminder>c'
    assert.equal(stripSystemReminders(t), 'abc')
  })
})

// ── End-to-end stateful extractor ─────────────────────────────────────────

describe('ContentFeatureExtractor', () => {
  test('first step: no history → match_ratio_5=0, self_sim_max=0', () => {
    const ext = new ContentFeatureExtractor()
    const v = ext.addStep({ tool: 'bash', tool_name: 'Bash', cmd: 'ls', output: '' })
    assert.equal(v[0], 0) // match_ratio_5
    assert.equal(v[1], 0) // self_sim_max
    assert.equal(v[2], 0) // repeat_no_error (no prior match)
    assert.equal(v[3], 0) // cur_bash_and_match_ratio
  })

  test('repeated identical bash command builds match_ratio_5', () => {
    const ext = new ContentFeatureExtractor()
    const step = { tool: 'bash', tool_name: 'Bash', cmd: 'ls -la', output: 'foo\nbar' }
    const v1 = ext.addStep(step)
    const v2 = ext.addStep(step)
    const v3 = ext.addStep(step)
    assert.equal(v1[0], 0)        // first: 0 priors match
    assert.equal(v2[0], 1 / 5)    // second: 1 prior matches
    assert.equal(v3[0], 2 / 5)    // third: 2 priors match
    assert.equal(v3[3], 2 / 5)    // is_bash * match_ratio_5
  })

  test('changing the command resets the slot match (different action key)', () => {
    const ext = new ContentFeatureExtractor()
    ext.addStep({ tool: 'bash', tool_name: 'Bash', cmd: 'ls', output: '' })
    ext.addStep({ tool: 'bash', tool_name: 'Bash', cmd: 'pwd', output: '' })
    const v = ext.addStep({ tool: 'bash', tool_name: 'Bash', cmd: 'cat foo.txt', output: '' })
    // None of the priors match this new action key, so match_ratio_5=0.
    assert.equal(v[0], 0)
  })

  test('identical output on the same action raises self_sim_max one step later', () => {
    const ext = new ContentFeatureExtractor()
    const step = { tool: 'bash', tool_name: 'Bash', cmd: 'ls', output: 'one\ntwo\nthree' }
    // self_sim_max reads PRIORS' stored self_relative_sim values. The current
    // step computes its OWN self_sim against bucket history but writes it into
    // its info AFTER feature extraction, so the *next* step is the first one
    // that can see a prior with non-zero self_relative_sim.
    const v1 = ext.addStep(step)
    assert.equal(v1[1], 0) // no priors at all
    const v2 = ext.addStep(step)
    assert.equal(v2[1], 0) // step 1's stored selfRelativeSim was 0
    const v3 = ext.addStep(step)
    assert.equal(v3[1], 1.0) // step 2's stored selfRelativeSim was 1.0
  })

  test('repeat_no_error: consecutive same action with no error → 1', () => {
    const ext = new ContentFeatureExtractor()
    ext.addStep({ tool: 'bash', tool_name: 'Bash', cmd: 'ls', output: 'foo' })
    const v = ext.addStep({ tool: 'bash', tool_name: 'Bash', cmd: 'ls', output: 'foo' })
    assert.equal(v[2], 1.0)
  })

  test('repeat_no_error: same action but current step is error → 0', () => {
    const ext = new ContentFeatureExtractor()
    ext.addStep({ tool: 'bash', tool_name: 'Bash', cmd: 'ls', output: 'foo' })
    const v = ext.addStep({ tool: 'bash', tool_name: 'Bash', cmd: 'ls', output: 'error: nope' })
    assert.equal(v[2], 0.0)
  })

  test('output parity check against full Python reference', () => {
    // Cross-validate every step of every OOD session against the dump
    // produced by benchmarks/content_feature_parity.py.
    const ref = JSON.parse(readFileSync(
      resolve(REPO, 'data/generated/content_features_ood_python.json'), 'utf8'
    ))
    const TOL = 1e-6
    let mismatches = 0
    let totalSteps = 0
    for (const sess of ref.sessions) {
      const ext = new ContentFeatureExtractor()
      for (const s of sess.steps) {
        const v = ext.addStep(s.parsed)
        totalSteps++
        for (let i = 0; i < LR_FEATURE_NAMES.length; i++) {
          if (Math.abs(v[i] - s.features[LR_FEATURE_NAMES[i]]) > TOL) mismatches++
        }
      }
    }
    assert.ok(totalSteps > 0)
    assert.equal(mismatches, 0, `${mismatches} feature mismatches across ${totalSteps} steps`)
  })
})
