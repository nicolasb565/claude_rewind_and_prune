import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import {
  parseToolCall,
  computeFeatures,
  cmdSemanticKey,
  jaccard,
  normalizeToSet,
  TOOL_TO_IDX,
} from '../features.mjs'

describe('parseToolCall', () => {
  test('maps Bash → bash, extracts cmd and output', () => {
    const s = parseToolCall('Bash', { command: 'ls -la' }, 'total 8')
    assert.equal(s.tool, 'bash')
    assert.equal(s.cmd, 'ls -la')
    assert.equal(s.file, null)
    assert.equal(s.output, 'total 8')
  })

  test('maps Edit → edit, extracts file', () => {
    const s = parseToolCall('Edit', { file_path: 'src/foo.py' }, 'OK')
    assert.equal(s.tool, 'edit')
    assert.equal(s.file, 'src/foo.py')
  })

  test('maps Write → edit', () => {
    assert.equal(parseToolCall('Write', {}, '').tool, 'edit')
  })

  test('maps MultiEdit → edit', () => {
    assert.equal(parseToolCall('MultiEdit', {}, '').tool, 'edit')
  })

  test('maps Read → view', () => {
    assert.equal(parseToolCall('Read', { file_path: 'README.md' }, '').tool, 'view')
  })

  test('maps Grep → search', () => {
    assert.equal(parseToolCall('Grep', { pattern: 'foo' }, '').tool, 'search')
  })

  test('maps Glob → search', () => {
    assert.equal(parseToolCall('Glob', { pattern: '**/*.py' }, '').tool, 'search')
  })

  test('uses description fallback for Task/Agent tools', () => {
    const s = parseToolCall('Task', { description: 'run the build' }, '')
    assert.equal(s.cmd, 'run the build')
  })

  test('uses prompt fallback when description absent', () => {
    const s = parseToolCall('Agent', { prompt: 'analyze the codebase' }, '')
    assert.equal(s.cmd, 'analyze the codebase')
  })

  test('truncates description/prompt fallback to 200 chars', () => {
    const long = 'x'.repeat(300)
    const s = parseToolCall('Task', { description: long }, '')
    assert.equal(s.cmd.length, 200)
  })

  test('does not truncate regular bash command', () => {
    const long = 'a'.repeat(300)
    const s = parseToolCall('Bash', { command: long }, '')
    assert.equal(s.cmd.length, 300)
  })

  test('does not truncate long file path', () => {
    const long = '/very/long/' + 'a'.repeat(300) + '.py'
    const s = parseToolCall('Read', { file_path: long }, '')
    assert.equal(s.cmd, long)
  })

  test('falls back to description when command is empty string', () => {
    const s = parseToolCall('Task', { command: '', description: 'fallback text' }, '')
    assert.equal(s.cmd, 'fallback text')
  })

  test('falls back to description when command is null', () => {
    const s = parseToolCall('Task', { command: null, description: 'fallback text' }, '')
    assert.equal(s.cmd, 'fallback text')
  })

  test('unknown tool maps to other', () => {
    assert.equal(parseToolCall('UnknownTool', {}, '').tool, 'other')
  })

  test('null input does not throw', () => {
    const s = parseToolCall('Bash', null, 'out')
    assert.equal(s.cmd, '')
    assert.equal(s.file, null)
    assert.equal(s.output, 'out')
  })

  test('undefined output defaults to empty string', () => {
    const s = parseToolCall('Bash', { command: 'ls' }, undefined)
    assert.equal(s.output, '')
  })
})

describe('cmdSemanticKey', () => {
  test('extracts bare command', () => {
    assert.equal(cmdSemanticKey('ls -la'), 'ls')
  })

  test('extracts base:target when file argument present', () => {
    assert.equal(cmdSemanticKey('gcc -O2 test.c'), 'gcc:test.c')
  })

  test('strips leading path from binary', () => {
    assert.equal(cmdSemanticKey('/usr/bin/gcc test.c'), 'gcc:test.c')
  })

  test('skips silent commands, uses first real command', () => {
    assert.equal(cmdSemanticKey('cd /tmp && ls'), 'ls')
  })

  test('handles pipe — uses first segment', () => {
    assert.equal(cmdSemanticKey('ls -la | grep foo'), 'ls')
  })

  test('empty string returns empty string', () => {
    assert.equal(cmdSemanticKey(''), '')
  })
})

describe('computeFeatures', () => {
  test('returns Float32Array of length 7', () => {
    const feats = computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: '' }, new Map())
    assert.ok(feats instanceof Float32Array)
    assert.equal(feats.length, 7)
  })

  test('tool_idx [0] matches TOOL_TO_IDX for each tool', () => {
    for (const [tool, idx] of Object.entries(TOOL_TO_IDX)) {
      const feats = computeFeatures({ tool, cmd: '', file: null, output: '' }, new Map())
      assert.equal(feats[0], idx, `tool=${tool}`)
    }
  })

  test('output_length [5] is 0 for empty output', () => {
    const feats = computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: '' }, new Map())
    assert.equal(feats[5], 0)
  })

  test('output_length [5] is log1p(newline_count)', () => {
    const output = 'a\nb\nc' // 2 newlines
    const feats = computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output }, new Map())
    assert.ok(Math.abs(feats[5] - Math.log1p(2)) < 1e-6)
  })

  test('is_error [6] is 1 for error output', () => {
    const feats = computeFeatures(
      { tool: 'bash', cmd: 'ls', file: null, output: 'Error: file not found' },
      new Map(),
    )
    assert.equal(feats[6], 1.0)
  })

  test('is_error [6] is 0 for clean output', () => {
    const feats = computeFeatures(
      { tool: 'bash', cmd: 'ls', file: null, output: 'hello world' },
      new Map(),
    )
    assert.equal(feats[6], 0.0)
  })

  test('has_prior_output [4] is 0 on first call', () => {
    const feats = computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: 'hi' }, new Map())
    assert.equal(feats[4], 0.0)
  })

  test('has_prior_output [4] is 1 after same command run again', () => {
    const history = new Map()
    computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: 'hi' }, history)
    const feats2 = computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: 'hi' }, history)
    assert.equal(feats2[4], 1.0)
  })

  test('output_similarity [3] is 1.0 for identical output repeated', () => {
    const history = new Map()
    computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: 'foo\nbar' }, history)
    const feats2 = computeFeatures(
      { tool: 'bash', cmd: 'ls', file: null, output: 'foo\nbar' },
      history,
    )
    assert.equal(feats2[3], 1.0)
  })

  test('output_similarity [3] is 0.0 for completely different output', () => {
    const history = new Map()
    computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: 'foo' }, history)
    const feats2 = computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: 'bar' }, history)
    assert.equal(feats2[3], 0.0)
  })

  test('edit tool has output_similarity=0 and has_prior=0 even after repeat', () => {
    const history = new Map()
    computeFeatures({ tool: 'edit', cmd: 'foo.py', file: 'foo.py', output: 'ok' }, history)
    const feats2 = computeFeatures(
      { tool: 'edit', cmd: 'foo.py', file: 'foo.py', output: 'ok' },
      history,
    )
    assert.equal(feats2[3], 0.0) // output_similarity
    assert.equal(feats2[4], 0.0) // has_prior_output
  })

  test('cmd_hash [1] is in [0, 1)', () => {
    const feats = computeFeatures(
      { tool: 'bash', cmd: 'ls -la', file: null, output: '' },
      new Map(),
    )
    assert.ok(feats[1] >= 0 && feats[1] < 1)
  })

  test('file_hash [2] is in [0, 1)', () => {
    const feats = computeFeatures(
      { tool: 'view', cmd: '', file: 'src/main.py', output: '' },
      new Map(),
    )
    assert.ok(feats[2] >= 0 && feats[2] < 1)
  })

  test('file_hash [2] is 0 for null file', () => {
    const feats = computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: '' }, new Map())
    assert.equal(feats[2], 0.0)
  })

  test('same command produces same cmd_hash across calls', () => {
    const feats1 = computeFeatures(
      { tool: 'bash', cmd: 'make test', file: null, output: '' },
      new Map(),
    )
    const feats2 = computeFeatures(
      { tool: 'bash', cmd: 'make test', file: null, output: '' },
      new Map(),
    )
    assert.equal(feats1[1], feats2[1])
  })
})

describe('jaccard', () => {
  test('returns 0 for null prior (no prior output)', () => {
    assert.equal(jaccard(new Set(['a']), null), 0.0)
  })

  test('returns 1 for two empty sets', () => {
    assert.equal(jaccard(new Set(), new Set()), 1.0)
  })

  test('returns 1 for identical sets', () => {
    assert.equal(jaccard(new Set(['a', 'b']), new Set(['a', 'b'])), 1.0)
  })

  test('returns 0 for disjoint sets', () => {
    assert.equal(jaccard(new Set(['a']), new Set(['b'])), 0.0)
  })

  test('returns 1/3 for one-of-three overlap', () => {
    // {a,b} ∩ {b,c} = {b}, union = {a,b,c} → 1/3
    assert.ok(Math.abs(jaccard(new Set(['a', 'b']), new Set(['b', 'c'])) - 1 / 3) < 1e-9)
  })

  test('returns 0.5 when one set is a subset of size half', () => {
    // {a,b} ∩ {a} = {a}, union = {a,b} → 1/2
    assert.equal(jaccard(new Set(['a', 'b']), new Set(['a'])), 0.5)
  })
})

describe('normalizeToSet', () => {
  test('returns empty set for empty input', () => {
    assert.equal(normalizeToSet('').size, 0)
  })

  test('normalizes hex addresses', () => {
    const s = normalizeToSet('addr=0xdeadbeef')
    assert.ok(s.has('addr=0xADDR'))
  })

  test('normalizes timestamps', () => {
    const s = normalizeToSet('2024-01-15 12:34:56 event')
    assert.ok(s.has('TIMESTAMP event'))
  })

  test('deduplicates identical lines', () => {
    const s = normalizeToSet('foo\nfoo\nfoo')
    assert.equal(s.size, 1)
  })
})
