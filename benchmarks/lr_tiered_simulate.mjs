#!/usr/bin/env node
/**
 * Replay the OOD benchmark sessions through the LR + three-tier filter +
 * tiered nudge controller. Reads per-step LR scores precomputed by
 * benchmarks/lr_export_weights.py (so the JS side owns the filter state
 * machine and nudge logic; the feature pipeline stays in Python until
 * we port it to JS).
 *
 * Usage:
 *   node benchmarks/lr_tiered_simulate.mjs
 *   node benchmarks/lr_tiered_simulate.mjs --session bench_03_llvm_loop_vec --trace
 */
import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { TieredFilter, TieredNudgeController } from '../proxy/tiered_filter.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))
const REPO = resolve(__dirname, '..')

function parseArgs(argv) {
  const out = { session: null, trace: false, json: false }
  const a = argv.slice(2)
  for (let i = 0; i < a.length; i++) {
    const arg = a[i]
    if (arg === '--session') out.session = a[++i]
    else if (arg === '--trace') out.trace = true
    else if (arg === '--json') out.json = true
    else if (arg.startsWith('--')) {
      console.error('unknown flag:', arg); process.exit(2)
    }
  }
  return out
}

const args = parseArgs(process.argv)

const data = JSON.parse(
  readFileSync(resolve(REPO, 'benchmarks/results/lr_scores_ood.json'), 'utf8')
)

const LEVEL_NAMES = { [-1]: 'idle', 0: 'soft', 1: 'medium', 2: 'hard' }

// ── Per-session replay ─────────────────────────────────────────────────
function replay(session) {
  const filter = new TieredFilter()
  const nudge = new TieredNudgeController()
  const events = []
  let stuckSteps = 0 // label = STUCK
  let productiveSteps = 0 // label = PRODUCTIVE (UNSURE excluded)
  const fires = { soft: 0, medium: 0, hard: 0 }
  const resets = []
  let prevNudgeLevel = -1
  for (const s of session.steps) {
    if (s.label === 1.0) stuckSteps++
    else if (s.label === 0.0) productiveSteps++
    const f = filter.update(s.score)
    const d = nudge.update(f)
    if (d.prevLevel > -1 && d.level === -1) resets.push(s.step)
    if (d.fire) fires[LEVEL_NAMES[d.level]]++
    events.push({
      step: s.step,
      label: s.label,
      score: +s.score.toFixed(4),
      soft: f.soft, medium: f.medium, hard: f.hard,
      mean2: f.aggregates.soft  === null ? null : +f.aggregates.soft.toFixed(4),
      med4:  f.aggregates.medium === null ? null : +f.aggregates.medium.toFixed(4),
      med9:  f.aggregates.hard === null ? null : +f.aggregates.hard.toFixed(4),
      fire: d.fire ? LEVEL_NAMES[d.level] : null,
      level: d.level,
    })
    prevNudgeLevel = d.level
  }

  // Episode-level summary: contiguous stuck runs, whether nudged inside
  const episodes = []
  let i = 0
  const labels = session.steps.map((s) => s.label)
  while (i < labels.length) {
    if (labels[i] === 1.0) {
      let j = i
      while (j < labels.length && labels[j] === 1.0) j++
      const firesInside = events.slice(i, j).filter((e) => e.fire !== null)
      episodes.push({
        start: session.steps[i].step,
        end: session.steps[j - 1].step,
        length: j - i,
        firedInside: firesInside.map((e) => e.fire),
      })
      i = j
    } else i++
  }

  return {
    session_id: session.session_id,
    n_steps: session.steps.length,
    stuckSteps,
    productiveSteps,
    fires,
    resets,
    episodes,
    events,
  }
}

// ── Run ─────────────────────────────────────────────────────────────────
const results = []
for (const sess of data.sessions) {
  if (args.session && sess.session_id !== args.session) continue
  results.push(replay(sess))
}

if (args.json) {
  process.stdout.write(JSON.stringify(results, null, 2) + '\n')
  process.exit(0)
}

// ── Pretty summary ─────────────────────────────────────────────────────
const fmt = (n, w = 3) => String(n).padStart(w)
const pad = (s, w) => String(s).padEnd(w)
console.log('=== LR + 3-tier filter benchmark replay (OOD) ===')
console.log(`tiers: soft=mean-of-2@0.34  medium=median-of-4@0.645  hard=median-of-9@0.605`)
console.log()
console.log(`${pad('session', 26)} ${pad('n', 4)} ${pad('stk', 4)} ${pad('epi', 4)}`
  + `   ${pad('soft', 4)} ${pad('med', 4)} ${pad('hard', 4)} ${pad('resets', 7)}   episodes`)
console.log('-'.repeat(100))
let tot = { n: 0, stk: 0, soft: 0, med: 0, hard: 0, epi: 0, epi_nudged: 0 }
for (const r of results) {
  const nEpi = r.episodes.length
  const epiNudged = r.episodes.filter((e) => e.firedInside.length > 0).length
  const epSummary = r.episodes.length
    ? r.episodes
        .map((e) => {
          const fires = e.firedInside.length ? e.firedInside.join('>') : '-'
          return `[${e.start}..${e.end}:${e.length}→${fires}]`
        })
        .join(' ')
    : ''
  console.log(
    `${pad(r.session_id, 26)} ${fmt(r.n_steps, 4)} ${fmt(r.stuckSteps, 4)} ${fmt(nEpi, 4)}`
      + `   ${fmt(r.fires.soft, 4)} ${fmt(r.fires.medium, 4)} ${fmt(r.fires.hard, 4)} `
      + `${fmt(r.resets.length, 7)}   ${epSummary}`
  )
  tot.n += r.n_steps
  tot.stk += r.stuckSteps
  tot.soft += r.fires.soft
  tot.med += r.fires.medium
  tot.hard += r.fires.hard
  tot.epi += nEpi
  tot.epi_nudged += epiNudged
}
console.log('-'.repeat(100))
console.log(
  `${pad('TOTAL', 26)} ${fmt(tot.n, 4)} ${fmt(tot.stk, 4)} ${fmt(tot.epi, 4)}`
    + `   ${fmt(tot.soft, 4)} ${fmt(tot.med, 4)} ${fmt(tot.hard, 4)}`
)
console.log()
console.log(`stuck episodes nudged at least once: ${tot.epi_nudged}/${tot.epi}`)

// ── Trace a single session when requested ─────────────────────────────
if (args.trace && results.length === 1) {
  const r = results[0]
  console.log()
  console.log(`--- trace: ${r.session_id} ---`)
  console.log(`${pad('step', 4)} ${pad('lbl', 4)} ${pad('score', 7)}`
    + ` ${pad('mean2', 7)} ${pad('med4', 7)} ${pad('med9', 7)}`
    + ` ${pad('flt', 10)} ${pad('nudge', 8)}`)
  for (const e of r.events) {
    const lbl = e.label === 1.0 ? 'STK' : e.label === 0.0 ? 'pro' : 'unk'
    const flt = [
      e.soft ? 's' : '-',
      e.medium ? 'm' : '-',
      e.hard ? 'h' : '-',
    ].join('')
    const fire = e.fire ? `→${e.fire}` : ''
    console.log(
      `${fmt(e.step, 4)} ${pad(lbl, 4)} ${pad(e.score.toFixed(3), 7)}`
        + ` ${pad(e.mean2 ?? '-', 7)} ${pad(e.med4 ?? '-', 7)} ${pad(e.med9 ?? '-', 7)}`
        + ` ${pad(flt, 10)} ${pad(fire, 8)}`
    )
  }
}
