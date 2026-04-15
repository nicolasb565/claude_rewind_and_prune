#!/usr/bin/env node
/**
 * Offline replay: feed a benchmark transcript through the LR detector +
 * three-tier filter + tiered nudge controller and emit what the proxy
 * WOULD have done, without making any API calls.
 *
 * Usage:
 *   node proxy/simulate.mjs <transcript.jsonl> [--json] [--trace]
 *
 * What it cannot tell you:
 *   - How the agent would have RESPONDED to different nudges. The
 *     transcript is frozen history from the original run. Feature
 *     extraction and classifier scoring are deterministic against it,
 *     but any nudge-induced behaviour change is out of reach without a
 *     real re-run. This tool answers "when would the proxy fire?" not
 *     "did firing help?".
 */

import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { loadLR } from './lr.mjs'
import { LRSessionDetector } from './lr_detector.mjs'
import { TieredNudgeController, DEFAULT_TIERED_CONFIG } from './tiered_filter.mjs'
import { extractAllToolCalls, recentToolSummary } from './messages.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))

// ── CLI ────────────────────────────────────────────────────────────────────
function parseArgs(argv) {
  const out = { transcript: null, json: false, trace: false }
  const a = argv.slice(2)
  for (let i = 0; i < a.length; i++) {
    const arg = a[i]
    if (arg === '--json') out.json = true
    else if (arg === '--trace') out.trace = true
    else if (arg.startsWith('--')) {
      console.error('unknown flag:', arg); process.exit(2)
    } else if (!out.transcript) out.transcript = arg
    else {
      console.error('unexpected arg:', arg); process.exit(2)
    }
  }
  if (!out.transcript) {
    console.error('usage: simulate.mjs <transcript.jsonl> [--json] [--trace]')
    process.exit(2)
  }
  return out
}

const args = parseArgs(process.argv)

const lr = loadLR(resolve(__dirname, 'lr_weights.json'))

// ── Transcript → messages array ────────────────────────────────────────────
// stream-json emits {type:"user"|"assistant",message:{role,content:[...]}}
// per line. Reassemble the API-shape messages array the proxy operates on.
const rawLines = readFileSync(args.transcript, 'utf8').split('\n').filter(Boolean)
const messages = []
for (const line of rawLines) {
  let ev
  try { ev = JSON.parse(line) } catch { continue }
  if (ev.type !== 'user' && ev.type !== 'assistant') continue
  if (!ev.message || typeof ev.message !== 'object') continue
  messages.push(ev.message)
}

// ── Simulation ─────────────────────────────────────────────────────────────
const detector = new LRSessionDetector(lr)
const nudge = new TieredNudgeController()
const LEVEL_NAMES = ['soft', 'medium', 'hard']

const toolCalls = extractAllToolCalls(messages)

const events = []
let turn = 0
for (const tc of toolCalls) {
  turn++
  const result = detector.addStep(tc.toolName, tc.input, tc.output)
  const recent = recentToolSummary(messages.slice(0, turn + 1))
  const decision = nudge.update(
    result.filters, result.score, turn, recent,
  )
  events.push({
    turn,
    tool: tc.toolName,
    score: +result.score.toFixed(4),
    soft: result.filters.soft,
    medium: result.filters.medium,
    hard: result.filters.hard,
    mean2: result.aggregates.soft == null ? null : +result.aggregates.soft.toFixed(4),
    med4: result.aggregates.medium == null ? null : +result.aggregates.medium.toFixed(4),
    med9: result.aggregates.hard == null ? null : +result.aggregates.hard.toFixed(4),
    fired: decision.fire,
    level: decision.level,
    nudgeTier: decision.fire ? LEVEL_NAMES[decision.level] : null,
  })
}

// ── Output ────────────────────────────────────────────────────────────────
const summary = {
  transcript: args.transcript,
  classifier: 'lr_content_v1',
  tiers: DEFAULT_TIERED_CONFIG,
  toolCalls: events.length,
  maxScore: events.reduce((m, e) => Math.max(m, e.score), 0),
  firedNudges: events.filter((e) => e.fired).length,
  firedByTier: {
    soft:   events.filter((e) => e.fired && e.level === 0).length,
    medium: events.filter((e) => e.fired && e.level === 1).length,
    hard:   events.filter((e) => e.fired && e.level === 2).length,
  },
}

if (args.json) {
  process.stdout.write(JSON.stringify({ summary, events }, null, 2) + '\n')
} else {
  const fmt = (n) => n.toFixed(3)
  console.log(`transcript:    ${args.transcript}`)
  console.log(`classifier:    ${summary.classifier}`)
  console.log(`tool calls:    ${summary.toolCalls}`)
  console.log(`max score:     ${fmt(summary.maxScore)}`)
  console.log(`fired nudges:  ${summary.firedNudges} `
    + `(soft=${summary.firedByTier.soft} `
    + `medium=${summary.firedByTier.medium} `
    + `hard=${summary.firedByTier.hard})`)
  if (args.trace) {
    console.log('--- per-turn trace ---')
    for (const e of events) {
      const flt = (e.soft ? 's' : '-') + (e.medium ? 'm' : '-') + (e.hard ? 'h' : '-')
      const fire = e.fired ? ` NUDGE[${e.nudgeTier}]` : ''
      console.log(
        `  turn=${String(e.turn).padStart(3)} ${e.tool.padEnd(8)} `
        + `score=${fmt(e.score)} flt=${flt}${fire}`,
      )
    }
  }
}
