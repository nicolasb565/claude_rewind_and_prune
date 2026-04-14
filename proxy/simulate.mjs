#!/usr/bin/env node
/**
 * Offline replay: feed a benchmark transcript through the v5 detector +
 * nudge controller and emit what the proxy WOULD have done for a given
 * nudge strategy, without making any API calls.
 *
 * Usage:
 *   node proxy/simulate.mjs <transcript.jsonl> [--threshold 0.5]
 *                                              [--reset-factor 0.94]
 *                                              [--cooldowns 1,4,8,8]
 *                                              [--json]
 *
 * Defaults match proxy/stuck_config.json. Override any of them to
 * evaluate a hypothetical alternate strategy (see
 * memory/project_next_steps.md — Strategy B suggests raising the silent
 * buffer and skipping the soft level, which translates to
 * `--cooldowns 5,5,8,8 --skip-soft`).
 *
 * What it cannot tell you:
 *   - How the agent would have RESPONDED to different nudges. The
 *     transcript is frozen history from the original run. Feature
 *     extraction and classifier scoring are deterministic against it,
 *     but any nudge-induced behaviour change is out of reach without a
 *     real re-run. This tool answers "when would the proxy fire?" not
 *     "did firing help?".
 *
 * Cross-validation:
 *   If the corresponding live run's proxy_events.jsonl is present in
 *   the same directory as the transcript, the simulator will compare
 *   its per-turn scores against the live `mlp_score` entries and flag
 *   any mismatch — this validates that the simulator and the live
 *   proxy agree on the classifier output.
 */

import { readFileSync, existsSync } from 'node:fs'
import { resolve, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { loadMLP } from './mlp.mjs'
import { SessionDetector } from './detector.mjs'
import { NudgeController } from './nudge.mjs'
import { extractAllToolCalls, recentToolSummary } from './messages.mjs'
import { parseToolCall, computeFeatures, cmdSemanticKey } from './features.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))

// ── CLI ────────────────────────────────────────────────────────────────────
function parseArgs(argv) {
  const out = {
    transcript: null,
    threshold: null,
    resetFactor: null,
    cooldowns: null,
    skipSoft: false,
    silentBuffer: 1,
    json: false,
  }
  const a = argv.slice(2)
  for (let i = 0; i < a.length; i++) {
    const arg = a[i]
    if (arg === '--threshold')       out.threshold    = parseFloat(a[++i])
    else if (arg === '--reset-factor') out.resetFactor = parseFloat(a[++i])
    else if (arg === '--cooldowns')  out.cooldowns    = a[++i].split(',').map(Number)
    else if (arg === '--skip-soft')  out.skipSoft     = true
    else if (arg === '--silent-buffer') out.silentBuffer = parseInt(a[++i], 10)
    else if (arg === '--strategy-b') out.strategyB    = true
    else if (arg === '--dump-features') out.dumpFeatures = true
    else if (arg === '--json')       out.json         = true
    else if (arg.startsWith('--'))   { console.error('unknown flag:', arg); process.exit(2) }
    else if (!out.transcript)        out.transcript   = arg
    else                             { console.error('unexpected arg:', arg); process.exit(2) }
  }
  if (!out.transcript) {
    console.error('usage: simulate.mjs <transcript.jsonl> [flags]')
    process.exit(2)
  }
  return out
}

const args = parseArgs(process.argv)

// Load classifier + config (same files the proxy uses)
const mlp = loadMLP(resolve(__dirname, 'stuck_weights.json'))
const config = JSON.parse(readFileSync(resolve(__dirname, 'stuck_config.json'), 'utf8'))
const threshold = args.threshold ?? config.threshold

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

// ── Strategy B controller ─────────────────────────────────────────────────
// Different state machine from the current NudgeController. Instead of
// absorbing exactly 1 stuck hit and then firing on the next one, Strategy B
// requires N *consecutive* stuck signals before firing, and skips the soft
// level entirely (first fire is medium). Once firing, escalates to hard on
// the next hit (subject to cooldown). A score drop below resetThreshold
// resets the consecutive counter and the fire level.
class StrategyBController {
  constructor({ threshold, resetFactor = 0.94, silentBuffer = 5, cooldowns = [8, 8] }) {
    this.threshold = threshold
    this.resetThreshold = threshold * resetFactor
    this.silentBuffer = silentBuffer
    this.cooldowns = cooldowns // indexed by (fireLevel): 0=medium, 1=hard
    this.consecutiveStuck = 0
    this.fireLevel = -1 // -1 = not firing yet, 1 = medium, 2 = hard
    this.lastFireTurn = Number.NEGATIVE_INFINITY
  }
  update(score, turn) {
    if (score < this.resetThreshold) {
      this.consecutiveStuck = 0
      this.fireLevel = -1
      return { fire: false, level: this.fireLevel, text: '' }
    }
    if (score < this.threshold) {
      return { fire: false, level: this.fireLevel, text: '' }
    }
    this.consecutiveStuck++
    if (this.consecutiveStuck < this.silentBuffer) {
      return { fire: false, level: this.fireLevel, text: '' }
    }
    // eligible to fire at medium (first) or hard (escalated)
    const targetLevel = this.fireLevel < 0 ? 1 : Math.min(this.fireLevel + 1, 2)
    const cooldownIdx = this.fireLevel < 0 ? 0 : Math.min(this.fireLevel, 1)
    const cooldown = this.cooldowns[cooldownIdx] ?? 8
    if (turn - this.lastFireTurn < cooldown) {
      return { fire: false, level: this.fireLevel, text: '' }
    }
    this.lastFireTurn = turn
    this.fireLevel = targetLevel
    return { fire: true, level: this.fireLevel, text: `[strategy-B nudge level=${this.fireLevel}]` }
  }
}

// ── Simulation ─────────────────────────────────────────────────────────────
const detector = new SessionDetector(mlp)
const useStrategyB = args.strategyB === true
const nudge = useStrategyB
  ? new StrategyBController({
      threshold,
      resetFactor: args.resetFactor ?? 0.94,
      silentBuffer: args.silentBuffer ?? 5,
      cooldowns: args.cooldowns ?? [8, 8],
    })
  : new NudgeController({
      threshold,
      resetFactor: args.resetFactor ?? 0.94,
      cooldowns: args.cooldowns ?? [1, 4, 8, 8],
    })

if (!useStrategyB && args.skipSoft) nudge._nudgeLevel = 0

const toolCalls = extractAllToolCalls(messages)

// For --dump-features we need to peek at the feature vector before addStep
// mutates the detector's state. Run feature computation in parallel with the
// detector so both see the same session state but we can inspect features.
const featureOutputHistory = new Map()

const events = []
let turn = 0
for (const tc of toolCalls) {
  turn++
  // Compute features for inspection before we commit to the detector
  const step = parseToolCall(tc.toolName, tc.input, tc.output)
  const cmdKey = step.tool === 'bash' ? cmdSemanticKey(step.cmd) : `${step.tool}:${step.cmd}`
  const features = computeFeatures(step, featureOutputHistory)
  const score = detector.addStep(tc.toolName, tc.input, tc.output)
  const recent = recentToolSummary(messages.slice(0, turn + 1))
  const decision = useStrategyB
    ? nudge.update(score, turn)
    : nudge.update(score, turn, recent)
  const ev = {
    turn,
    tool: tc.toolName,
    score: +score.toFixed(4),
    stuck: score >= threshold,
    fired: decision.fire,
    level: decision.level,
    nudgeText: decision.fire ? decision.text.split('\n')[0] : null,
  }
  if (args.dumpFeatures) {
    ev.features = {
      tool_idx:          features[0],
      cmd_hash:          +features[1].toFixed(6),
      file_hash:         +features[2].toFixed(6),
      output_similarity: +features[3].toFixed(4),
      has_prior_output:  features[4],
      output_length:     +features[5].toFixed(4),
      is_error:          features[6],
      cmd_semantic_key:  cmdKey ? cmdKey.slice(0, 80) : null,
      file:              step.file ? step.file.slice(0, 80) : null,
    }
  }
  events.push(ev)
}

// ── Optional cross-validation against a live run log ──────────────────────
const transcriptDir = dirname(resolve(args.transcript))
const eventsFile = join(transcriptDir, '..', 'proxy_logs') // may or may not exist
let crossCheck = { checked: 0, mismatches: 0, sample: [] }
// Walk up a couple of levels looking for proxy_logs/events-*.jsonl
for (const candidate of [
  join(transcriptDir, '..', 'proxy_logs'),
  join(transcriptDir, '..', '..', 'proxy_logs'),
]) {
  if (!existsSync(candidate)) continue
  // Find any events-*.jsonl file
  const fs = await import('node:fs/promises')
  const entries = await fs.readdir(candidate)
  const evFile = entries.find((f) => f.startsWith('events-') && f.endsWith('.jsonl'))
  if (!evFile) continue
  const live = readFileSync(join(candidate, evFile), 'utf8').split('\n').filter(Boolean)
  // Filter to this task via sessionKeyPrefix match against transcript's first user message
  let taskPrefix = ''
  for (const m of messages) {
    if (m.role === 'user' && Array.isArray(m.content)) {
      const text = m.content.map((b) => b.text ?? '').join('').slice(0, 64)
      if (text) { taskPrefix = text; break }
    }
  }
  const liveScores = []
  for (const l of live) {
    try {
      const ev = JSON.parse(l)
      if (ev.type === 'mlp_score' && ev.sessionKeyPrefix?.startsWith(taskPrefix)) {
        liveScores.push({ turn: ev.turn, score: ev.score })
      }
    } catch { /* skip */ }
  }
  if (liveScores.length === 0) break
  // Compare to simulated per-turn scores
  const simByTurn = new Map(events.map((e) => [e.turn, e.score]))
  for (const ls of liveScores) {
    crossCheck.checked++
    const sim = simByTurn.get(ls.turn)
    if (sim == null) continue
    if (Math.abs(sim - ls.score) > 0.002) {
      crossCheck.mismatches++
      if (crossCheck.sample.length < 5) {
        crossCheck.sample.push({ turn: ls.turn, live: ls.score, sim })
      }
    }
  }
  break
}

// ── Output ────────────────────────────────────────────────────────────────
const summary = {
  transcript: args.transcript,
  config: {
    threshold,
    resetFactor: args.resetFactor ?? 0.94,
    cooldowns: args.cooldowns ?? [1, 4, 8, 8],
    skipSoft: args.skipSoft,
  },
  toolCalls: events.length,
  stuckTurns: events.filter((e) => e.stuck).length,
  firedNudges: events.filter((e) => e.fired).length,
  firedByLevel: {
    soft:   events.filter((e) => e.fired && e.level === 0).length,
    medium: events.filter((e) => e.fired && e.level === 1).length,
    hard:   events.filter((e) => e.fired && e.level === 2).length,
  },
  maxScore: events.reduce((m, e) => Math.max(m, e.score), 0),
  crossCheck: crossCheck.checked > 0 ? crossCheck : null,
}

if (args.json) {
  process.stdout.write(JSON.stringify({ summary, events }, null, 2) + '\n')
} else {
  const fmt = (n) => n.toFixed(3)
  console.log(`transcript:    ${args.transcript}`)
  console.log(`tool calls:    ${summary.toolCalls}`)
  console.log(`max score:     ${fmt(summary.maxScore)}`)
  console.log(`stuck turns:   ${summary.stuckTurns}`)
  console.log(`fired nudges:  ${summary.firedNudges} (soft=${summary.firedByLevel.soft} medium=${summary.firedByLevel.medium} hard=${summary.firedByLevel.hard})`)
  if (summary.crossCheck) {
    const cc = summary.crossCheck
    console.log(`cross-check:   ${cc.mismatches}/${cc.checked} mismatches vs live proxy log`)
    if (cc.sample.length) {
      for (const s of cc.sample) console.log(`  turn=${s.turn} live=${fmt(s.live)} sim=${fmt(s.sim)}`)
    }
  }
  console.log('--- per-turn trace ---')
  for (const e of events) {
    const flag = e.fired ? ` NUDGE[L${e.level}]` : e.stuck ? ' stuck' : ''
    console.log(`  turn=${String(e.turn).padStart(3)} ${e.tool.padEnd(8)} score=${fmt(e.score)}${flag}`)
  }
}
