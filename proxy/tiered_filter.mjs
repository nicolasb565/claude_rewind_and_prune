/**
 * Two-tier filter state machine for per-step LR stuck scores.
 *
 * Design: see README Key Finding #8. Threshold sweep
 * (benchmarks/nudge_2tier_sweep.mjs) shows that a 2-tier state machine
 * with medium and hard both at 0.85 is the F1-optimal point on the
 * labeled OOD benchmark — soft nudges added noise without improving
 * episode recovery, so the soft tier was dropped entirely.
 *
 *   medium  →  median-of-4 @ 0.85   ("you appear stuck, diagnose" nudge)
 *   hard    →  median-of-9 @ 0.85   ("STOP, root-cause analysis" nudge)
 *
 * State machine (step-driven, no wall-clock):
 *   level -1  = idle
 *   level  0  = medium fired
 *   level  1  = hard fired
 *
 * Transitions:
 *   -1 → 0  when med4 >= medium.threshold (fire MEDIUM_NUDGE)
 *    0 → 1  when med9 >= hard.threshold   (fire HARD_NUDGE)
 *   reset to -1 whenever med4 drops below threshold (medium is the
 *   "stuck envelope"; its collapse defines recovery)
 *
 * Properties:
 *   - Medium cannot fire before step 4, hard cannot fire before step 9
 *   - One level advance per step (hard cannot skip medium)
 *   - A single step with med4 < threshold resets to idle
 */

import { NUDGE_TEMPLATES } from './nudge.mjs'

export const DEFAULT_TIERED_CONFIG = {
  medium: { kind: 'median', n: 4, threshold: 0.85 },
  hard:   { kind: 'median', n: 9, threshold: 0.85 },
}

// NUDGE_TEMPLATES layout: [SOFT, MEDIUM, HARD]. The 2-tier controller
// offsets by +1 so level 0 renders MEDIUM and level 1 renders HARD.
const NUDGE_LEVEL_TO_TEMPLATE_IDX = [1, 2]

function median(arr) {
  const s = Array.from(arr).sort((a, b) => a - b)
  const n = s.length
  const mid = n >> 1
  return n % 2 === 1 ? s[mid] : 0.5 * (s[mid - 1] + s[mid])
}

function mean(arr) {
  let s = 0
  for (const v of arr) s += v
  return s / arr.length
}

class RollingBuffer {
  constructor(n) {
    this._n = n
    this._buf = []
  }
  push(v) {
    this._buf.push(v)
    if (this._buf.length > this._n) this._buf.shift()
  }
  ready() { return this._buf.length >= this._n }
  values() { return this._buf }
  clear() { this._buf.length = 0 }
}

export class TieredFilter {
  /**
   * @param {{medium: FilterSpec, hard: FilterSpec}} config
   *   FilterSpec = { kind: 'mean'|'median', n: number, threshold: number }
   */
  constructor(config = DEFAULT_TIERED_CONFIG) {
    this._cfg = config
    this._med = new RollingBuffer(config.medium.n)
    this._hard = new RollingBuffer(config.hard.n)
  }

  _aggregate(kind, buf) {
    return kind === 'median' ? median(buf.values()) : mean(buf.values())
  }

  _filterFires(spec, buf) {
    if (!buf.ready()) return false
    return this._aggregate(spec.kind, buf) >= spec.threshold
  }

  /**
   * Push a new score and return the current filter state.
   *
   * @param {number} score  LR P(stuck) in [0, 1]
   * @returns {{medium: boolean, hard: boolean, aggregates: {medium: number|null, hard: number|null}}}
   */
  update(score) {
    this._med.push(score)
    this._hard.push(score)
    return {
      medium: this._filterFires(this._cfg.medium, this._med),
      hard: this._filterFires(this._cfg.hard, this._hard),
      aggregates: {
        medium: this._med.ready()
          ? this._aggregate(this._cfg.medium.kind, this._med) : null,
        hard: this._hard.ready()
          ? this._aggregate(this._cfg.hard.kind, this._hard) : null,
      },
    }
  }

  reset() {
    this._med.clear()
    this._hard.clear()
  }
}

export class TieredNudgeController {
  constructor() {
    this._level = -1
  }

  /**
   * Advance the state machine given a fresh filter reading.
   *
   * Returns `{ fire, level, prevLevel, text }`:
   *   - fire=true means a nudge for this level should be sent THIS step
   *   - level is the (possibly updated) current level after the step
   *   - prevLevel is the level before the step — useful for logging resets
   *   - text is the rendered nudge message when fire=true, '' otherwise
   *
   * Level → nudge template mapping:
   *   level 0 ⇒ MEDIUM_NUDGE (NUDGE_TEMPLATES[1])
   *   level 1 ⇒ HARD_NUDGE   (NUDGE_TEMPLATES[2])
   *
   * @param {{medium: boolean, hard: boolean}} filters
   * @param {number} score          last LR P(stuck) score, for nudge text
   * @param {number} turn           current session turn counter
   * @param {string[]} recentTools  recent tool call summaries
   */
  update(filters, score = 0, turn = 0, recentTools = []) {
    const prev = this._level

    // Recovery: medium-filter collapse clears all state. Medium is the
    // "stuck envelope" in the 2-tier shape — its fall is recovery.
    if (!filters.medium) {
      this._level = -1
      return { fire: false, level: this._level, prevLevel: prev, text: '' }
    }

    // Advance at most one level per step. Each advancement fires once.
    let fireLevel = -1
    if (this._level === -1) {
      this._level = 0
      fireLevel = 0
    } else if (this._level === 0 && filters.hard) {
      this._level = 1
      fireLevel = 1
    }

    if (fireLevel < 0) {
      return { fire: false, level: this._level, prevLevel: prev, text: '' }
    }

    const pct = Math.round(score * 100)
    const recentList = recentTools.join('\n  ')
    const template = NUDGE_TEMPLATES[NUDGE_LEVEL_TO_TEMPLATE_IDX[fireLevel]]
    const text = template(turn, pct, recentList)
    return { fire: true, level: fireLevel, prevLevel: prev, text }
  }

  reset() {
    this._level = -1
  }

  get level() { return this._level }
}
