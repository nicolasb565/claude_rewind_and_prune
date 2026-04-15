/**
 * Three-tier filter state machine for per-step LR stuck scores.
 *
 * Design motivation: the LR baseline's step-level signal is too noisy for
 * hard intervention at any single threshold — see README Key Finding #8.
 * Each escalation level uses a different filter calibrated to its
 * confidence requirement:
 *
 *   soft   →  mean-of-2 @ 0.34    (session recall 7/7 on OOD benchmark, noisy)
 *   medium →  median-of-4 @ 0.645 (balanced, 5/7 sessions, 1 FP session)
 *   hard   →  median-of-9 @ 0.605 (step P=0.78, 0 FP sessions, 2/7 sessions)
 *
 * State machine (step-driven, no wall-clock):
 *   level -1 = idle
 *   level  0 = soft fired
 *   level  1 = medium fired
 *   level  2 = hard fired
 *
 * Per step:
 *   1. Push score onto buffers. If the soft filter is NOT firing, reset to
 *      level -1. (The soft envelope defines stuck; its collapse is recovery.)
 *   2. Otherwise advance at most one level per step:
 *        level -1 → 0 if soft fires
 *        level  0 → 1 if medium fires
 *        level  1 → 2 if hard fires
 *      Each advancement fires the corresponding nudge once.
 *
 * Properties:
 *   - A single productive step (soft filter off) clears all state.
 *   - Hard cannot fire until step 9 (median-of-9 needs 9 samples).
 *   - Every level transition requires its filter to independently agree.
 *   - At most one nudge per step.
 */

const DEFAULT_CONFIG = {
  soft:   { kind: 'mean',   n: 2, threshold: 0.34  },
  medium: { kind: 'median', n: 4, threshold: 0.645 },
  hard:   { kind: 'median', n: 9, threshold: 0.605 },
}

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
   * @param {{soft: FilterSpec, medium: FilterSpec, hard: FilterSpec}} config
   *   where FilterSpec = { kind: 'mean'|'median', n: number, threshold: number }
   */
  constructor(config = DEFAULT_CONFIG) {
    this._cfg = config
    this._soft = new RollingBuffer(config.soft.n)
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
   * @param {number} score  LR P(stuck) in [0, 1]
   * @returns {{soft: boolean, medium: boolean, hard: boolean, aggregates: {soft: number|null, medium: number|null, hard: number|null}}}
   */
  update(score) {
    this._soft.push(score)
    this._med.push(score)
    this._hard.push(score)
    return {
      soft: this._filterFires(this._cfg.soft, this._soft),
      medium: this._filterFires(this._cfg.medium, this._med),
      hard: this._filterFires(this._cfg.hard, this._hard),
      aggregates: {
        soft: this._soft.ready()
          ? this._aggregate(this._cfg.soft.kind, this._soft) : null,
        medium: this._med.ready()
          ? this._aggregate(this._cfg.medium.kind, this._med) : null,
        hard: this._hard.ready()
          ? this._aggregate(this._cfg.hard.kind, this._hard) : null,
      },
    }
  }

  reset() {
    this._soft.clear()
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
   * Returns `{ fire: boolean, level: number, prevLevel: number }`.
   *   - fire=true means a nudge for this level should be sent THIS step
   *   - level is the (possibly updated) current level after the step
   *   - prevLevel is the level before the step — useful for logging resets
   *
   * @param {{soft: boolean, medium: boolean, hard: boolean}} filters
   */
  update(filters) {
    const prev = this._level

    // Recovery: soft filter collapse clears all state.
    if (!filters.soft) {
      this._level = -1
      return { fire: false, level: this._level, prevLevel: prev }
    }

    // Try to advance exactly one level. Each advance requires the next
    // tier's filter to have independently agreed.
    if (this._level === -1) {
      this._level = 0
      return { fire: true, level: 0, prevLevel: prev }
    }
    if (this._level === 0 && filters.medium) {
      this._level = 1
      return { fire: true, level: 1, prevLevel: prev }
    }
    if (this._level === 1 && filters.hard) {
      this._level = 2
      return { fire: true, level: 2, prevLevel: prev }
    }
    return { fire: false, level: this._level, prevLevel: prev }
  }

  reset() {
    this._level = -1
  }

  get level() { return this._level }
}
