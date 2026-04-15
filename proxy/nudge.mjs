/**
 * Nudge escalation controller.
 *
 * Tracks stuck detection state for one session and decides when and how to
 * inject a corrective nudge into the Claude Code conversation.
 *
 * Levels:
 *   -1  silent  — absorbs the first stuck detection without firing; prevents
 *                 single-turn false positive spikes from immediately nudging
 *    0  soft    — asks the agent to reflect on its recent actions
 *    1  medium  — demands an explicit 3-point strategy change
 *    2  hard    — STOP directive with mandatory root-cause analysis
 *
 * Each detection increments the level (capped at 2). A score drop below
 * resetThreshold resets the level back to -1 (agent has responded and moved on).
 * Cooldowns per level prevent re-firing too quickly.
 */

// Named constants — not inline strings — so tests can verify content without
// depending on exact wording.
const SOFT_NUDGE = (turn, pct, recentList) =>
  `[CONTEXT MONITOR — turn ${turn}, confidence ${pct}%]\n\n` +
  `Your recent actions show signs of repetitive patterns. ` +
  `You may be going in circles.\n\n` +
  `Recent tool calls:\n  ${recentList}\n\n` +
  `Review your last few turns critically:\n` +
  `- Are you retrying the same approach with minor variations?\n` +
  `- Are you investigating the same files/functions repeatedly?\n` +
  `- Has your hypothesis changed or are you stuck on the same one?\n\n` +
  `If you are going in circles, try a fundamentally different strategy.\n` +
  `State what you have learned so far and what new approach you will try.`

const MEDIUM_NUDGE = (turn, pct, recentList) =>
  `[CONTEXT MONITOR — turn ${turn}, confidence ${pct}% — repeated signal]\n\n` +
  `You have been nudged before and the repetitive pattern continues.\n\n` +
  `Recent tool calls:\n  ${recentList}\n\n` +
  `You appear to be stuck in a loop. The approach you are using is not working.\n` +
  `Before your next tool call:\n` +
  `1. State in one sentence what you have been trying to do.\n` +
  `2. State specifically why it has not worked.\n` +
  `3. Propose a different approach you have not tried yet.\n\n` +
  `Do not retry the same command. Switch strategy.`

const HARD_NUDGE = (turn, pct, recentList) =>
  `[CONTEXT MONITOR — turn ${turn}, confidence ${pct}% — escalated]\n\n` +
  `STOP. You are deeply stuck and have not responded to prior nudges.\n\n` +
  `Recent tool calls:\n  ${recentList}\n\n` +
  `Do not run any more tool calls until you have answered these:\n` +
  `1. What is the root cause of the problem you are trying to solve?\n` +
  `2. What have you tried, and why did each attempt fail?\n` +
  `3. What fundamentally different approach will you take next?\n\n` +
  `If you cannot answer these, state that clearly and ask for guidance.`

export const NUDGE_TEMPLATES = [SOFT_NUDGE, MEDIUM_NUDGE, HARD_NUDGE]

// Default cooldowns indexed by (nudgeLevel + 1): level -1→[0], 0→[1], 1→[2], 2→[3]
const DEFAULT_COOLDOWNS = [1, 4, 8, 8]

export class NudgeController {
  /**
   * @param {{ threshold: number, resetFactor?: number, cooldowns?: number[] }} opts
   *   threshold    — MLP score above which stuck is detected
   *   resetFactor  — score must drop below threshold * resetFactor to reset level
   *   cooldowns    — turns before re-firing, indexed by (nudgeLevel + 1)
   */
  constructor({ threshold, resetFactor = 0.94, cooldowns = DEFAULT_COOLDOWNS }) {
    this._threshold = threshold
    this._resetThreshold = threshold * resetFactor
    this._cooldowns = cooldowns
    this._nudgeLevel = -1
    this._lastNudgeTurn = Number.NEGATIVE_INFINITY
  }

  /**
   * Update state for the current API turn and decide whether to fire a nudge.
   *
   * @param {number}   score        MLP stuck score in [0, 1]
   * @param {number}   turn         monotonically increasing turn counter
   * @param {string[]} recentTools  recent tool call summaries for nudge text
   * @returns {{ fire: boolean, level: number, text: string }}
   */
  update(score, turn, recentTools = []) {
    // Reset check runs unconditionally — recovery should happen even during cooldown
    if (score < this._resetThreshold) this._nudgeLevel = -1

    if (score < this._threshold) {
      return { fire: false, level: this._nudgeLevel, text: '' }
    }

    const cooldown = this._cooldowns[this._nudgeLevel + 1] ?? 4
    if (turn - this._lastNudgeTurn < cooldown) {
      return { fire: false, level: this._nudgeLevel, text: '' }
    }

    // Stuck detected — record this turn and handle level logic
    this._lastNudgeTurn = turn

    if (this._nudgeLevel < 0) {
      // Silent detection: absorb without firing, bump level to 0
      this._nudgeLevel++
      return { fire: false, level: this._nudgeLevel, text: '' }
    }

    const level = this._nudgeLevel
    this._nudgeLevel = Math.min(this._nudgeLevel + 1, 2)

    const pct = Math.round(score * 100)
    const recentList = recentTools.join('\n  ') // caller already limits to 8 entries
    const text = NUDGE_TEMPLATES[level](turn, pct, recentList)

    return { fire: true, level, text }
  }
}
