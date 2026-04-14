/**
 * Per-session stuck detector.
 *
 * Composes feature extraction, ring buffer history, and MLP inference into a
 * stateful object — one instance per active Claude Code session.
 *
 * Call addStep() once per tool call in order. The ring buffer holds the last
 * N=5 per-step feature vectors; previous scores are NOT fed back (ablation
 * showed identical F1 with or without them, and dropping them eliminates the
 * train/inference distribution mismatch).
 */

import { parseToolCall, computeFeatures } from './features.mjs'
import { RingBuffer } from './ring_buffer.mjs'

export class SessionDetector {
  /**
   * @param {{ forward: (input: Float32Array) => number }} mlp  loaded MLP instance
   */
  constructor(mlp) {
    this._mlp = mlp
    this._ring = new RingBuffer()
    this._outputHistory = new Map() // cmdHashInt → outputSet
    this._stepCount = 0
  }

  /**
   * Process one tool call and return the stuck score.
   *
   * @param {string} toolName  Claude Code tool name (e.g. "Bash", "Edit")
   * @param {object} input     Tool input object from the API message
   * @param {string} output    Tool output text
   * @returns {number}  MLP sigmoid score in [0, 1]
   */
  addStep(toolName, input, output) {
    const step = parseToolCall(toolName, input, output)
    const features = computeFeatures(step, this._outputHistory)
    const inputVec = this._ring.buildInput(features)
    const score = this._mlp.forward(inputVec)
    this._ring.push(features)
    this._stepCount++
    return score
  }
}
