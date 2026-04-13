/**
 * Per-session stuck detector.
 *
 * Composes feature extraction, ring buffer history, and MLP inference into a
 * stateful object — one instance per active Claude Code session.
 *
 * Call addStep() once per tool call in order. The ring buffer score feedback
 * (each step's score is stored and fed back as input to subsequent steps) mirrors
 * inference-time behaviour described in src/training/train.py.
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
    const features = computeFeatures(step, this._outputHistory, this._stepCount)
    const inputVec = this._ring.buildInput(features)
    const score = this._mlp.forward(inputVec)
    this._ring.push(features, score)
    this._stepCount++
    return score
  }
}
