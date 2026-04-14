/**
 * v5 per-step MLP inference (no score history, no step_index_norm).
 *
 * Architecture: Linear(42,64) → ReLU → Linear(64,32) → ReLU → Linear(32,1) → Sigmoid
 * Normalization: mean/std applied to all 42 dims (every input is a feature).
 *
 * History: removed score feedback (5 dims) and step_index_norm (1 dim × 6 slots).
 * Both were either weak signal or had train/inference mismatches; multi-seed
 * ablation showed no statistically significant cost to dropping them.
 */

import { readFileSync } from 'node:fs'

const INPUT_DIM = 42

/**
 * Load an MLP instance from a JSON weights file.
 *
 * @param {string} weightsPath  path to stuck_weights.json
 * @returns {MLP}
 */
export function loadMLP(weightsPath) {
  const weights = JSON.parse(readFileSync(weightsPath, 'utf8'))
  return new MLP(weights)
}

export class MLP {
  /** @param {object} weights  parsed stuck_weights.json */
  constructor(weights) {
    this._fc1w = weights['fc1.weight'] // 64 × 53
    this._fc1b = weights['fc1.bias'] // 64
    this._fc2w = weights['fc2.weight'] // 32 × 64
    this._fc2b = weights['fc2.bias'] // 32
    this._fc3w = weights['fc3.weight'] // 1 × 32
    this._fc3b = weights['fc3.bias'] // 1
    this._mean = new Float32Array(weights['norm_mean']) // 53
    this._std = new Float32Array(weights['norm_std']) // 53
  }

  /**
   * Run the forward pass on a raw (un-normalized) input vector.
   *
   * @param {Float32Array} input  length-53 input from RingBuffer.buildInput()
   * @returns {number}  sigmoid score in [0, 1]
   */
  forward(input) {
    const x = new Float32Array(INPUT_DIM)
    for (let i = 0; i < INPUT_DIM; i++) {
      x[i] = (input[i] - this._mean[i]) / (this._std[i] || 1e-6)
    }

    const h1 = _matVec(this._fc1w, x, this._fc1b)
    _relu(h1)
    const h2 = _matVec(this._fc2w, h1, this._fc2b)
    _relu(h2)
    const h3 = _matVec(this._fc3w, h2, this._fc3b)
    return _sigmoid(h3[0])
  }
}

/** Dense matrix-vector multiply: out[i] = bias[i] + sum_j(weight[i][j] * input[j]) */
function _matVec(weight, input, bias) {
  const out = new Float32Array(weight.length)
  for (let i = 0; i < weight.length; i++) {
    let s = bias[i]
    const row = weight[i]
    for (let j = 0; j < input.length; j++) s += row[j] * input[j]
    out[i] = s
  }
  return out
}

function _relu(arr) {
  for (let i = 0; i < arr.length; i++) if (arr[i] < 0) arr[i] = 0
}

function _sigmoid(x) {
  return 1 / (1 + Math.exp(-x))
}
