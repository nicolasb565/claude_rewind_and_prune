/**
 * Logistic-regression stuck classifier.
 *
 * Shippable alternative to the v5 MLP: 9 parameters (8 weights + 1 bias)
 * over 8 content features. See benchmarks/lr_filter_sweep.py for the
 * filter-sweep study that motivated the three-tier filter design it
 * composes with, and Key Findings #7/#8 in README.md for the OOD
 * ceiling context.
 */

import { readFileSync } from 'node:fs'

export class LRClassifier {
  /**
   * @param {object} payload  parsed lr_weights.json
   */
  constructor(payload) {
    this.features = payload.features
    this._w = new Float64Array(payload.weights)
    this._b = payload.intercept
    this._mean = new Float64Array(payload.feature_mean)
    this._std = new Float64Array(payload.feature_std)
    if (this._w.length !== this._mean.length
        || this._w.length !== this._std.length) {
      throw new Error('lr weights/mean/std length mismatch')
    }
  }

  /**
   * Score one step's feature row.
   *
   * @param {number[]|Float64Array} featureRow  in the same order as
   *     payload.features (see proxy/lr_weights.json)
   * @returns {number} P(stuck) in [0, 1]
   */
  score(featureRow) {
    if (featureRow.length !== this._w.length) {
      throw new Error(
        `feature row length ${featureRow.length} != expected ${this._w.length}`
      )
    }
    let logit = this._b
    for (let i = 0; i < this._w.length; i++) {
      const z = (featureRow[i] - this._mean[i]) / this._std[i]
      logit += z * this._w[i]
    }
    return 1.0 / (1.0 + Math.exp(-logit))
  }
}

export function loadLR(path) {
  const payload = JSON.parse(readFileSync(path, 'utf8'))
  return new LRClassifier(payload)
}
