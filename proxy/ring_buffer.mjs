/**
 * Fixed-size ring buffer for N steps of feature vectors.
 *
 * Mirrors the np.roll-based implementation in src/training/train.py:
 *   feat_buf = np.roll(feat_buf, 1, axis=0); feat_buf[0] = curr
 *
 * Layout: buf[0] = T-1 (most recent), buf[N-1] = T-N (oldest).
 * All entries are zero-initialized (zero-padding for the first N steps).
 *
 * The previous-scores feedback channel was removed — it created a train/inference
 * mismatch (training fed perfect labels, inference fed continuous sigmoid output)
 * and ablation showed F1=0.961 with or without it, so the simpler 48-dim model wins.
 */

export const N_HISTORY = 5
export const NUM_FEATURES = 7

export class RingBuffer {
  /**
   * @param {number} n           history depth (default: N_HISTORY = 5)
   * @param {number} featureDim  features per step (default: NUM_FEATURES = 8)
   */
  constructor(n = N_HISTORY, featureDim = NUM_FEATURES) {
    this.n = n
    this.featureDim = featureDim
    this._feats = new Float32Array(n * featureDim) // zero-initialized
  }

  /**
   * Roll the buffer and insert the new step at position 0 (T-1 slot).
   *
   * @param {Float32Array} features  length-featureDim vector for the current step
   */
  push(features) {
    // Shift features right by one slot: [0..n-2] → [1..n-1]
    for (let i = this.n - 1; i > 0; i--) {
      const src = (i - 1) * this.featureDim
      const dst = i * this.featureDim
      this._feats.copyWithin(dst, src, src + this.featureDim)
    }
    this._feats.set(features.subarray(0, this.featureDim), 0)
  }

  /**
   * Build the full MLP input vector for the current step.
   *
   * Layout: [curr(featureDim), hist_T-1(featureDim), ..., hist_T-N(featureDim)]
   * Total length: featureDim * (1 + N)  →  8 * 6 = 48 for default params.
   *
   * Called BEFORE push() so the ring contains T-1..T-N at this point.
   *
   * @param {Float32Array} currFeats  current step's feature vector
   * @returns {Float32Array}
   */
  buildInput(currFeats) {
    const dim = this.featureDim
    const result = new Float32Array(dim * (1 + this.n))
    result.set(currFeats.subarray(0, dim), 0)
    result.set(this._feats, dim)
    return result
  }
}
