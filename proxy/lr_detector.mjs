/**
 * Per-session LR stuck detector + two-tier filter.
 *
 * Drop-in replacement for SessionDetector that wraps the ContentFeature
 * extractor, LR classifier, and TieredFilter into one stateful object.
 * One instance per live Claude Code session.
 *
 * Interface:
 *   addStep(toolName, input, output) → { score, filters, aggregates }
 *
 *   score      — LR P(stuck) in [0,1]
 *   filters    — { medium, hard } booleans from TieredFilter
 *   aggregates — { medium, hard } aggregate values or null if not ready
 */

import { parseToolCall } from './features.mjs'
import { ContentFeatureExtractor } from './content_features.mjs'
import { TieredFilter } from './tiered_filter.mjs'

export class LRSessionDetector {
  /**
   * @param {{ score: (featureRow: Float64Array) => number }} lr
   * @param {object} [filterConfig]  optional TieredFilter config override
   */
  constructor(lr, filterConfig) {
    this._lr = lr
    this._features = new ContentFeatureExtractor()
    this._filter = new TieredFilter(filterConfig)
    this._stepCount = 0
  }

  addStep(toolName, input, output) {
    const step = parseToolCall(toolName, input, output)
    const featureRow = this._features.addStep(step)
    const score = this._lr.score(featureRow)
    const filterState = this._filter.update(score)
    this._stepCount++
    return {
      score,
      filters: {
        medium: filterState.medium,
        hard: filterState.hard,
      },
      aggregates: filterState.aggregates,
    }
  }

  get stepCount() { return this._stepCount }
}
