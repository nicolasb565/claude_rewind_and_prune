#!/usr/bin/env node
/**
 * Cross-validate proxy/content_features.mjs against the Python reference
 * dumped by benchmarks/content_feature_parity.py.
 *
 * Strategy: read the reference JSON, run each session through the JS
 * ContentFeatureExtractor step-by-step, compare feature-by-feature with
 * the Python values, and report per-feature mismatch counts.
 *
 * Exit 0 if all within tolerance, 1 otherwise. Intended to be run as a
 * test/CI step.
 */
import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { ContentFeatureExtractor, LR_FEATURE_NAMES } from '../proxy/content_features.mjs'

const __dirname = dirname(fileURLToPath(import.meta.url))
const REPO = resolve(__dirname, '..')
const REF_PATH = resolve(REPO, 'data/generated/content_features_ood_python.json')
const TOL = 1e-6 // floats should match bit-for-bit modulo FP noise

const ref = JSON.parse(readFileSync(REF_PATH, 'utf8'))

const perFeature = {}
for (const name of LR_FEATURE_NAMES) perFeature[name] = { matched: 0, mismatched: 0, maxDelta: 0, samples: [] }

let totalSteps = 0
let totalMismatches = 0
const sessionSummaries = []

for (const sess of ref.sessions) {
  const ext = new ContentFeatureExtractor()
  let sessMismatch = 0
  for (const s of sess.steps) {
    const v = ext.addStep(s.parsed)
    totalSteps++
    for (let i = 0; i < LR_FEATURE_NAMES.length; i++) {
      const name = LR_FEATURE_NAMES[i]
      const jsVal = v[i]
      const pyVal = s.features[name]
      const delta = Math.abs(jsVal - pyVal)
      const rec = perFeature[name]
      if (delta <= TOL) {
        rec.matched++
      } else {
        rec.mismatched++
        totalMismatches++
        sessMismatch++
        if (delta > rec.maxDelta) rec.maxDelta = delta
        if (rec.samples.length < 5) {
          rec.samples.push({
            session: sess.session_id,
            step: s.step,
            py: pyVal,
            js: jsVal,
            delta,
          })
        }
      }
    }
  }
  sessionSummaries.push({ session_id: sess.session_id, n_steps: sess.steps.length, mismatches: sessMismatch })
}

// ── Report ────────────────────────────────────────────────────────────────
console.log(`=== Content feature parity check ===`)
console.log(`ref file: ${REF_PATH}`)
console.log(`tolerance: ${TOL}`)
console.log(`total steps: ${totalSteps}, total mismatches: ${totalMismatches}`)
console.log()
console.log(`${'feature'.padEnd(28)} ${'matched'.padStart(8)} ${'mismatched'.padStart(12)} ${'maxΔ'.padStart(12)}`)
console.log('-'.repeat(64))
for (const name of LR_FEATURE_NAMES) {
  const r = perFeature[name]
  console.log(`${name.padEnd(28)} ${String(r.matched).padStart(8)} ${String(r.mismatched).padStart(12)} ${r.maxDelta.toExponential(2).padStart(12)}`)
}

let fail = false
for (const name of LR_FEATURE_NAMES) {
  const r = perFeature[name]
  if (r.mismatched > 0) {
    fail = true
    console.log(`\n${name} — first ${r.samples.length} mismatches:`)
    for (const s of r.samples) {
      console.log(`  ${s.session}[${s.step}]  py=${s.py}  js=${s.js}  Δ=${s.delta}`)
    }
  }
}

if (fail) {
  console.error(`\nFAIL — features out of tolerance`)
  process.exit(1)
} else {
  console.log(`\nPASS — all features match within ${TOL}`)
  process.exit(0)
}
