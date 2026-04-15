/**
 * Content feature extraction for the LR stuck classifier.
 *
 * Ports src/pipeline/extract_features.py (the v9 relational piece),
 * benchmarks/v9_tier1_train.py (the 4 syntactic features) and
 * benchmarks/v9_content_features.py (the 4 error/token features) into
 * a single incremental extractor.
 *
 * Output order (must match benchmarks/lr_export_weights.py FEATS):
 *   0: match_ratio_5
 *   1: self_sim_max
 *   2: repeat_no_error
 *   3: cur_bash_and_match_ratio
 *   4: unique_err_sigs_6
 *   5: new_token_ratio_vs_5
 *   6: has_success_marker
 *   7: err_volume_ratio_vs_5
 *
 * Stateful: one instance per live session. Call addStep(parsedStep) once
 * per tool call in order; returns a Float64Array of length 8.
 */

import { crc32 } from 'node:zlib'

const V9_N_HISTORY = 5

// ── Regexes (Python → JS; no re.I / re.DOTALL compromises needed) ──────────

const SILENT_CMD_RE = /^(cd|pushd|popd|source|export|set|unset|alias|ulimit|umask)\b/
const FILE_EXT_RE = /\.[a-zA-Z]{1,5}$/
const SYSTEM_REMINDER_RE = /<system-reminder>[\s\S]*?<\/system-reminder>/gi

// v9_action_of helpers
const V9_PROGS_WITH_INLINE_SCRIPT = new Set([
  'node', 'python', 'python3', 'ruby', 'perl',
  'sh', 'bash', 'zsh', 'fish', 'awk', 'sed', 'tclsh',
])
const V9_INLINE_SCRIPT_FLAGS = new Set(['-e', '-c', '--command', '--eval', '-p', '-P'])
const V9_SUBCOMMAND_RE = /^[a-zA-Z][a-zA-Z0-9_\-]*$/
const V9_PATH_TOKEN_RE =
  /(?:\/?[\w@.\-]+\/)+[\w@.\-]+(?:\.[a-zA-Z0-9_]{1,8})?|[\w@.\-]+\.[a-zA-Z0-9_]{1,8}/

// Error-indicator regex (mirrors extract_features.ERROR_PATTERNS) — used by the
// coarse is_error bit; a DIFFERENT, wider regex is used for content-feature
// error-line hashing below.
const ERROR_PATTERNS =
  /error|traceback|exception|failed|failure|fatal|cannot|unable to|not found|permission denied|segmentation fault|core dumped|FAIL|ModuleNotFoundError|ImportError|SyntaxError|TypeError|ValueError|KeyError|AttributeError|RuntimeError|FileNotFoundError/i

// Wider error-line detector (matches v9_content_features._ERR_LINE_RE).
// Used to pick out lines that *contain* error-ish content so we can hash them.
const CONTENT_ERR_LINE_RE =
  /(error|traceback|exception|failed|failure|fatal|cannot|unable to|not found|permission denied|segmentation fault|core dumped|\bfail\b|ModuleNotFoundError|ImportError|SyntaxError|TypeError|ValueError|KeyError|AttributeError|RuntimeError|FileNotFoundError|undefined|warning:|\berr:|[^\s]+\.[a-z]{1,5}:\d+[:.]|at \S+:\d+|panicked at)/i

// Success marker + negation (mirrors v9_content_features SUCCESS / NEG regexes).
// Use (?:) lookarounds equivalent — JS supports lookbehind since ES2018.
const SUCCESS_RE =
  /(?<![a-z])(passed|succeeded|built|ok\b|done\b|completed|no errors?|all tests pass|\d+ passed|\d+ tests? passed|compilation succeeded|finished successfully)(?![a-z])/i
const SUCCESS_NEG_RE =
  /(not ok|0 passed|failed|not passed|did not pass|failing|errors? found)/i

// Output normalization for err-line hashing (mirrors _normalize_line)
// Must NOT collapse numbers when computing the v9 self_sim / _normalize_to_set
// path — that uses a smaller subset.
const HEX_RE = /0x[0-9a-fA-F]+/g
const TS_RE = /\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}/g
const PID_RE = /pid[=: ]\d+/gi
const TMP_RE = /\/tmp\/\S+/g
const TIME_RE = /\d+\.\d{3,}s/g
const LONG_NUM_RE = /\b\d{3,}\b/g // used ONLY by err-line normalization
const WORD_RE = /[a-zA-Z_][a-zA-Z0-9_]{2,}/g

// Max content scanned for content features (match Python MAX_OUTPUT_CHARS)
const CONTENT_MAX_CHARS = 8000
// Max lines kept by v9 _normalize_to_set
const MAX_OUTPUT_LINES = 100

// ── Output normalization helpers ──────────────────────────────────────────

/** Strip embedded <system-reminder> blocks; they're prompt-layer hints, not
 *  real tool output. */
export function stripSystemReminders(text) {
  if (!text || !text.includes('<system-reminder')) return text
  return text.replace(SYSTEM_REMINDER_RE, '')
}

/** Reproduce src/pipeline/extract_features._normalize_to_set: a set of
 *  normalized lines for jaccard comparison. Note: does NOT squash numbers. */
export function v9NormalizeToSet(output) {
  if (!output) return new Set()
  const lines = output.trim().split('\n').slice(0, MAX_OUTPUT_LINES)
  const out = new Set()
  for (let line of lines) {
    line = line.replace(HEX_RE, '0xADDR')
    line = line.replace(TS_RE, 'TIMESTAMP')
    line = line.replace(PID_RE, 'pid=PID')
    line = line.replace(TMP_RE, '/tmp/TMPFILE')
    line = line.replace(TIME_RE, 'N.NNNs')
    line = line.trim()
    if (line) out.add(line)
  }
  return out
}

function jaccard(a, b) {
  if (!b) return 0.0
  if (a.size === 0 && b.size === 0) return 1.0
  let inter = 0
  for (const v of a) if (b.has(v)) inter++
  const uni = a.size + b.size - inter
  return uni === 0 ? 1.0 : inter / uni
}

export function hasErrorIndicators(output) {
  if (!output) return false
  return ERROR_PATTERNS.test(output.slice(0, 2000))
}

// ── v9_action_of / v9_target_file_of / v9_target_scope_of ─────────────────

/** Build a compact string key for a (tool, cmd) pair that matches Python's
 *  _v9_action_of tuple semantics. Two steps get the same key iff Python
 *  would treat their actions as equal. */
export function v9ActionKey(step) {
  const tool = step.tool || 'other'
  const cmd = step.cmd || ''
  if (tool !== 'bash') {
    const toolName = step.tool_name || tool
    return `${tool}|${toolName}`
  }
  // bash: split on && / ;, drop silent-command parts
  const parts = cmd.trim().split(/\s*(?:&&|;)\s*/)
  const real = parts.filter((p) => p.trim() && !SILENT_CMD_RE.test(p.trim()))
  if (real.length === 0) {
    const toks = cmd.trim().split(/\s+/).filter(Boolean)
    return `bash|${toks[0] || ''}`
  }
  const firstPipe = real[0].trim().split(/\s*\|\s*/)[0]
  const tokens = firstPipe.trim().split(/\s+/).filter(Boolean)
  if (tokens.length === 0) return 'bash|'
  const slash = tokens[0].lastIndexOf('/')
  const prog = slash >= 0 ? tokens[0].slice(slash + 1) : tokens[0]
  // Inline-script program? Then collapse to just the program name.
  if (V9_PROGS_WITH_INLINE_SCRIPT.has(prog)) {
    for (let i = 1; i < tokens.length; i++) {
      if (V9_INLINE_SCRIPT_FLAGS.has(tokens[i])) return `bash|${prog}`
    }
  }
  if (tokens.length >= 2) {
    const t = tokens[1]
    if (!t.startsWith('-') && !t.includes('/') && !t.includes('.')
        && V9_SUBCOMMAND_RE.test(t)) {
      return `bash|${prog}|${t}`
    }
  }
  return `bash|${prog}`
}

/** Match Python's _v9_target_file_of. Returns a string or null. */
export function v9TargetFile(step) {
  if (step.file) return String(step.file)
  const tool = step.tool || 'other'
  const cmd = step.cmd || ''
  if (tool === 'bash') {
    const m = cmd.match(V9_PATH_TOKEN_RE)
    return m ? m[0] : null
  }
  return cmd || null
}

// ── Content-feature helpers (err hashes, tokens, success) ─────────────────

function normalizeErrLine(line) {
  let s = line
    .replace(HEX_RE, '0xADDR')
    .replace(TS_RE, 'TIMESTAMP')
    .replace(PID_RE, 'pid=PID')
    .replace(TMP_RE, '/tmp/TMPFILE')
    .replace(TIME_RE, 'N.NNNs')
    .replace(LONG_NUM_RE, 'NUM')
  return s.trim()
}

function crcUint(str) {
  return crc32(Buffer.from(str, 'utf8')) >>> 0
}

/** Set<number> of CRC32 hashes of normalized error-containing lines. */
function errLineHashes(output) {
  if (!output) return new Set()
  const out = output.slice(0, CONTENT_MAX_CHARS)
  const hashes = new Set()
  for (const line of out.split('\n')) {
    if (CONTENT_ERR_LINE_RE.test(line)) {
      const norm = normalizeErrLine(line)
      if (norm.length >= 4) hashes.add(crcUint(norm))
    }
  }
  return hashes
}

/** Set<string> of unique word-tokens in the first 8k chars. */
function tokenSet(output) {
  if (!output) return new Set()
  const out = output.slice(0, CONTENT_MAX_CHARS)
  const set = new Set()
  let m
  WORD_RE.lastIndex = 0
  while ((m = WORD_RE.exec(out)) !== null) set.add(m[0])
  return set
}

function hasSuccessMarker(output) {
  if (!output) return false
  const out = output.slice(0, CONTENT_MAX_CHARS)
  if (SUCCESS_NEG_RE.test(out)) return false
  return SUCCESS_RE.test(out)
}

function errLineCount(output) {
  if (!output) return 0
  const out = output.slice(0, CONTENT_MAX_CHARS)
  let n = 0
  for (const line of out.split('\n')) if (CONTENT_ERR_LINE_RE.test(line)) n++
  return n
}

// ── Incremental extractor ─────────────────────────────────────────────────

export const LR_FEATURE_NAMES = Object.freeze([
  'match_ratio_5',
  'self_sim_max',
  'repeat_no_error',
  'cur_bash_and_match_ratio',
  'unique_err_sigs_6',
  'new_token_ratio_vs_5',
  'has_success_marker',
  'err_volume_ratio_vs_5',
])

export class ContentFeatureExtractor {
  constructor() {
    // v9 bucket: key "actionKey|targetFile" → array of last ≤5 output-sets
    this._bucket = new Map()
    // Last ≤5 infos (most recent last) for slot-based act_match/file_match
    // and for stored self_relative_sim per slot.
    this._prevInfos = []
    // Content-feature rolling histories (most recent last).
    this._errHashHistory = []
    this._tokenHistory = []
    this._errCountHistory = []
  }

  /**
   * @param {{tool: string, cmd: string, file: string|null, output: string, tool_name?: string}} step
   * @returns {Float64Array} 8-dim feature vector in LR_FEATURE_NAMES order
   */
  addStep(step) {
    const clean = stripSystemReminders(step.output || '')
    const actionKey = v9ActionKey(step)
    const targetFile = v9TargetFile(step)
    const bucketKey = `${actionKey}|${targetFile ?? ''}`
    const outputSet = v9NormalizeToSet(clean)
    const isError = hasErrorIndicators(clean) ? 1.0 : 0.0
    const isBash = (step.tool || '') === 'bash' ? 1.0 : 0.0

    // self_relative_sim: max jaccard vs priors in the same (action, file) bucket.
    // Must be computed BEFORE we append this step to the bucket.
    let selfRelativeSim = 0.0
    const priors = this._bucket.get(bucketKey)
    if (priors) {
      for (const p of priors) {
        const j = jaccard(outputSet, p)
        if (j > selfRelativeSim) {
          selfRelativeSim = j
          if (selfRelativeSim >= 1.0) break
        }
      }
    }

    // Compute v9 slot arrays (act_match and file_match vs each of last 5 infos).
    // Slot 0 = most recent prior, slot 4 = 5 steps back. Empty slots = 0.
    const actMatch = new Array(V9_N_HISTORY).fill(0.0)
    const fileMatch = new Array(V9_N_HISTORY).fill(0.0)
    const priorSelfSim = new Array(V9_N_HISTORY).fill(0.0)
    const n = this._prevInfos.length
    for (let slot = 0; slot < V9_N_HISTORY; slot++) {
      const idx = n - 1 - slot
      if (idx < 0) break
      const prior = this._prevInfos[idx]
      actMatch[slot] = prior.actionKey === actionKey ? 1.0 : 0.0
      fileMatch[slot] =
        prior.targetFile !== null && prior.targetFile === targetFile ? 1.0 : 0.0
      priorSelfSim[slot] = prior.selfRelativeSim
    }

    // ── Tier-1 syntactic features (mirror v9_tier1_train.compute_tier1_features)
    // match_ratio_5: just the mean of act_match boolean (not AND'd with file_match)
    let matchSum = 0
    for (let s = 0; s < V9_N_HISTORY; s++) if (actMatch[s] >= 0.5) matchSum++
    const matchRatio5 = matchSum / V9_N_HISTORY

    // self_sim_max: max of the 5 priors' stored self_relative_sim values
    let selfSimMax = 0.0
    for (const v of priorSelfSim) if (v > selfSimMax) selfSimMax = v

    // repeat_no_error: p1 actMatch AND current step not an error
    const p1Match = actMatch[0] >= 0.5
    const noErr = isError < 0.5
    const repeatNoError = p1Match && noErr ? 1.0 : 0.0

    // cur_bash_and_match_ratio: is_bash * match_ratio_5
    const curBashMatchRatio = isBash * matchRatio5

    // ── Tier-3 content features ──────────────────────────────────────────
    const curErrHashes = errLineHashes(clean)
    const curTokens = tokenSet(clean)
    const curErrCount = errLineCount(clean)

    // unique_err_sigs_6: |current ∪ priors|, capped at 6, divided by 6.
    const union = new Set(curErrHashes)
    for (const prior of this._errHashHistory) for (const h of prior) union.add(h)
    const uniqueErrSigs6 = Math.min(union.size, 6) / 6.0

    // new_token_ratio_vs_5: fraction of current tokens not in any prior token set
    let newTokenRatio = 0.0
    if (curTokens.size === 0) newTokenRatio = 0.0
    else if (this._tokenHistory.length === 0) newTokenRatio = 1.0
    else {
      const priorUnion = new Set()
      for (const t of this._tokenHistory) for (const w of t) priorUnion.add(w)
      let newCount = 0
      for (const w of curTokens) if (!priorUnion.has(w)) newCount++
      newTokenRatio = newCount / curTokens.size
    }

    // has_success_marker
    const hasSuccess = hasSuccessMarker(clean) ? 1.0 : 0.0

    // err_volume_ratio_vs_5: log1p(cur) - log1p(mean(prior_err_counts)), clamped [-3, 3]
    let errVolumeRatio = 0.0
    if (this._errCountHistory.length > 0) {
      let sum = 0
      for (const c of this._errCountHistory) sum += c
      const priorMean = sum / this._errCountHistory.length
      let ratio = Math.log1p(curErrCount) - Math.log1p(priorMean)
      if (ratio < -3.0) ratio = -3.0
      else if (ratio > 3.0) ratio = 3.0
      errVolumeRatio = ratio
    }

    // ── Update rolling state AFTER feature computation ──────────────────
    // v9 bucket: store this step's output set
    const bucketList = this._bucket.get(bucketKey) ?? []
    bucketList.push(outputSet)
    if (bucketList.length > 5) bucketList.shift()
    this._bucket.set(bucketKey, bucketList)

    // Info ring (last 5 for slot comparisons)
    this._prevInfos.push({ actionKey, targetFile, selfRelativeSim })
    if (this._prevInfos.length > V9_N_HISTORY) this._prevInfos.shift()

    // Content histories
    this._errHashHistory.push(curErrHashes)
    if (this._errHashHistory.length > V9_N_HISTORY) this._errHashHistory.shift()
    this._tokenHistory.push(curTokens)
    if (this._tokenHistory.length > V9_N_HISTORY) this._tokenHistory.shift()
    this._errCountHistory.push(curErrCount)
    if (this._errCountHistory.length > V9_N_HISTORY) this._errCountHistory.shift()

    // Pack into the final vector in LR_FEATURE_NAMES order
    const out = new Float64Array(8)
    out[0] = matchRatio5
    out[1] = selfSimMax
    out[2] = repeatNoError
    out[3] = curBashMatchRatio
    out[4] = uniqueErrSigs6
    out[5] = newTokenRatio
    out[6] = hasSuccess
    out[7] = errVolumeRatio
    return out
  }
}
