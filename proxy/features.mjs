/**
 * Per-step feature extraction for the v5 MLP stuck detector.
 *
 * Stateless pure functions — all session state (outputHistory, stepCount) is
 * owned by the caller (SessionDetector). This makes each function independently
 * testable without needing a full session object.
 *
 * Feature order matches src/training/train.py STEP_FEATURES (with step_index_norm
 * dropped — it was a known train/inference mismatch and ablation showed no
 * statistically significant cost to removing it):
 *   [tool_idx, cmd_hash, file_hash, output_similarity, has_prior_output,
 *    output_length, is_error]
 */

import { crc32 } from 'node:zlib'

// CRC32 → [0, 1): use 2**32, NOT 1<<32
// JS bitwise operators truncate to 32-bit signed integers: 1<<32 === 1, not 4294967296.
const CRC32_NORM = 1 / 2 ** 32

// Tool name mapping: Claude Code names → abstract categories (matches parsers/nlile.py)
export const TOOL_MAP = {
  Bash: 'bash',
  bash: 'bash',
  Edit: 'edit',
  edit: 'edit',
  Write: 'edit',
  write: 'edit',
  MultiEdit: 'edit',
  Read: 'view',
  read: 'view',
  Grep: 'search',
  grep: 'search',
  Glob: 'search',
  glob: 'search',
  Agent: 'other',
  Task: 'other',
  TodoRead: 'other',
  TodoWrite: 'other',
}

export const TOOL_NAMES = ['bash', 'edit', 'view', 'search', 'create', 'submit', 'other']
export const TOOL_TO_IDX = Object.fromEntries(TOOL_NAMES.map((t, i) => [t, i]))

// Tools whose outputs are meaningless (edit success strings) — skip output processing
const EDIT_TOOLS = new Set(['edit', 'create', 'submit'])

const MAX_OUTPUT_LINES = 100
const SILENT_CMD_RE = /^(cd|pushd|popd|source|export|set|unset|alias|ulimit|umask)\b/
const FILE_EXT_RE = /\.[a-zA-Z]{1,5}$/
const SYSTEM_REMINDER_RE = /<system-reminder>[\s\S]*?<\/system-reminder>/gi
const ERROR_RE =
  /error|traceback|exception|failed|failure|fatal|cannot|unable to|not found|permission denied|segmentation fault|core dumped|FAIL|ModuleNotFoundError|ImportError|SyntaxError|TypeError|ValueError|KeyError|AttributeError|RuntimeError|FileNotFoundError/i

/**
 * Parse a raw Claude Code tool call into a normalized step dict.
 *
 * @param {string} toolName  Claude Code tool name (e.g. "Bash", "Edit")
 * @param {object} input     Tool input object from the API message
 * @param {string} output    Tool output text
 * @returns {{ tool: string, cmd: string, file: string|null, output: string }}
 */
export function parseToolCall(toolName, input, output) {
  const tool = TOOL_MAP[toolName] ?? 'other'
  // command/file_path/pattern are used at full length. description/prompt are
  // fallbacks for Task/Agent tools, truncated to 200 chars. Uses || (falsy-check)
  // to mirror Python's "if not cmd: cmd = description[:200]" (nlile.py:56-59).
  const primaryCmd = input?.command || input?.file_path || input?.pattern || ''
  const cmd = primaryCmd
    ? String(primaryCmd)
    : String(input?.description ?? input?.prompt ?? '').slice(0, 200)
  const rawFile = input?.file_path ?? input?.path ?? null
  return {
    tool,
    cmd,
    file: rawFile !== null && rawFile !== undefined ? String(rawFile) : null,
    output: output ?? '',
  }
}

/**
 * Extract 'base_command:target_file' for semantic command matching.
 * Must produce the same key as Python's _cmd_semantic_key() in extract_features.py.
 *
 * @param {string} cmd  raw bash command string
 * @returns {string}    semantic key
 */
export function cmdSemanticKey(cmd) {
  if (!cmd) return ''
  const parts = cmd.trim().split(/\s*(?:&&|;)\s*/)
  const real = parts.filter((p) => p.trim() && !SILENT_CMD_RE.test(p.trim()))
  if (real.length === 0) {
    const t = cmd.trim().split(/\s+/)
    return t.length > 0 ? t[0] : ''
  }
  const first = real[0].trim().split(/\s*\|\s*/)[0]
  const tokens = first.trim().split(/\s+/)
  if (tokens.length === 0) return ''
  const si = tokens[0].lastIndexOf('/')
  const base = si >= 0 ? tokens[0].slice(si + 1) : tokens[0]
  let target = null
  for (let i = 1; i < tokens.length; i++) {
    if (tokens[i].startsWith('-')) continue
    if (FILE_EXT_RE.test(tokens[i]) || tokens[i].includes('/')) {
      const ti = tokens[i].lastIndexOf('/')
      target = ti >= 0 ? tokens[i].slice(ti + 1) : tokens[i]
      break
    }
  }
  return target ? `${base}:${target}` : base
}

/**
 * Normalize an output string to a set of canonical lines for Jaccard comparison.
 * Strips addresses, timestamps, PIDs, and temp paths so minor variations don't
 * prevent output-similarity detection.
 *
 * @param {string} output
 * @returns {Set<string>}
 */
export function normalizeToSet(output) {
  if (!output) return new Set()
  const lines = output.trim().split('\n').slice(0, MAX_OUTPUT_LINES)
  const result = new Set()
  for (let line of lines) {
    line = line.replace(/0x[0-9a-fA-F]+/g, '0xADDR')
    line = line.replace(/\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}/g, 'TIMESTAMP')
    line = line.replace(/pid[=: ]\d+/gi, 'pid=PID')
    line = line.replace(/\/tmp\/[^\s]+/g, '/tmp/TMPFILE')
    line = line.replace(/\d+\.\d{3,}s/g, 'N.NNNs')
    line = line.trim()
    if (line) result.add(line)
  }
  return result
}

/**
 * Jaccard similarity between two output sets.
 *
 * @param {Set} setA
 * @param {Set|null} setB  null means no prior output (returns 0)
 * @returns {number}  value in [0, 1]
 */
export function jaccard(setA, setB) {
  if (!setB) return 0.0
  if (setA.size === 0 && setB.size === 0) return 1.0
  let intersection = 0
  for (const v of setA) if (setB.has(v)) intersection++
  const union = setA.size + setB.size - intersection
  return union === 0 ? 1.0 : intersection / union
}

/**
 * Compute the 8 v5 per-step features for one tool call.
 *
 * Side effect: mutates outputHistory by storing the current step's outputSet
 * keyed by cmdHashInt so subsequent steps can compute output_similarity.
 *
 * @param {{ tool: string, cmd: string, file: string|null, output: string }} step
 * @param {Map<number, Set>} outputHistory  keyed by cmdHashInt, mutated in-place
 * @returns {Float32Array}  length-7 feature vector
 */
export function computeFeatures(step, outputHistory) {
  const { tool, cmd, file, output } = step
  const toolIdx = TOOL_TO_IDX[tool] ?? TOOL_TO_IDX['other']

  // CRC32 hashes (unsigned 32-bit via >>> 0)
  const fileHashInt = file ? crc32(Buffer.from(file, 'utf8')) >>> 0 : null
  const cmdKey = tool === 'bash' && cmd ? cmdSemanticKey(cmd) : cmd ? `${tool}:${cmd}` : null
  const cmdHashInt = cmdKey ? crc32(Buffer.from(cmdKey, 'utf8')) >>> 0 : null

  const cleanOutput = stripSystemReminders(output)
  const isEditTool = EDIT_TOOLS.has(tool)

  const outputSet = isEditTool ? new Set() : normalizeToSet(cleanOutput)
  const hasPrior = !isEditTool && outputHistory.has(cmdHashInt)
  const outputSim = isEditTool ? 0.0 : jaccard(outputSet, outputHistory.get(cmdHashInt) ?? null)

  const features = new Float32Array(7)
  features[0] = toolIdx
  features[1] = cmdHashInt !== null ? cmdHashInt * CRC32_NORM : 0.0
  features[2] = fileHashInt !== null ? fileHashInt * CRC32_NORM : 0.0
  features[3] = outputSim
  features[4] = hasPrior ? 1.0 : 0.0
  features[5] = Math.log1p(cleanOutput ? cleanOutput.split('\n').length - 1 : 0)
  features[6] = cleanOutput && ERROR_RE.test(cleanOutput.slice(0, 2000)) ? 1.0 : 0.0

  if (cmdHashInt !== null && !isEditTool) {
    outputHistory.set(cmdHashInt, outputSet)
  }

  return features
}

function stripSystemReminders(text) {
  if (!text || !text.includes('<system-reminder')) return text
  return text.replace(SYSTEM_REMINDER_RE, '')
}
