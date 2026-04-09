/**
 * Abstract Claude Code messages into feature vectors for the CNN.
 *
 * Maps Claude Code tool names to abstract categories, hashes commands/files
 * with CRC32, computes Jaccard output similarity, and tracks history for
 * cycle detection features.
 */

import { crc32 } from "node:zlib";

// Tool name mapping: Claude Code → abstract category
const TOOL_MAP = {
  Bash: "bash", bash: "bash",
  Read: "view", read: "view",
  Edit: "edit", edit: "edit", Write: "edit", write: "edit", MultiEdit: "edit",
  Grep: "search", grep: "search", Glob: "search", glob: "search",
  Agent: "other", Task: "other", TodoRead: "other", TodoWrite: "other",
};

const TOOL_NAMES = ["bash", "edit", "view", "search", "create", "submit", "other"];
const TOOL_TO_IDX = Object.fromEntries(TOOL_NAMES.map((t, i) => [t, i]));

// Feature names (must match training order)
const CONTINUOUS_FEATURES = [
  "steps_since_same_tool", "steps_since_same_file", "steps_since_same_cmd",
  "tool_count_in_window", "file_count_in_window", "cmd_count_in_window",
  "output_similarity", "output_length", "is_error", "step_index_norm",
  "false_start", "strategy_change", "circular_lang",
  "thinking_length", "self_similarity",
];

const WINDOW_FEATURES = [
  "unique_tools_ratio", "unique_files_ratio", "unique_cmds_ratio",
  "error_rate", "output_similarity_avg", "output_diversity",
];

// Regex patterns
const ERROR_RE = /error|traceback|exception|failed|failure|fatal|cannot|unable to|not found|permission denied|segmentation fault|FAIL|ModuleNotFoundError|ImportError|SyntaxError|TypeError|ValueError|KeyError|AttributeError|RuntimeError|FileNotFoundError/i;
const FALSE_START_RE = /\b(actually|wait|hmm|let me reconsider|on second thought)\b/i;
const STRATEGY_RE = /\b(different approach|try another|instead|alternatively|let me try a different)\b/i;
const CIRCULAR_RE = /\b(try again|let me try|one more time|retry|attempt again)\b/i;

const MAX_OUTPUT_LINES = 100;

function hash(str) {
  if (!str) return null;
  return crc32(Buffer.from(str));
}

/**
 * Normalize output to a Set of lines for Jaccard comparison.
 */
function normalizeToSet(output) {
  if (!output) return new Set();
  const lines = output.trim().split("\n").slice(0, MAX_OUTPUT_LINES);
  const normalized = new Set();
  for (let line of lines) {
    line = line.replace(/0x[0-9a-fA-F]+/g, "0xADDR");
    line = line.replace(/\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}/g, "TIMESTAMP");
    line = line.replace(/pid[=: ]\d+/gi, "pid=PID");
    line = line.replace(/\/tmp\/[^\s]+/g, "/tmp/TMPFILE");
    line = line.replace(/\d+\.\d{3,}s/g, "N.NNNs");
    line = line.trim();
    if (line) normalized.add(line);
  }
  return normalized;
}

function jaccard(setA, setB) {
  if (!setB) return 0.5; // neutral — no prior comparison
  if (setA.size === 0 && setB.size === 0) return 1.0;
  const union = new Set([...setA, ...setB]);
  if (union.size === 0) return 1.0;
  let intersection = 0;
  for (const v of setA) if (setB.has(v)) intersection++;
  return intersection / union.size;
}

function stepsSince(value, history) {
  if (value === null) return history.length;
  for (let j = history.length - 1; j >= 0; j--) {
    if (history[j] === value) return history.length - j;
  }
  return history.length;
}

function countIn(value, history) {
  if (value === null) return 0;
  let count = 0;
  for (const h of history) if (h === value) count++;
  return count;
}

function wordOverlap(text1, text2) {
  if (!text1 || !text2) return 0;
  const w1 = new Set(text1.toLowerCase().split(/\s+/));
  const w2 = new Set(text2.toLowerCase().split(/\s+/));
  if (w1.size === 0 || w2.size === 0) return 0;
  let overlap = 0;
  for (const w of w1) if (w2.has(w)) overlap++;
  return overlap / Math.sqrt(w1.size * w2.size);
}

/**
 * Maintains sliding window state for a session.
 */
export class StuckDetectorState {
  constructor() {
    this.toolHistory = [];
    this.fileHashHistory = [];
    this.cmdHashHistory = [];
    this.outputHistory = new Map(); // cmdHash → Set
    this.prevThinking = null;
    this.stepCount = 0;
    this.abstractSteps = []; // rolling buffer of abstract steps
  }

  /**
   * Process a tool call from a Claude Code message.
   * Returns the abstract step features.
   */
  addStep(toolName, input, output, thinking) {
    const tool = TOOL_MAP[toolName] || "other";
    const toolIdx = TOOL_TO_IDX[tool] ?? 6;

    const cmd = input?.command || input?.file_path || input?.pattern || "";
    const filePath = input?.file_path || input?.path || null;

    const fileHash = hash(filePath);
    const cmdHash = hash(cmd);
    const outputSet = normalizeToSet(output);
    const outputSim = jaccard(outputSet, this.outputHistory.get(cmdHash) || null);

    const totalSteps = Math.max(this.stepCount + 1, 1);
    const i = this.stepCount;

    const step = {
      tool,
      tool_idx: toolIdx,
      steps_since_same_tool: stepsSince(tool, this.toolHistory) / totalSteps,
      steps_since_same_file: stepsSince(fileHash, this.fileHashHistory) / totalSteps,
      steps_since_same_cmd: stepsSince(cmdHash, this.cmdHashHistory) / totalSteps,
      tool_count_in_window: countIn(tool, this.toolHistory) / Math.max(i + 1, 1),
      file_count_in_window: countIn(fileHash, this.fileHashHistory) / Math.max(i + 1, 1),
      cmd_count_in_window: countIn(cmdHash, this.cmdHashHistory) / Math.max(i + 1, 1),
      output_similarity: outputSim,
      output_set: outputSet,
      output_length: Math.log1p(output ? output.split("\n").length : 0),
      is_error: output ? (ERROR_RE.test(output.slice(0, 2000)) ? 1.0 : 0.0) : 0.0,
      step_index_norm: i / Math.max(totalSteps - 1, 1),
      false_start: thinking && FALSE_START_RE.test(thinking) ? 1.0 : 0.0,
      strategy_change: thinking && STRATEGY_RE.test(thinking) ? 1.0 : 0.0,
      circular_lang: thinking && CIRCULAR_RE.test(thinking) ? 1.0 : 0.0,
      thinking_length: Math.log1p(thinking ? thinking.length : 0),
      self_similarity: wordOverlap(thinking, this.prevThinking),
    };

    // Update histories
    this.toolHistory.push(tool);
    this.fileHashHistory.push(fileHash);
    this.cmdHashHistory.push(cmdHash);
    if (cmdHash !== null) this.outputHistory.set(cmdHash, outputSet);
    this.prevThinking = thinking;
    this.stepCount++;
    this.abstractSteps.push(step);

    return step;
  }

  /**
   * Get the last N steps as a window ready for CNN classification.
   * Returns null if not enough steps.
   */
  getWindow(windowSize = 10) {
    if (this.abstractSteps.length < windowSize) return null;

    const window = this.abstractSteps.slice(-windowSize);

    // Tool indices
    const toolIndices = window.map(s => s.tool_idx);

    // Continuous features (raw, will be normalized by classify_cnn)
    const continuous = window.map(s =>
      CONTINUOUS_FEATURES.map(f => s[f])
    );

    // Window-level features
    const tools = window.map(s => s.tool);
    const uniqueToolsRatio = new Set(tools).size / tools.length;

    const fileHashes = this.fileHashHistory.slice(-windowSize).filter(h => h !== null);
    const uniqueFilesRatio = fileHashes.length > 0
      ? new Set(fileHashes).size / fileHashes.length : 1.0;

    const cmdHashes = this.cmdHashHistory.slice(-windowSize).filter(h => h !== null);
    const uniqueCmdsRatio = cmdHashes.length > 0
      ? new Set(cmdHashes).size / cmdHashes.length : 1.0;

    const errorRate = window.reduce((a, s) => a + s.is_error, 0) / window.length;
    const outputSimAvg = window.reduce((a, s) => a + s.output_similarity, 0) / window.length;

    // Output diversity: unique lines / total lines across all outputs
    const allLines = [];
    for (const s of window) {
      if (s.output_set) {
        for (const line of s.output_set) allLines.push(line);
      }
    }
    const outputDiversity = allLines.length > 0
      ? new Set(allLines).size / allLines.length : 1.0;

    const windowFeatures = [
      uniqueToolsRatio, uniqueFilesRatio, uniqueCmdsRatio,
      errorRate, outputSimAvg, outputDiversity,
    ];

    return { toolIndices, continuous, windowFeatures };
  }
}

export { TOOL_MAP, TOOL_TO_IDX, TOOL_NAMES, CONTINUOUS_FEATURES, WINDOW_FEATURES };
