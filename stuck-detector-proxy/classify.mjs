/**
 * Pure JS stuck classifier — no Python dependency.
 *
 * Loads pre-trained logistic regression weights from model_weights.json.
 * Feature extraction is a direct port of classify.py.
 */

import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Load model weights once at import time
const weightsPath = process.env.STUCK_MODEL_PATH ||
  join(__dirname, "model_weights.json");
const model = JSON.parse(readFileSync(weightsPath, "utf-8"));

const { feature_names, scaler_mean, scaler_scale, coefficients, intercept } = model;

// ---------- Feature extraction (port of classify.py) ----------

function extractFeatures(text) {
  const feats = {};

  // max_substr_repeat: max repetition of 20-char substrings
  const seen = new Map();
  let maxRepeat = 0;
  for (let i = 0; i < text.length - 20; i += 10) {
    const sub = text.substring(i, i + 20);
    const count = (seen.get(sub) || 0) + 1;
    seen.set(sub, count);
    if (count > maxRepeat) maxRepeat = count;
  }
  feats.max_substr_repeat = maxRepeat;

  // vocab_diversity: unique words / total words
  const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
  const uniqueWords = new Set(words);
  feats.vocab_diversity = words.length > 0 ? uniqueWords.size / words.length : 0;

  // circle_kw: count of circular reasoning keywords
  const circlePattern = /\b(try again|let me try|another approach|actually,|wait,|hmm|let me reconsider|that didn.t work|same error|still failing|let me re-read|let me look again|I was wrong|no that.s not right)\b/gi;
  const circleMatches = text.match(circlePattern);
  feats.circle_kw = circleMatches ? circleMatches.length : 0;

  // false_starts: lines starting with backtracking phrases
  const fsMatches = text.match(/(?:^|\n)\s*(?:Actually|Wait|Hmm|No,|Let me)/g);
  feats.false_starts = fsMatches ? fsMatches.length : 0;

  // self_sim: word overlap between first and second half
  if (text.length > 200) {
    const half = Math.floor(text.length / 2);
    const w1 = new Set(text.slice(0, half).toLowerCase().split(/\s+/).filter(w => w.length > 0));
    const w2 = text.slice(half).toLowerCase().split(/\s+/).filter(w => w.length > 0);
    let overlap = 0;
    for (const w of w2) {
      if (w1.has(w)) overlap++;
    }
    feats.self_sim = w2.length > 0 ? overlap / w2.length : 0;
  } else {
    feats.self_sim = 0;
  }

  // avg_sent_len and sent_len_std
  const sents = text.split(/[.!?\n]+/).filter(s => s.trim().length > 0);
  const sentLens = sents.map(s => s.trim().split(/\s+/).length);
  if (sentLens.length > 0) {
    const mean = sentLens.reduce((a, b) => a + b, 0) / sentLens.length;
    feats.avg_sent_len = mean;
    const variance = sentLens.reduce((sum, l) => sum + (l - mean) ** 2, 0) / sentLens.length;
    feats.sent_len_std = Math.sqrt(variance);
  } else {
    feats.avg_sent_len = 0;
    feats.sent_len_std = 0;
  }

  // question_marks
  feats.question_marks = (text.match(/\?/g) || []).length;

  // code_ratio
  const codeChars = (text.match(/[{}\[\]();=<>]/g) || []).length;
  feats.code_ratio = text.length > 0 ? codeChars / text.length : 0;

  return feats;
}

// ---------- Logistic regression inference ----------

function sigmoid(x) {
  if (x > 500) return 1;
  if (x < -500) return 0;
  return 1 / (1 + Math.exp(-x));
}

/**
 * Classify text as stuck or productive.
 * @param {string} text - Thinking block text
 * @param {Object} [toolFeats] - Optional tool-call behavioral features from message history
 * @returns {{ score: number, label: string, reasons: string[] }}
 */
export function classify(text, toolFeats) {
  if (text.length < 100) {
    return { score: 0, label: "productive", reasons: [] };
  }

  const feats = extractFeatures(text);

  // Merge tool features if provided and model expects them
  if (toolFeats) {
    Object.assign(feats, toolFeats);
  }

  // Scale features: (x - mean) / std
  const scaled = feature_names.map((name, i) => {
    const val = feats[name] ?? 0;
    return (val - scaler_mean[i]) / scaler_scale[i];
  });

  // Dot product + intercept
  let logit = intercept;
  for (let i = 0; i < scaled.length; i++) {
    logit += scaled[i] * coefficients[i];
  }

  const score = sigmoid(logit);
  const label = score >= 0.5 ? "stuck" : "productive";

  // Top contributing features (same logic as classify.py)
  const reasons = [];
  for (let i = 0; i < feature_names.length; i++) {
    if (coefficients[i] > 0.5 && (feats[feature_names[i]] ?? 0) > 0) {
      reasons.push(feature_names[i]);
    }
  }

  return {
    score: Math.round(score * 1000) / 1000,
    label,
    reasons,
  };
}

/**
 * Expose feature extraction for debugging/analysis.
 */
export { extractFeatures };
