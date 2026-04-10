/**
 * Pure JS CNN inference for stuck detection.
 * No dependencies — loads weights from cnn_weights.json.
 *
 * Architecture: tool_embed(7,4) → [conv3(19,16,k3) + conv5(19,16,k5)] → maxpool → concat(32+6) → fc1(38,16) → fc2(16,1)
 */

import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const weightsPath = join(__dirname, "cnn_weights.json");
const configPath = join(__dirname, "cnn_config.json");

const W = JSON.parse(readFileSync(weightsPath, "utf-8"));
const config = JSON.parse(readFileSync(configPath, "utf-8"));

const toolEmbed = W["tool_embed.weight"];   // (7, 4)
const conv3W = W["conv3.weight"];           // (16, 19, 3)
const conv3B = W["conv3.bias"];             // (16,)
const conv5W = W["conv5.weight"];           // (16, 19, 5)
const conv5B = W["conv5.bias"];             // (16,)
const fc1W = W["fc1.weight"];              // (16, 38)
const fc1B = W["fc1.bias"];               // (16,)
const fc2W = W["fc2.weight"];             // (1, 16)
const fc2B = W["fc2.bias"];              // (1,)
const normMean = W["norm_mean"];          // (15,)
const normStd = W["norm_std"];            // (15,)

const WINDOW_SIZE = config.window_size;     // 10
const TOOL_EMBED_DIM = config.tool_embed_dim; // 4
const NUM_CONTINUOUS = config.num_continuous;  // 15
const STEP_DIM = TOOL_EMBED_DIM + NUM_CONTINUOUS; // 19

/**
 * Conv1d: input (seqLen, inC), weight (outC, inC, kernelSize), bias (outC)
 * Returns (seqLen, outC) with same-padding.
 */
function conv1d(input, weight, bias, kernelSize) {
  const seqLen = input.length;
  const inC = input[0].length;
  const outC = weight.length;
  const pad = Math.floor(kernelSize / 2);

  const output = new Array(seqLen);
  for (let t = 0; t < seqLen; t++) {
    output[t] = new Float64Array(outC);
    for (let oc = 0; oc < outC; oc++) {
      let sum = bias[oc];
      for (let k = 0; k < kernelSize; k++) {
        const ti = t + k - pad;
        if (ti < 0 || ti >= seqLen) continue;
        for (let ic = 0; ic < inC; ic++) {
          sum += input[ti][ic] * weight[oc][ic][k];
        }
      }
      output[t][oc] = sum;
    }
  }
  return output;
}

/**
 * Global max pooling over sequence dimension.
 * Input: (seqLen, channels) → Output: (channels,)
 */
function globalMaxPool(input) {
  const channels = input[0].length;
  const result = new Float64Array(channels).fill(-Infinity);
  for (let t = 0; t < input.length; t++) {
    for (let c = 0; c < channels; c++) {
      if (input[t][c] > result[c]) result[c] = input[t][c];
    }
  }
  return result;
}

function relu(arr) {
  return arr.map(v => Math.max(0, v));
}

function dense(input, weight, bias) {
  const outDim = weight.length;
  const result = new Float64Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let sum = bias[o];
    for (let i = 0; i < input.length; i++) {
      sum += input[i] * weight[o][i];
    }
    result[o] = sum;
  }
  return result;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Classify a window of 10 steps.
 *
 * @param {number[]} toolIndices - (10,) tool category indices (0-6)
 * @param {number[][]} continuousFeatures - (10, 15) normalized continuous features
 * @param {number[]} windowFeatures - (6,) window-level features
 * @returns {{ score: number, stuck: boolean }}
 */
export function classifyWindow(toolIndices, continuousFeatures, windowFeatures) {
  // 1. Embed tools and concat with continuous features → (10, 19)
  const embedded = new Array(WINDOW_SIZE);
  for (let t = 0; t < WINDOW_SIZE; t++) {
    const emb = toolEmbed[toolIndices[t]]; // (4,)
    const cont = continuousFeatures[t];     // (15,)
    embedded[t] = new Float64Array(STEP_DIM);
    for (let i = 0; i < TOOL_EMBED_DIM; i++) embedded[t][i] = emb[i];
    for (let i = 0; i < NUM_CONTINUOUS; i++) embedded[t][TOOL_EMBED_DIM + i] = cont[i];
  }

  // 2. Conv1d branches + ReLU + global max pool
  const c3 = relu(globalMaxPool(conv1d(embedded, conv3W, conv3B, 3)));  // (16,)
  const c5 = relu(globalMaxPool(conv1d(embedded, conv5W, conv5B, 5)));  // (16,)

  // 3. Concat conv outputs + window features → (38,)
  const pooled = new Float64Array(32 + 6);
  for (let i = 0; i < 16; i++) pooled[i] = c3[i];
  for (let i = 0; i < 16; i++) pooled[16 + i] = c5[i];
  for (let i = 0; i < 6; i++) pooled[32 + i] = windowFeatures[i];

  // 4. FC layers (no dropout at inference)
  const hidden = relu(dense(pooled, fc1W, fc1B));  // (16,)
  const logit = dense(hidden, fc2W, fc2B)[0];       // scalar

  const score = sigmoid(logit);
  return { score, stuck: score >= config.threshold };
}

/**
 * Normalize continuous features using training stats.
 */
export function normalizeFeatures(raw) {
  return raw.map((v, i) => (v - normMean[i]) / normStd[i]);
}

export { config };
