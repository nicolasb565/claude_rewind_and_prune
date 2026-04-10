/**
 * End-to-end test: parse a real Claude Code session JSONL, run CNN, report scores.
 *
 * Usage:
 *   node proxy/test_e2e.mjs /path/to/session.jsonl [/path/to/session2.jsonl ...]
 *
 * With no arguments, prints usage and exits.
 */

import { readFileSync } from "fs";
import { StuckDetectorState } from "./abstract_step.mjs";
import { classifyWindow, normalizeFeatures, config } from "./classify_cnn.mjs";

const WINDOW_SIZE = config.window_size;

function parseSession(filepath) {
  const lines = readFileSync(filepath, "utf-8").trim().split("\n");
  const toolCalls = [];

  for (const line of lines) {
    const entry = JSON.parse(line);
    if (!entry.message) continue;
    const msg = entry.message;

    if (msg.role === "assistant" && Array.isArray(msg.content)) {
      let thinking = "";
      for (const block of msg.content) {
        if (block.type === "thinking") {
          thinking = block.thinking || "";
        } else if (block.type === "tool_use") {
          toolCalls.push({ id: block.id, name: block.name, input: block.input || {}, output: "", thinking });
          thinking = "";
        }
      }
    } else if (msg.role === "user" && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === "tool_result") {
          const tc = toolCalls.find(t => t.id === block.tool_use_id);
          if (tc) {
            const content = block.content;
            tc.output = Array.isArray(content)
              ? content.filter(b => b.type === "text").map(b => b.text).join(" ")
              : String(content || "");
          }
        }
      }
    }
  }
  return toolCalls;
}

function analyzeSession(filepath) {
  const toolCalls = parseSession(filepath);
  const detector = new StuckDetectorState();

  console.log(`Session: ${filepath.split("/").pop()}`);
  console.log(`  Tool calls: ${toolCalls.length}`);

  if (toolCalls.length < WINDOW_SIZE) {
    console.log("  Too few tool calls for CNN analysis\n");
    return;
  }

  for (const tc of toolCalls) {
    detector.addStep(tc.name, tc.input, tc.output, tc.thinking);
  }

  const scores = [];
  for (let start = 0; start <= detector.abstractSteps.length - WINDOW_SIZE; start += 5) {
    // Use getWindow() which returns correctly formatted 11-feature arrays
    const savedSteps = detector.abstractSteps;
    const savedHist = {
      tool: detector.toolHistory,
      file: detector.fileHashHistory,
      cmd: detector.cmdHashHistory,
    };

    const window = detector.abstractSteps.slice(start, start + WINDOW_SIZE);
    const toolIndices = window.map(s => s.tool_idx);
    const continuous = window.map(s =>
      config.continuous_features.map(f => s[f] ?? 0)
    );
    const normalizedCont = continuous.map(row => normalizeFeatures(row));

    // Window-level features
    const tools = window.map(s => s.tool);
    const allLines = [];
    for (const s of window) if (s.output_set) for (const l of s.output_set) allLines.push(l);
    const windowFeatures = [
      new Set(tools).size / tools.length,
      1.0, 1.0,  // file/cmd ratios — approximate
      window.reduce((a, s) => a + s.is_error, 0) / window.length,
      window.reduce((a, s) => a + s.output_similarity, 0) / window.length,
      allLines.length > 0 ? new Set(allLines).size / allLines.length : 1.0,
    ];

    const { score } = classifyWindow(toolIndices, normalizedCont, windowFeatures);
    scores.push({ start, score });
  }

  const maxScore = Math.max(...scores.map(s => s.score));
  const meanScore = scores.reduce((a, s) => a + s.score, 0) / scores.length;
  const stuckWindows = scores.filter(s => s.score >= config.threshold).length;

  console.log(`  Windows: ${scores.length}`);
  console.log(`  Scores: mean=${meanScore.toFixed(4)} max=${maxScore.toFixed(4)}`);
  console.log(`  Stuck windows (>=${config.threshold}): ${stuckWindows}`);

  const top3 = [...scores].sort((a, b) => b.score - a.score).slice(0, 3);
  console.log(`  Top 3 windows:`);
  for (const s of top3) {
    const w = detector.abstractSteps.slice(s.start, s.start + WINDOW_SIZE);
    console.log(`    start=${s.start} score=${s.score.toFixed(4)} tools=[${w.map(st => st.tool).join(" → ")}]`);
  }
  console.log();
}

const args = process.argv.slice(2);
if (args.length === 0) {
  console.log("Usage: node proxy/test_e2e.mjs /path/to/session.jsonl [...]");
  console.log("  Parses Claude Code session files and scores each one with the CNN.");
  process.exit(0);
}

for (const filepath of args) {
  analyzeSession(filepath);
}
