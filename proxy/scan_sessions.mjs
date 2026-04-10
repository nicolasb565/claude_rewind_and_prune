#!/usr/bin/env node
/**
 * Scan Claude Code session files and score them with the CNN.
 * Prints paths of sessions that exceeded the stuck threshold.
 *
 * Usage:
 *   node proxy/scan_sessions.mjs [--dir ~/.claude/projects] [--threshold 0.96] [--all]
 *
 * Options:
 *   --dir <path>       Root to scan (default: ~/.claude/projects)
 *   --threshold <f>    Min CNN score to report (default: config.threshold)
 *   --min-steps <n>    Skip sessions with fewer than N tool calls (default: 20)
 *   --all              Print all sessions regardless of score
 *   --verbose          Show per-session score details on stderr (use 2>&1 | grep to filter)
 */

import { readFileSync, readdirSync, statSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import { StuckDetectorState } from "./abstract_step.mjs";
import { classifyWindow, normalizeFeatures, config } from "./classify_cnn.mjs";

const WINDOW_SIZE = config.window_size;

// ── CLI ──────────────────────────────────────────────────────────────────────

const argv = process.argv.slice(2);
function getFlag(name, def) {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : def;
}

const scanDir   = getFlag("dir", join(homedir(), ".claude", "projects"));
const threshold = parseFloat(getFlag("threshold", String(config.threshold)));
const minSteps  = parseInt(getFlag("min-steps", "20"), 10);
const showAll   = argv.includes("--all");
const verbose   = argv.includes("--verbose");

// ── Session scoring ───────────────────────────────────────────────────────────

function scoreSession(filepath) {
  let lines;
  try {
    lines = readFileSync(filepath, "utf-8").trim().split("\n");
  } catch {
    return null;
  }

  const outputMap = new Map();
  for (const line of lines) {
    let entry; try { entry = JSON.parse(line); } catch { continue; }
    const msg = entry.message;
    if (!msg || msg.role !== "user" || !Array.isArray(msg.content)) continue;
    for (const b of msg.content) {
      if (b.type !== "tool_result") continue;
      const c = b.content;
      outputMap.set(b.tool_use_id,
        Array.isArray(c) ? c.filter(x => x.type === "text").map(x => x.text).join(" ")
                         : String(c || ""));
    }
  }

  const detector = new StuckDetectorState();
  for (const line of lines) {
    let entry; try { entry = JSON.parse(line); } catch { continue; }
    const msg = entry.message;
    if (!msg || msg.role !== "assistant" || !Array.isArray(msg.content)) continue;
    for (const b of msg.content) {
      if (b.type !== "tool_use") continue;
      detector.addStep(b.name, b.input || {}, outputMap.get(b.id) || "", "");
    }
  }

  if (detector.abstractSteps.length < minSteps) return null;

  let maxScore = 0;
  let firedCount = 0;
  for (let start = 0; start <= detector.abstractSteps.length - WINDOW_SIZE; start++) {
    const window   = detector.abstractSteps.slice(start, start + WINDOW_SIZE);
    const toolIdxs = window.map(s => s.tool_idx);
    const cont     = window.map(s => config.continuous_features.map(f => s[f] ?? 0));
    const normCont = cont.map(row => normalizeFeatures(row));
    const tools    = window.map(s => s.tool);
    const allLines = [];
    for (const s of window) if (s.output_set) for (const l of s.output_set) allLines.push(l);
    const wf = [
      new Set(tools).size / tools.length, 1.0, 1.0,
      window.reduce((a, s) => a + s.is_error, 0) / window.length,
      window.reduce((a, s) => a + s.output_similarity, 0) / window.length,
      allLines.length > 0 ? new Set(allLines).size / allLines.length : 1.0,
    ];
    const { score } = classifyWindow(toolIdxs, normCont, wf);
    if (score > maxScore) maxScore = score;
    if (score >= threshold) firedCount++;
  }

  return { maxScore, firedCount, steps: detector.abstractSteps.length };
}

// ── Scan ──────────────────────────────────────────────────────────────────────

function findJsonlFiles(dir) {
  const results = [];
  try {
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry);
      try {
        const st = statSync(full);
        if (st.isDirectory()) results.push(...findJsonlFiles(full));
        else if (entry.endsWith(".jsonl")) results.push(full);
      } catch { /* skip unreadable */ }
    }
  } catch { /* skip unreadable dir */ }
  return results;
}

// Derive the session's original CWD from the Claude project dir name.
// Claude encodes cwd as: absolute path with '/' replaced by '-', leading '/'->''.
// e.g. -home-nicolas-source-foo → /home/nicolas/source/foo
function deriveCwd(filepath, projectsRoot) {
  const rel = filepath.slice(projectsRoot.length + 1); // strip projects root + sep
  const projectDir = rel.split("/")[0];                 // first segment = project dir
  // project dir starts with '-' representing the leading '/'
  return projectDir.replace(/-/g, "/").replace(/^\//, "/");
}

const files = findJsonlFiles(scanDir);
if (files.length === 0) {
  console.log(`No .jsonl files found in ${scanDir}`);
  process.exit(1);
}

if (verbose) {
  console.log(`Scanning ${files.length} session files in ${scanDir} ...`);
  console.log(`Threshold: ${threshold}  min-steps: ${minSteps}\n`);
}

let found = 0;
for (const file of files) {
  const result = scoreSession(file);
  if (!result) continue;

  const { maxScore, firedCount, steps } = result;
  if (!showAll && maxScore < threshold) continue;

  found++;
  const tag = maxScore >= threshold ? " <stuck>" : "";
  const cwd = deriveCwd(file, scanDir);
  if (verbose) {
    console.log(`  max=${maxScore.toFixed(3)} fired=${firedCount} steps=${steps}  cwd=${cwd}${tag}`);
    console.log(`  ${file}`);
  } else {
    console.log(file);
  }
}

if (verbose) {
  console.log(`\n${found} session(s) found (out of ${files.length} scanned)`);
}
