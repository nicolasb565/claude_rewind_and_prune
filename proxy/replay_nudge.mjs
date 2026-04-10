#!/usr/bin/env node
/**
 * Nudge replay via Claude Code — creates a synthetic session file truncated
 * at a stuck point with a nudge injected, then prints the `claude --resume`
 * command to run it in the full Claude Code environment (tools, system prompt,
 * extended thinking — everything the raw API test cannot provide).
 *
 * Usage:
 *   node proxy/replay_nudge.mjs /path/to/session.jsonl [options]
 *
 *   node proxy/replay_nudge.mjs ~/.claude/projects/my-proj/abc.jsonl --level 1
 *   # prints: claude --resume <new-uuid> --fork-session
 *
 * Options:
 *   --level  0|1|2   Nudge level (default: 0)
 *   --step   N       Truncate after tool call N (default: auto = peak CNN window)
 *   --out    <dir>   Write session file here (default: same dir as source)
 *   --run            Exec `claude --resume` immediately (interactive)
 */

import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { randomUUID } from "crypto";
import { dirname, join } from "path";
import { execSync } from "child_process";
import { StuckDetectorState } from "./abstract_step.mjs";
import { classifyWindow, normalizeFeatures, config } from "./classify_cnn.mjs";

const WINDOW_SIZE = config.window_size;

// ── CLI ──────────────────────────────────────────────────────────────────────

const argv = process.argv.slice(2);
const sessionFile = argv.find(a => !a.startsWith("-"));
if (!sessionFile) {
  console.log("Usage: node proxy/replay_nudge.mjs session.jsonl [--level 0|1|2] [--step N] [--out dir] [--run] [--list]");
  console.log("  --list   show all tool calls with CNN scores and exit (pick a --step N from this)");
  process.exit(0);
}

function getFlag(name, def) {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : def;
}

const nudgeLevel = parseInt(getFlag("level", "0"), 10);
const forcedStep = getFlag("step", null);
const outDir     = getFlag("out", dirname(sessionFile));
const autoRun    = argv.includes("--run");
const listMode   = argv.includes("--list");

// ── Parse session ─────────────────────────────────────────────────────────────
// We need two things:
//   1. Raw JSONL lines (to copy verbatim and preserve Claude Code's metadata)
//   2. Tool call list (for CNN scoring)
// We also sniff the format of an existing user message line to use as template
// for the injected nudge (preserving any extra fields like uuid, timestamp, etc.)

function parseSession(filepath) {
  const raw        = readFileSync(filepath, "utf-8").trim().split("\n");
  const toolCalls  = [];   // { name, input, output, lineIdx }
  const outputMap  = new Map();
  let   nudgeTemplate = null;   // raw line from a user message — used as format template

  // First pass: collect tool results + find a user message template
  for (const line of raw) {
    let entry; try { entry = JSON.parse(line); } catch { continue; }
    const msg = entry.message;
    if (!msg) continue;
    if (msg.role === "user" && Array.isArray(msg.content)) {
      // Use the first simple user text message as format template
      if (!nudgeTemplate && msg.content.some(b => b.type === "text")) {
        nudgeTemplate = entry;
      }
      for (const b of msg.content) {
        if (b.type !== "tool_result") continue;
        const c = b.content;
        outputMap.set(b.tool_use_id,
          Array.isArray(c) ? c.filter(x => x.type === "text").map(x => x.text).join(" ")
                           : String(c || ""));
      }
    }
  }

  // Second pass: build ordered tool call list with line indices
  let lastAssistantLine = -1;
  for (let i = 0; i < raw.length; i++) {
    let entry; try { entry = JSON.parse(raw[i]); } catch { continue; }
    const msg = entry.message;
    if (!msg || !Array.isArray(msg.content)) continue;
    if (msg.role === "assistant") {
      lastAssistantLine = i;
      for (const b of msg.content) {
        if (b.type !== "tool_use") continue;
        toolCalls.push({
          id: b.id, name: b.name, input: b.input || {},
          output: outputMap.get(b.id) || "",
          lineIdx: i,
        });
      }
    }
  }

  return { raw, toolCalls, nudgeTemplate };
}

// ── CNN scoring ───────────────────────────────────────────────────────────────

function scoreWindows(toolCalls) {
  const detector = new StuckDetectorState();
  for (const tc of toolCalls) detector.addStep(tc.name, tc.input, tc.output, "");
  const scores = [];
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
    scores.push({ start, end: start + WINDOW_SIZE - 1, score });
  }
  return scores;
}

// ── Nudge text ────────────────────────────────────────────────────────────────

function makeNudgeText(level, turnNum, score, recentList) {
  const pct = (score * 100).toFixed(0);
  if (level === 0) {
    return `[CONTEXT MONITOR — turn ${turnNum}, confidence ${pct}%]\n\n` +
      `Your recent actions show signs of repetitive patterns. You may be going in circles.\n\n` +
      `Recent tool calls:\n  ${recentList}\n\n` +
      `Review your last few turns critically:\n` +
      `- Are you retrying the same approach with minor variations?\n` +
      `- Are you investigating the same files/functions repeatedly?\n` +
      `- Has your hypothesis changed or are you stuck on the same one?\n\n` +
      `If you are going in circles, try a fundamentally different strategy.\n` +
      `State what you have learned so far and what new approach you will try.`;
  } else if (level === 1) {
    return `[CONTEXT MONITOR — turn ${turnNum}, confidence ${pct}% — repeated signal]\n\n` +
      `You have been nudged before and the repetitive pattern continues.\n\n` +
      `Recent tool calls:\n  ${recentList}\n\n` +
      `You appear to be stuck in a loop. The approach you are using is not working.\n` +
      `Before your next tool call:\n` +
      `1. State in one sentence what you have been trying to do.\n` +
      `2. State specifically why it has not worked.\n` +
      `3. Propose a different approach you have not tried yet.\n\n` +
      `Do not retry the same command. Switch strategy.`;
  } else {
    return `[CONTEXT MONITOR — turn ${turnNum}, confidence ${pct}% — escalated]\n\n` +
      `STOP. You are deeply stuck and have not responded to prior nudges.\n\n` +
      `Recent tool calls:\n  ${recentList}\n\n` +
      `Do not run any more tool calls until you have answered these:\n` +
      `1. What is the root cause of the problem you are trying to solve?\n` +
      `2. What have you tried, and why did each attempt fail?\n` +
      `3. What fundamentally different approach will you take next?\n\n` +
      `If you cannot answer these, state that clearly and ask for guidance.`;
  }
}

// ── Build nudge JSONL line ────────────────────────────────────────────────────
// Clone the template entry (preserving whatever metadata Claude Code uses),
// replace the message content with our nudge text.

function makeNudgeLine(template, nudgeText, sessionId) {
  // If we have a template entry from the original file, clone its structure
  const entry = template ? JSON.parse(JSON.stringify(template)) : {};
  entry.message = {
    role: "user",
    content: [{ type: "text", text: nudgeText }],
  };
  // Update metadata fields if present
  if ("uuid"      in entry) entry.uuid      = randomUUID();
  if ("timestamp" in entry) entry.timestamp = new Date().toISOString();
  if ("sessionId" in entry) entry.sessionId = sessionId;
  return JSON.stringify(entry);
}

// ── Main ──────────────────────────────────────────────────────────────────────

const { raw, toolCalls, nudgeTemplate } = parseSession(sessionFile);
const sessionName = sessionFile.split("/").pop().replace(".jsonl", "");

if (toolCalls.length < WINDOW_SIZE) {
  console.error(`Too few tool calls (${toolCalls.length} < ${WINDOW_SIZE})`);
  process.exit(1);
}

// Select cutoff
const windowScores = scoreWindows(toolCalls);

// ── List mode: show all tool calls with CNN scores and exit ───────────────────
if (listMode) {
  // Build a score per step: score of the window that ENDS at that step
  const scoreByEnd = new Map(windowScores.map(w => [w.end, w.score]));
  console.log(`${"step".padStart(4)}  ${"tool".padEnd(8)}  ${"cnn".padStart(5)}  input`);
  console.log("─".repeat(72));
  for (let i = 0; i < toolCalls.length; i++) {
    const tc    = toolCalls[i];
    const score = scoreByEnd.get(i);
    const v     = tc.input?.command || tc.input?.file_path || tc.input?.pattern || "";
    const bar   = score !== undefined
      ? (score >= config.threshold ? "▓" : score >= 0.5 ? "░" : " ") + score.toFixed(2)
      : "     ";
    console.log(`${String(i).padStart(4)}  ${tc.name.padEnd(8)}  ${bar.padStart(5)}  ${String(v).replace(/\n/g, " ").slice(0, 50)}`);
  }
  console.log(`\n${toolCalls.length} tool calls total. Peak CNN window ends at step with ▓.`);
  console.log(`Re-run with --step N to inject the nudge at step N.`);
  process.exit(0);
}

let cutoffStep, cnnScore;

if (forcedStep !== null) {
  cutoffStep = Math.min(parseInt(forcedStep, 10), toolCalls.length - 1);
  const w = windowScores.find(s => s.end === cutoffStep);
  cnnScore = w?.score ?? 0;
} else {
  const best = windowScores.reduce((a, b) => b.score > a.score ? b : a);
  cutoffStep = best.end;
  cnnScore   = best.score;
}

// Find the JSONL line that follows the assistant turn containing the cutoff step.
// We want to include the user tool_result message after the assistant turn,
// so we cut after the first user message that comes after cutoffStep's line.
const cutoffLineIdx = toolCalls[cutoffStep].lineIdx;
let cutoffRawLine = cutoffLineIdx;
for (let i = cutoffLineIdx + 1; i < raw.length; i++) {
  let entry; try { entry = JSON.parse(raw[i]); } catch { continue; }
  if (entry.message?.role === "user") { cutoffRawLine = i; break; }
}

// Build recent tool summary for nudge text
const recentTcs  = toolCalls.slice(Math.max(0, cutoffStep - 7), cutoffStep + 1);
const recentList = recentTcs.map(tc => {
  const v = tc.input?.command || tc.input?.file_path || tc.input?.pattern || "";
  return `${tc.name}: ${String(v).replace(/\n/g, " ").slice(0, 52)}`;
}).join("\n  ");

const nudgeText = makeNudgeText(nudgeLevel, cutoffStep + 1, cnnScore, recentList);
const newUuid   = randomUUID();
const nudgeLine = makeNudgeLine(nudgeTemplate, nudgeText, newUuid);

// Truncate and append nudge
const outputLines = [...raw.slice(0, cutoffRawLine + 1), nudgeLine];

// Write new session file
mkdirSync(outDir, { recursive: true });
const outPath = join(outDir, `${newUuid}.jsonl`);
writeFileSync(outPath, outputLines.join("\n") + "\n");

// ── Report ────────────────────────────────────────────────────────────────────

const testMode = cnnScore < config.threshold ? "harmlessness (productive zone)" : "effectiveness (stuck zone)";
const levelNames = ["soft", "medium", "hard"];

console.log(`Source:   ${sessionFile}`);
console.log(`Cutoff:   step ${cutoffStep} / ${toolCalls.length - 1}  (line ${cutoffRawLine} / ${raw.length - 1})`);
console.log(`CNN:      ${cnnScore.toFixed(3)}  (threshold ${config.threshold})  →  ${testMode}`);
console.log(`Nudge:    level ${nudgeLevel} / ${levelNames[nudgeLevel]}`);
console.log(`Session:  ${outPath}`);
console.log();
console.log(`Resume with (interactive — runs to completion):`);
console.log(`  claude --resume ${newUuid} --fork-session`);
console.log(`Resume with (one response then exit — cheaper):`);
console.log(`  claude --resume ${newUuid} --fork-session --print`);
console.log();
console.log(`Recent tool calls before nudge:`);
for (const tc of recentTcs) {
  const v = tc.input?.command || tc.input?.file_path || tc.input?.pattern || "";
  console.log(`  ${tc.name}: ${String(v).replace(/\n/g, " ").slice(0, 60)}`);
}

if (autoRun) {
  console.log("\nRunning claude --resume ...");
  execSync(`claude --resume ${newUuid} --fork-session`, { stdio: "inherit" });
}
