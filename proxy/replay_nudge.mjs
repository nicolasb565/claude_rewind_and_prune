#!/usr/bin/env node
/**
 * Nudge replay via Claude Code.
 *
 * Finds the peak stuck window in a session, injects a nudge, and spawns
 * `claude --resume <session-id> --print "<nudge>"` from the session's
 * original cwd — giving the full Claude Code environment (tools, system
 * prompt, extended thinking).
 *
 * Usage:
 *   node proxy/replay_nudge.mjs session.jsonl [options]
 *
 * Options:
 *   --level  0|1|2   Nudge level: 0=soft 1=medium 2=hard (default: 0)
 *   --step   N       Inject after tool call N (default: auto = peak CNN window)
 *   --list           Show all tool calls with CNN scores and exit
 *   --list-threshold F  With --list, hide steps below this score (default: 0)
 *   --dry-run        Print the claude command but don't execute it
 */

import { readFileSync } from "fs";
import { spawn }        from "child_process";
import { basename }     from "path";
import { StuckDetectorState } from "./abstract_step.mjs";
import { classifyWindow, normalizeFeatures, config } from "./classify_cnn.mjs";

const WINDOW_SIZE = config.window_size;

// ── CLI ───────────────────────────────────────────────────────────────────────

const argv = process.argv.slice(2);
const sessionFile = argv.find(a => !a.startsWith("-"));
if (!sessionFile) {
  console.log("Usage: node proxy/replay_nudge.mjs session.jsonl [--level 0|1|2] [--step N] [--list [--list-threshold F]] [--dry-run]");
  process.exit(0);
}

function getFlag(name, def) {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : def;
}

const nudgeLevel     = parseInt(getFlag("level", "0"), 10);
const forcedStep     = getFlag("step", null);
const listMode       = argv.includes("--list");
const listThreshold  = parseFloat(getFlag("list-threshold", "0.0"));
const dryRun         = argv.includes("--dry-run");

// ── Parse session ─────────────────────────────────────────────────────────────

function parseSession(filepath) {
  const raw       = readFileSync(filepath, "utf-8").trim().split("\n");
  const toolCalls = [];
  const outputMap = new Map();
  let sessionId   = basename(filepath, ".jsonl");
  let sessionCwd  = null;

  // First pass: collect tool results, cwd, session id
  for (const line of raw) {
    let entry; try { entry = JSON.parse(line); } catch { continue; }
    if (entry.sessionId && !sessionId) sessionId = entry.sessionId;
    if (entry.cwd && !sessionCwd)      sessionCwd = entry.cwd;
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

  // Extract session id from entries if filename is a uuid (normal case)
  for (const line of raw) {
    let entry; try { entry = JSON.parse(line); } catch { continue; }
    if (entry.sessionId) { sessionId = entry.sessionId; break; }
    if (entry.cwd && !sessionCwd) sessionCwd = entry.cwd;
  }

  // Second pass: ordered tool call list with line indices
  for (let i = 0; i < raw.length; i++) {
    let entry; try { entry = JSON.parse(raw[i]); } catch { continue; }
    if (!entry.cwd && entry.cwd !== undefined) sessionCwd ??= entry.cwd;
    const msg = entry.message;
    if (!msg || msg.role !== "assistant" || !Array.isArray(msg.content)) continue;
    for (const b of msg.content) {
      if (b.type !== "tool_use") continue;
      toolCalls.push({
        id: b.id, name: b.name, input: b.input || {},
        output: outputMap.get(b.id) || "",
        lineIdx: i,
      });
    }
  }

  return { toolCalls, sessionId, sessionCwd };
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

// ── Main ──────────────────────────────────────────────────────────────────────

const { toolCalls, sessionId, sessionCwd } = parseSession(sessionFile);

if (toolCalls.length < WINDOW_SIZE) {
  console.error(`Too few tool calls (${toolCalls.length} < ${WINDOW_SIZE})`);
  process.exit(1);
}

const windowScores = scoreWindows(toolCalls);

// ── List mode ─────────────────────────────────────────────────────────────────

if (listMode) {
  const scoreByEnd = new Map(windowScores.map(w => [w.end, w.score]));
  console.log(`${"step".padStart(4)}  ${"tool".padEnd(8)}  ${"cnn".padStart(5)}  input`);
  console.log("─".repeat(72));
  let shown = 0;
  for (let i = 0; i < toolCalls.length; i++) {
    const tc    = toolCalls[i];
    const score = scoreByEnd.get(i);
    if (score === undefined || score < listThreshold) continue;
    const v   = tc.input?.command || tc.input?.file_path || tc.input?.pattern || "";
    const bar = (score >= config.threshold ? "▓" : score >= 0.5 ? "░" : " ") + score.toFixed(2);
    console.log(`${String(i).padStart(4)}  ${tc.name.padEnd(8)}  ${bar.padStart(5)}  ${String(v).replace(/\n/g, " ").slice(0, 50)}`);
    shown++;
  }
  console.log(`\n${shown} / ${toolCalls.length} steps shown (--list-threshold ${listThreshold}). ▓ = above fire threshold.`);
  console.log(`Re-run with --step N to inject nudge at that step.`);
  process.exit(0);
}

// ── Select cutoff ─────────────────────────────────────────────────────────────

let cutoffStep, cnnScore;
if (forcedStep !== null) {
  cutoffStep = Math.min(parseInt(forcedStep, 10), toolCalls.length - 1);
  cnnScore   = windowScores.find(s => s.end === cutoffStep)?.score ?? 0;
} else {
  const best = windowScores.reduce((a, b) => b.score > a.score ? b : a);
  cutoffStep = best.end;
  cnnScore   = best.score;
}

const recentTcs  = toolCalls.slice(Math.max(0, cutoffStep - 7), cutoffStep + 1);
const recentList = recentTcs.map(tc => {
  const v = tc.input?.command || tc.input?.file_path || tc.input?.pattern || "";
  return `${tc.name}: ${String(v).replace(/\n/g, " ").slice(0, 52)}`;
}).join("\n  ");

const nudgeText  = makeNudgeText(nudgeLevel, cutoffStep + 1, cnnScore, recentList);
const levelNames = ["soft", "medium", "hard"];
const testMode   = cnnScore < config.threshold ? "harmlessness (productive zone)" : "effectiveness (stuck zone)";

// ── Report ────────────────────────────────────────────────────────────────────

console.log(`Source:   ${sessionFile}`);
console.log(`Cutoff:   step ${cutoffStep} / ${toolCalls.length - 1}`);
console.log(`CNN:      ${cnnScore.toFixed(3)}  (threshold ${config.threshold})  →  ${testMode}`);
console.log(`Nudge:    level ${nudgeLevel} / ${levelNames[nudgeLevel]}`);
console.log(`Cwd:      ${sessionCwd ?? "(unknown)"}`);
console.log(`Session:  ${sessionId}`);
console.log();
console.log(`Recent tool calls before nudge:`);
for (const tc of recentTcs) {
  const v = tc.input?.command || tc.input?.file_path || tc.input?.pattern || "";
  console.log(`  ${tc.name}: ${String(v).replace(/\n/g, " ").slice(0, 60)}`);
}
console.log();

// ── Spawn claude ──────────────────────────────────────────────────────────────

const claudeArgs = ["--resume", sessionId, "--print", nudgeText,
                    "--output-format", "stream-json", "--verbose"];

if (dryRun) {
  console.log(`Would run (from ${sessionCwd}):`);
  console.log(`  claude --resume ${sessionId} --print "<nudge text>"`);
  process.exit(0);
}

if (!sessionCwd) {
  console.error("Could not determine session cwd — use --dry-run to see the command and run it manually.");
  process.exit(1);
}

console.log(`Spawning claude from ${sessionCwd} ...\n`);

const child = spawn("claude", claudeArgs, {
  cwd:   sessionCwd,
  stdio: ["ignore", "pipe", "inherit"],
});

child.on("error", err => {
  console.error(`Failed to spawn claude: ${err.message}`);
  process.exit(1);
});

// Stream-json formatter: print tool calls and text as they arrive
let buf = "";
child.stdout.on("data", chunk => {
  buf += chunk.toString();
  const lines = buf.split("\n");
  buf = lines.pop(); // keep incomplete line
  for (const line of lines) {
    if (!line.trim()) continue;
    let ev; try { ev = JSON.parse(line); } catch { process.stdout.write(line + "\n"); continue; }
    const t = ev.type;
    if (t === "assistant" && ev.message?.content) {
      for (const b of ev.message.content) {
        if (b.type === "text" && b.text?.trim()) {
          process.stdout.write("\n" + b.text.trim() + "\n");
        } else if (b.type === "tool_use") {
          const v = b.input?.command || b.input?.file_path || b.input?.pattern || "";
          process.stdout.write(`  → ${b.name}: ${String(v).replace(/\n/g, " ").slice(0, 80)}\n`);
        }
      }
    } else if (t === "result") {
      process.stdout.write(`\n[done — ${ev.subtype ?? ""}]\n`);
    }
  }
});

child.on("exit", code => process.exit(code ?? 0));
