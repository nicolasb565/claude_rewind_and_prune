#!/usr/bin/env node
/**
 * Bootstrap for Claude Code Rewind fork.
 * Sets up global config based on CLAUDE_REWIND_MODE env var,
 * initializes telemetry, then loads the patched CLI.
 */

import { appendFileSync, mkdirSync, existsSync } from "fs";
import { join } from "path";
import { homedir } from "os";

// ── Configuration ────────────────────────────────────────────
// CLAUDE_REWIND_MODE:
//   "off"          — stock behavior (no patches active)
//   "compact_only" — auto-compact truncation only, no Rewind tool
//   "full"         — auto-compact + Rewind tool (default)

const mode = process.env.CLAUDE_REWIND_MODE || "full";
globalThis.__REWIND_MODE__ = mode;

// Auto-compact tuning (overridable via env)
globalThis.__REWIND_COMPACT_CFG__ = {
  staleAfterTurns: parseInt(process.env.REWIND_STALE_TURNS || "3", 10),
  keepFirstLines: parseInt(process.env.REWIND_KEEP_FIRST || "30", 10),
  keepLastLines: parseInt(process.env.REWIND_KEEP_LAST || "10", 10),
  minLinesForCompaction: parseInt(process.env.REWIND_MIN_LINES || "50", 10),
  // Tools whose output is never truncated — the model refers back to these
  // while making iterative edits; truncating causes expensive re-reads.
  neverTruncateTools: (process.env.REWIND_NEVER_TRUNCATE || "Read,Edit,Write,Grep,Glob,NotebookEdit").split(","),
};

// ── Telemetry ────────────────────────────────────────────────

const logDir = join(homedir(), ".claude-rewind-logs");
if (!existsSync(logDir)) {
  mkdirSync(logDir, { recursive: true });
}

const sessionId = Math.random().toString(36).slice(2, 10) + "-" + Date.now().toString(36);
const sessionStart = Date.now();
const logFile = join(logDir, `events-${new Date().toISOString().slice(0, 10)}.jsonl`);

function logEvent(type, data) {
  const entry = {
    sessionId,
    timestamp: Date.now(),
    type,
    mode,
    ...data,
  };
  try {
    appendFileSync(logFile, JSON.stringify(entry) + "\n");
  } catch {
    // best-effort
  }
}

globalThis.__REWIND_LOG__ = logEvent;

// Log session start
logEvent("session_start", {
  config: globalThis.__REWIND_COMPACT_CFG__,
  version: "0.1.0",
});

// Log session end on exit
process.on("exit", () => {
  logEvent("session_end", {
    durationSeconds: Math.round((Date.now() - sessionStart) / 1000),
  });
});

// ── Status banner ────────────────────────────────────────────

if (process.stderr.isTTY) {
  const modeLabel = mode === "full" ? "compact+rewind" : mode === "compact_only" ? "compact" : "off";
  process.stderr.write(`\x1b[2m[rewind-fork: mode=${modeLabel}]\x1b[0m\n`);
}

// ── Load patched CLI ─────────────────────────────────────────

import("../bin/claude-rewind.js").catch((err) => {
  console.error("Failed to load patched CLI:", err.message);
  console.error("Run 'node src/patch.mjs' first to generate bin/claude-rewind.js");
  process.exit(1);
});
