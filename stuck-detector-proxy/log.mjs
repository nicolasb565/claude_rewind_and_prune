/**
 * Telemetry logging — JSONL to ~/.stuck-detector/logs/
 */

import { appendFileSync, mkdirSync, existsSync } from "fs";
import { join } from "path";
import { homedir } from "os";

const logDir =
  process.env.LOG_DIR || join(homedir(), ".stuck-detector", "logs");

if (!existsSync(logDir)) {
  mkdirSync(logDir, { recursive: true });
}

const sessionId =
  Math.random().toString(36).slice(2, 10) +
  "-" +
  Date.now().toString(36);

const logFile = join(
  logDir,
  `events-${new Date().toISOString().slice(0, 10)}.jsonl`,
);

export function log(type, data) {
  const entry = {
    sessionId,
    timestamp: Date.now(),
    type,
    ...data,
  };
  try {
    appendFileSync(logFile, JSON.stringify(entry) + "\n");
  } catch {
    // best-effort
  }
}

export function logRequest(method, url, messageCount) {
  if (process.env.LOG_REQUESTS === "1") {
    log("request", { method, url, messageCount });
  }
}
