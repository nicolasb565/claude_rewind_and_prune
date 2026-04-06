/**
 * Stuck detection: analyze thinking blocks for circular reasoning.
 * Uses a trained logistic regression classifier (pure JS, no Python dependency).
 * Combines text features with tool-call behavioral features.
 * When detected, inject a corrective nudge into the messages.
 */

import { classify } from "./classify.mjs";

let lastNudgeTurn = -999;
let turnCounter = 0;

export function resetState() {
  lastNudgeTurn = -999;
  turnCounter = 0;
}

/**
 * Extract behavioral features from recent tool calls in the message history.
 * These capture patterns invisible to text-only analysis.
 */
function extractToolFeatures(messages, windowSize = 20) {
  const recent = messages.slice(-windowSize);

  // Collect all tool_use blocks
  const toolCalls = [];
  for (const msg of recent) {
    if (!Array.isArray(msg.content)) continue;
    for (const block of msg.content) {
      if (block.type === "tool_use") {
        toolCalls.push({
          name: block.name,
          file: block.input?.file_path || block.input?.path || "",
          command: block.input?.command || "",
          pattern: block.input?.pattern || "",
        });
      }
    }
  }

  if (toolCalls.length === 0) {
    return {
      file_read_repeat: 0,
      grep_pattern_repeat: 0,
      bash_cmd_repeat: 0,
      unique_files_ratio: 1,
      tool_diversity: 1,
      same_tool_streak: 0,
    };
  }

  // file_read_repeat: max times any file was read
  const fileCounts = {};
  for (const tc of toolCalls) {
    if (tc.file && (tc.name === "Read" || tc.name === "Grep" || tc.name === "Glob")) {
      fileCounts[tc.file] = (fileCounts[tc.file] || 0) + 1;
    }
  }
  const fileReadRepeat = Math.max(0, ...Object.values(fileCounts));

  // grep_pattern_repeat: how similar are recent grep patterns?
  const grepPatterns = toolCalls
    .filter(tc => tc.name === "Grep" && tc.pattern)
    .map(tc => tc.pattern.toLowerCase());
  let grepRepeat = 0;
  if (grepPatterns.length >= 2) {
    // Count pairs with >50% character overlap
    for (let i = 0; i < grepPatterns.length; i++) {
      for (let j = i + 1; j < grepPatterns.length; j++) {
        const a = new Set(grepPatterns[i]);
        const b = grepPatterns[j];
        let overlap = 0;
        for (const c of b) if (a.has(c)) overlap++;
        if (overlap / Math.max(a.size, b.length) > 0.5) grepRepeat++;
      }
    }
  }

  // bash_cmd_repeat: how many similar bash commands?
  const bashCmds = toolCalls
    .filter(tc => tc.name === "Bash" && tc.command)
    .map(tc => tc.command.trim().slice(0, 80)); // normalize
  const bashCounts = {};
  for (const cmd of bashCmds) {
    bashCounts[cmd] = (bashCounts[cmd] || 0) + 1;
  }
  const bashCmdRepeat = Math.max(0, ...Object.values(bashCounts));

  // unique_files_ratio: unique files / total file references
  const allFiles = toolCalls.filter(tc => tc.file).map(tc => tc.file);
  const uniqueFilesRatio = allFiles.length > 0
    ? new Set(allFiles).size / allFiles.length
    : 1;

  // tool_diversity: unique tool names / total calls
  const toolNames = toolCalls.map(tc => tc.name);
  const toolDiversity = new Set(toolNames).size / toolNames.length;

  // same_tool_streak: longest consecutive run of same tool
  let maxStreak = 1;
  let streak = 1;
  for (let i = 1; i < toolNames.length; i++) {
    if (toolNames[i] === toolNames[i - 1]) {
      streak++;
      maxStreak = Math.max(maxStreak, streak);
    } else {
      streak = 1;
    }
  }

  return {
    file_read_repeat: fileReadRepeat,
    grep_pattern_repeat: grepRepeat,
    bash_cmd_repeat: bashCmdRepeat,
    unique_files_ratio: uniqueFilesRatio,
    tool_diversity: toolDiversity,
    same_tool_streak: maxStreak,
  };
}

export function pruneIfStuck(messages, log) {
  turnCounter++;
  const cooldown = parseInt(process.env.STUCK_COOLDOWN || "5", 10);
  if (turnCounter - lastNudgeTurn < cooldown) return messages;

  // Extract thinking from the last assistant message
  let lastAssistantIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") {
      lastAssistantIdx = i;
      break;
    }
  }
  if (lastAssistantIdx === -1) return messages;

  const lastAssistant = messages[lastAssistantIdx];
  if (!Array.isArray(lastAssistant.content)) return messages;

  let thinking = "";
  for (const block of lastAssistant.content) {
    if (block.type === "thinking" && block.thinking) {
      thinking += block.thinking;
    }
  }

  if (thinking.length < 500) return messages;

  // Extract tool-call behavioral features
  const toolFeats = extractToolFeatures(messages);

  // Classify thinking text + tool behavior
  const threshold = parseFloat(process.env.STUCK_THRESHOLD || "0.80");
  const result = classify(thinking, toolFeats);

  if (result.score < threshold) return messages;

  log?.("stuck_detected", {
    turnCount: turnCounter,
    thinkingLength: thinking.length,
    score: result.score,
    label: result.label,
    reasons: result.reasons,
    toolFeats,
  });

  lastNudgeTurn = turnCounter;

  // Build recent tool call summary
  const recentTools = [];
  for (const msg of messages.slice(-20)) {
    if (!Array.isArray(msg.content)) continue;
    for (const block of msg.content) {
      if (block.type === "tool_use") {
        const detail =
          block.input?.command || block.input?.file_path || block.input?.pattern || "";
        recentTools.push(`${block.name}: ${String(detail).slice(-60)}`);
      }
    }
  }

  const nudge = {
    role: "user",
    content: [
      {
        type: "text",
        text:
          `[CONTEXT MONITOR — turn ${turnCounter}, confidence ${(result.score * 100).toFixed(0)}%]\n\n` +
          `Your recent thinking shows signs of repeated reasoning patterns. ` +
          `You may be going in circles.\n\n` +
          `Recent tool calls:\n  ${recentTools.slice(-8).join("\n  ")}\n\n` +
          `Review your last few turns critically:\n` +
          `- Are you retrying the same approach with minor variations?\n` +
          `- Are you investigating the same files/functions repeatedly?\n` +
          `- Has your hypothesis changed or are you stuck on the same one?\n\n` +
          `If you are going in circles, try a fundamentally different strategy.\n` +
          `State what you have learned so far and what new approach you will try.`,
      },
    ],
  };

  log?.("stuck_nudge_injected", {
    turnCount: turnCounter,
    score: result.score,
    method: "classifier",
    recentTools: recentTools.slice(-5),
  });

  return [...messages, nudge];
}
