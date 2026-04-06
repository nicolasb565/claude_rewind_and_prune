/**
 * Stuck detection: analyze thinking blocks for circular reasoning.
 * Uses a trained logistic regression classifier (pure JS, no Python dependency).
 * When detected, inject a corrective nudge into the messages.
 */

import { classify } from "./classify.mjs";

let lastNudgeTurn = -999;
let turnCounter = 0;

export function resetState() {
  lastNudgeTurn = -999;
  turnCounter = 0;
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

  // Extract thinking from previous assistant messages for cross-window comparison
  const prevThinkings = [];
  for (let i = lastAssistantIdx - 1; i >= 0; i--) {
    if (messages[i].role !== "assistant" || !Array.isArray(messages[i].content)) continue;
    let prevThinking = "";
    for (const block of messages[i].content) {
      if (block.type === "thinking" && block.thinking) prevThinking += block.thinking;
    }
    if (prevThinking.length > 200) {
      prevThinkings.push(prevThinking);
      if (prevThinkings.length >= 3) break; // last 3 thinking blocks
    }
  }

  // Cross-window similarity: compare current thinking to recent previous ones
  let crossWindowScore = 0;
  if (prevThinkings.length > 0) {
    const currWords = new Set(thinking.toLowerCase().split(/\s+/).filter(w => w.length > 4));
    for (const prev of prevThinkings) {
      const prevWords = prev.toLowerCase().split(/\s+/).filter(w => w.length > 4);
      if (prevWords.length === 0) continue;
      let overlap = 0;
      for (const w of prevWords) if (currWords.has(w)) overlap++;
      const sim = overlap / prevWords.length;
      crossWindowScore = Math.max(crossWindowScore, sim);
    }
  }

  // Classify thinking text
  const threshold = parseFloat(process.env.STUCK_THRESHOLD || "0.85");
  const result = classify(thinking);

  // Combine classifier score with cross-window similarity.
  // High classifier score + low cross-window = wrong hypothesis but new ground (don't nudge)
  // High classifier score + high cross-window = genuinely stuck (nudge)
  const crossWindowThreshold = parseFloat(process.env.STUCK_CROSS_WINDOW_THRESHOLD || "0.5");
  const combinedStuck = result.score >= threshold &&
    (prevThinkings.length === 0 || crossWindowScore >= crossWindowThreshold);

  if (!combinedStuck) {
    if (result.score >= threshold) {
      log?.("stuck_suppressed", {
        turnCount: turnCounter,
        classifierScore: result.score,
        crossWindowScore: Math.round(crossWindowScore * 100) / 100,
        reason: "low cross-window similarity — likely wrong hypothesis but progressing",
      });
    }
    return messages;
  }

  log?.("stuck_detected", {
    turnCount: turnCounter,
    thinkingLength: thinking.length,
    score: result.score,
    crossWindowScore: Math.round(crossWindowScore * 100) / 100,
    label: result.label,
    reasons: result.reasons,
    method: "classifier",
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
