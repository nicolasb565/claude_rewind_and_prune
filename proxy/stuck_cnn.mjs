/**
 * CNN-based stuck detection for Claude Code.
 *
 * Replaces the LogReg text classifier with a CNN that operates on
 * tool-call behavioral features (CRC32 hashed commands, Jaccard output
 * similarity, cycle detection). Language-agnostic — works across
 * programming languages and task types.
 *
 * Detection rule: direct threshold on CNN output (no confirmation window).
 * Streak/confirmation rules were evaluated but did not improve over direct
 * thresholding — they only delay the first detection without suppressing FPs.
 *
 * Nudge escalation: two silent detections (-2→-1→0) before the first nudge
 * fires, then soft → medium → hard. Exponential backoff per level: 1/2/4/8
 * turns so silent levels clear fast and nudge levels give the agent increasing
 * time to respond. Resets to -2 when score drops below NUDGE_RESET_THRESHOLD.
 */

import { classifyWindow, normalizeFeatures, config } from "./classify_cnn.mjs";
import { StuckDetectorState } from "./abstract_step.mjs";

const WINDOW_SIZE = config.window_size;
// Hysteresis threshold for nudgeLevel reset: score must drop below this to
// consider the agent "unstuck" and reset escalation back to level 0.
// Kept below the fire threshold so a brief dip doesn't reset prematurely.
const NUDGE_RESET_THRESHOLD = parseFloat(process.env.STUCK_RESET_THRESHOLD || String(config.threshold * 0.94));

// Per-session state
const sessions = new Map();

function getSession(messages) {
  let key = "";
  for (const msg of messages) {
    if (msg.role === "user") {
      const text = Array.isArray(msg.content)
        ? msg.content.map(b => b.text || "").join("")
        : String(msg.content);
      key = text.slice(0, 200);
      break;
    }
  }
  if (!key) key = "__default__";

  if (!sessions.has(key)) {
    sessions.set(key, {
      detector: new StuckDetectorState(),
  
      turnCounter: 0,
      lastNudgeTurn: -999,
      nudgeLevel: -2,      // -2→-1 silent, 0→1→2 fire nudge; absorbs short FP bursts
      initialized: false,
    });
  }
  return sessions.get(key);
}

export function resetState() {
  sessions.clear();
}

/**
 * Extract tool calls from messages that haven't been processed yet.
 * Returns array of { toolName, input, output, thinking } objects.
 */
function extractNewToolCalls(messages, session) {
  const toolCalls = [];

  // Find all tool_use blocks in assistant messages
  const pendingResults = new Map();

  for (const msg of messages) {
    if (msg.role === "assistant" && Array.isArray(msg.content)) {
      let thinking = "";
      for (const block of msg.content) {
        if (block.type === "thinking") {
          thinking = block.thinking || "";
        } else if (block.type === "tool_use") {
          pendingResults.set(block.id, {
            toolName: block.name,
            input: block.input || {},
            output: "",
            thinking,
          });
          thinking = ""; // only first tool call gets thinking
        }
      }
    } else if (msg.role === "user" && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === "tool_result" && pendingResults.has(block.tool_use_id)) {
          const tc = pendingResults.get(block.tool_use_id);
          const content = block.content;
          if (Array.isArray(content)) {
            tc.output = content
              .filter(b => b.type === "text")
              .map(b => b.text || "")
              .join(" ");
          } else if (typeof content === "string") {
            tc.output = content;
          }
        }
      }
    }
  }

  // Return all tool calls in order
  for (const msg of messages) {
    if (msg.role === "assistant" && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === "tool_use" && pendingResults.has(block.id)) {
          toolCalls.push(pendingResults.get(block.id));
        }
      }
    }
  }

  return toolCalls;
}

export function pruneIfStuck(messages, log) {
  const session = getSession(messages);
  session.turnCounter++;

  // Exponential backoff per level: silent levels clear fast, nudge levels give
  // the agent increasing time to respond. Level -2: 1 turn, -1: 2, 0: 4, 1+: 8.
  const LEVEL_COOLDOWNS = { "-2": 1, "-1": 2, "0": 4, "1": 8, "2": 8 };
  const cooldown = LEVEL_COOLDOWNS[String(session.nudgeLevel)] ?? 4;
  if (session.turnCounter - session.lastNudgeTurn < cooldown) return messages;

  // On first call, process all existing tool calls to build history
  if (!session.initialized) {
    const allToolCalls = extractNewToolCalls(messages, session);
    for (const tc of allToolCalls) {
      session.detector.addStep(tc.toolName, tc.input, tc.output, tc.thinking);
    }
    session.initialized = true;
  } else {
    // Process only the last assistant message's tool calls
    const lastAssistant = [...messages].reverse().find(m => m.role === "assistant");
    if (lastAssistant && Array.isArray(lastAssistant.content)) {
      let thinking = "";
      for (const block of lastAssistant.content) {
        if (block.type === "thinking") {
          thinking = block.thinking || "";
        } else if (block.type === "tool_use") {
          // Find the corresponding tool result
          let output = "";
          for (const msg of messages) {
            if (msg.role === "user" && Array.isArray(msg.content)) {
              for (const b of msg.content) {
                if (b.type === "tool_result" && b.tool_use_id === block.id) {
                  output = Array.isArray(b.content)
                    ? b.content.filter(x => x.type === "text").map(x => x.text).join(" ")
                    : String(b.content || "");
                }
              }
            }
          }
          session.detector.addStep(block.name, block.input || {}, output, thinking);
          thinking = "";
        }
      }
    }
  }

  // Get current window
  const window = session.detector.getWindow(WINDOW_SIZE);
  if (!window) return messages;

  // Normalize and classify
  const normalizedCont = window.continuous.map(row => normalizeFeatures(row));
  const { score, stuck } = classifyWindow(
    window.toolIndices, normalizedCont, window.windowFeatures
  );

  // Direct threshold: fire whenever CNN score exceeds threshold.
  // Streak/confirmation rules were tested and did not improve over direct
  // thresholding — they only delay the first detection without suppressing FPs.
  const shouldFire = score >= config.threshold;

  if (!shouldFire) {
    if (score < NUDGE_RESET_THRESHOLD) session.nudgeLevel = -2; // agent responded, full reset
    return messages;
  }

  log?.("cnn_stuck_detected", {
    turnCount: session.turnCounter,
    score,
    nudgeLevel: session.nudgeLevel,
    threshold: config.threshold,
    nudgeResetThreshold: NUDGE_RESET_THRESHOLD,
    stepCount: session.detector.stepCount,
  });

  session.lastNudgeTurn = session.turnCounter;

  // Silent levels: absorb detection without injecting nudge
  if (session.nudgeLevel < 0) {
    session.nudgeLevel++;
    return messages;
  }

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
  const recentList = recentTools.slice(-8).join("\n  ");
  const pct = (score * 100).toFixed(0);
  const level = session.nudgeLevel;

  const nudgeText = level === 0
    ? // Soft — ask the agent to reflect
      `[CONTEXT MONITOR — turn ${session.turnCounter}, confidence ${pct}%]\n\n` +
      `Your recent actions show signs of repetitive patterns. ` +
      `You may be going in circles.\n\n` +
      `Recent tool calls:\n  ${recentList}\n\n` +
      `Review your last few turns critically:\n` +
      `- Are you retrying the same approach with minor variations?\n` +
      `- Are you investigating the same files/functions repeatedly?\n` +
      `- Has your hypothesis changed or are you stuck on the same one?\n\n` +
      `If you are going in circles, try a fundamentally different strategy.\n` +
      `State what you have learned so far and what new approach you will try.`

    : level === 1
    ? // Medium — more direct, demand a strategy change
      `[CONTEXT MONITOR — turn ${session.turnCounter}, confidence ${pct}% — repeated signal]\n\n` +
      `You have been nudged before and the repetitive pattern continues.\n\n` +
      `Recent tool calls:\n  ${recentList}\n\n` +
      `You appear to be stuck in a loop. The approach you are using is not working.\n` +
      `Before your next tool call:\n` +
      `1. State in one sentence what you have been trying to do.\n` +
      `2. State specifically why it has not worked.\n` +
      `3. Propose a different approach you have not tried yet.\n\n` +
      `Do not retry the same command. Switch strategy.`

    : // Hard — stop everything, force explicit plan
      `[CONTEXT MONITOR — turn ${session.turnCounter}, confidence ${pct}% — escalated]\n\n` +
      `STOP. You are deeply stuck and have not responded to prior nudges.\n\n` +
      `Recent tool calls:\n  ${recentList}\n\n` +
      `Do not run any more tool calls until you have answered these:\n` +
      `1. What is the root cause of the problem you are trying to solve?\n` +
      `2. What have you tried, and why did each attempt fail?\n` +
      `3. What fundamentally different approach will you take next?\n\n` +
      `If you cannot answer these, state that clearly and ask for guidance.`;

  session.nudgeLevel = Math.min(session.nudgeLevel + 1, 2);

  const nudge = {
    role: "user",
    content: [{ type: "text", text: nudgeText }],
  };

  log?.("cnn_nudge_injected", {
    turnCount: session.turnCounter,
    score,
    nudgeLevel: level,
    method: "cnn",
    recentTools: recentTools.slice(-5),
  });

  return [...messages, nudge];
}
