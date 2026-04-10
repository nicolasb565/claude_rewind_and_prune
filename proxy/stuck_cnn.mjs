/**
 * CNN-based stuck detection for Claude Code.
 *
 * Replaces the LogReg text classifier with a CNN that operates on
 * tool-call behavioral features (CRC32 hashed commands, Jaccard output
 * similarity, cycle detection). Language-agnostic — works across
 * programming languages and task types.
 *
 * Confirmation rule: streak-based. Fires if the current window scores
 * above threshold AND at least one of the immediately preceding windows
 * also scored above the streak threshold (sustained signal).
 *
 * This rule was chosen by analyzing 30+ temporal aggregations on the
 * test set (see src/analyze_temporal.py). streak_thresh_0.9 achieved
 * F1=0.886, recall=0.933 — a +11% recall gain vs raw single-window
 * detection at the cost of ~6% precision.
 */

import { classifyWindow, normalizeFeatures, config } from "./classify_cnn.mjs";
import { StuckDetectorState } from "./abstract_step.mjs";

const WINDOW_SIZE = config.window_size;
const STREAK_THRESHOLD = 0.9; // score considered "high enough" for streak counting
const STREAK_REQUIRED = 1;     // require at least 1 prior window above streak threshold

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
      windowScores: [],
      turnCounter: 0,
      lastNudgeTurn: -999,
      nudgeLevel: 0,       // escalates 0→1→2 while stuck persists across cooldowns
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

  const cooldown = parseInt(process.env.STUCK_COOLDOWN || "5", 10);
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

  // Streak-based confirmation: fire if current score is high AND at least one of
  // the prior windows scored above the streak threshold (sustained signal).
  // This replaces the old "2 of last 3 above threshold" rule.
  session.windowScores.push(score);
  // Keep enough history to count streaks; cap at 10 for memory
  if (session.windowScores.length > 10) {
    session.windowScores.shift();
  }

  // Count consecutive windows scoring >= STREAK_THRESHOLD ending at the
  // window BEFORE the current one (i.e. the "prior streak").
  let priorStreak = 0;
  for (let i = session.windowScores.length - 2; i >= 0; i--) {
    if (session.windowScores[i] >= STREAK_THRESHOLD) {
      priorStreak++;
    } else {
      break;
    }
  }

  // Fire if current is above threshold AND we have a sustained prior streak.
  // If score dropped below streak threshold, reset escalation level.
  const currentHigh = score >= config.threshold;
  const shouldFire = currentHigh && priorStreak >= STREAK_REQUIRED;

  if (!shouldFire) {
    if (score < STREAK_THRESHOLD) session.nudgeLevel = 0; // agent responded, reset
    return messages;
  }

  log?.("cnn_stuck_detected", {
    turnCount: session.turnCounter,
    score,
    nudgeLevel: session.nudgeLevel,
    threshold: config.threshold,
    streakThreshold: STREAK_THRESHOLD,
    priorStreak,
    windowScores: [...session.windowScores],
    stepCount: session.detector.stepCount,
  });

  session.lastNudgeTurn = session.turnCounter;

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
