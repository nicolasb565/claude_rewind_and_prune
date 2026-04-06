/**
 * Stuck detection: analyze thinking blocks for circular reasoning.
 * When detected, inject a corrective nudge into the messages.
 */

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

  // Run heuristic
  if (!isThinkingSuspicious(thinking)) return messages;

  log?.("stuck_heuristic_triggered", {
    turnCount: turnCounter,
    thinkingLength: thinking.length,
  });

  lastNudgeTurn = turnCounter;

  // Build summary of recent tool calls for context
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

  // Inject nudge as a user message
  const nudge = {
    role: "user",
    content: [
      {
        type: "text",
        text:
          `[CONTEXT MONITOR — turn ${turnCounter}]\n\n` +
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
    recentTools: recentTools.slice(-5),
  });

  return [...messages, nudge];
}

function isThinkingSuspicious(text) {
  // 1. Repeated 20-char substrings appearing 3+ times
  const seen = {};
  for (let i = 0; i < text.length - 20; i += 10) {
    const sub = text.substring(i, i + 20);
    seen[sub] = (seen[sub] || 0) + 1;
    if (seen[sub] >= 3) return true;
  }

  // 2. Circle keywords frequency
  const matches = text.match(
    /\b(try again|let me try|another approach|actually,|wait,|hmm|let me reconsider|that didn't work|same error|still failing)\b/gi,
  );
  if (matches && matches.length >= 5) return true;

  // 3. High word overlap between first half and second half
  if (text.length > 2000) {
    const half = Math.floor(text.length / 2);
    const words1 = new Set(
      text
        .slice(0, half)
        .toLowerCase()
        .split(/\s+/)
        .filter((w) => w.length > 4),
    );
    const words2 = text
      .slice(half)
      .toLowerCase()
      .split(/\s+/)
      .filter((w) => w.length > 4);
    let overlap = 0;
    for (const w of words2) if (words1.has(w)) overlap++;
    if (words2.length > 0 && overlap / words2.length > 0.6) return true;
  }

  return false;
}
