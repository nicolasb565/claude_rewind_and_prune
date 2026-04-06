#!/usr/bin/env node
/**
 * Patch script for Claude Code fork.
 * Applies context-rewind and model-driven compaction patches to the original cli.js.
 *
 * Usage: node src/patch.mjs [--check]
 *   --check: verify patches can be applied without writing
 */

import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");

const srcPath = resolve(ROOT, "vendor/cli-original.js");
const outPath = resolve(ROOT, "bin/claude-rewind.js");

const checkOnly = process.argv.includes("--check");

let src = readFileSync(srcPath, "utf-8");
let patchCount = 0;

function patch(name, searchStr, replaceStr) {
  const idx = src.indexOf(searchStr);
  if (idx === -1) {
    console.error(`PATCH FAILED: "${name}" — anchor not found`);
    console.error(`  Looking for: ${searchStr.slice(0, 120)}...`);
    process.exit(1);
  }
  const secondIdx = src.indexOf(searchStr, idx + 1);
  if (secondIdx !== -1) {
    console.error(`PATCH FAILED: "${name}" — anchor is not unique (found at ${idx} and ${secondIdx})`);
    process.exit(1);
  }
  if (!checkOnly) {
    src = src.slice(0, idx) + replaceStr + src.slice(idx + searchStr.length);
  }
  patchCount++;
  console.log(`  ✓ ${name}`);
}

console.log("Applying patches to cli.js...\n");

// ─────────────────────────────────────────────────────────────
// PATCH 1: Add `ephemeral` parameter to Bash tool input schema
// The model sets ephemeral:true when it knows the output is
// consume-once (test runs, build logs, exploratory commands).
// ─────────────────────────────────────────────────────────────

const BASH_SCHEMA_ANCHOR = `_simulatedSedEdit:h.object({filePath:h.string(),newContent:h.string()}).optional().describe("Internal: pre-computed sed edit result from preview")`;
const BASH_SCHEMA_REPLACEMENT = `_simulatedSedEdit:h.object({filePath:h.string(),newContent:h.string()}).optional().describe("Internal: pre-computed sed edit result from preview"),ephemeral:jP(h.boolean().optional()).describe("Defaults to true. Bash output is automatically compacted after you have processed it to free context window space. Set ephemeral: false ONLY when you need to refer back to this command's exact output in later turns (e.g. capturing a value you will use later, output you will compare against, or reference data). Most commands — test runs, builds, installs, git operations, linting — are ephemeral: you read the output once and act on it.")`;

patch("Add ephemeral to Bash schema", BASH_SCHEMA_ANCHOR, BASH_SCHEMA_REPLACEMENT);


// ─────────────────────────────────────────────────────────────
// PATCH 2: Track ephemeral tool_use IDs from assistant messages
// When the model emits a tool_use with ephemeral:true in the
// input, record its ID so the truncation pass can find it.
// Anchor: right after assistant messages are collected (X6.push).
// ─────────────────────────────────────────────────────────────

const TRACK_ANCHOR = `if(x6.type==="assistant"){X6.push(x6);let k6=x6.message.content.filter((M6)=>M6.type==="tool_use")`;
const TRACK_REPLACEMENT = `if(x6.type==="assistant"){X6.push(x6);let k6=x6.message.content.filter((M6)=>M6.type==="tool_use");
/* --- REWIND PATCH: track preserved (non-ephemeral) Bash tool calls --- */
if(typeof globalThis.__REWIND_MODE__!=="undefined"&&globalThis.__REWIND_MODE__!=="off"){
  if(!globalThis.__REWIND_PRESERVED_IDS__)globalThis.__REWIND_PRESERVED_IDS__=new Set();
  for(var _ei=0;_ei<k6.length;_ei++){
    if(k6[_ei].name==="Bash"&&k6[_ei].input&&k6[_ei].input.ephemeral===false){
      globalThis.__REWIND_PRESERVED_IDS__.add(k6[_ei].id);
    }
  }
}
/* --- END track preserved --- */
void 0`;

patch("Track preserved tool IDs", TRACK_ANCHOR, TRACK_REPLACEMENT);


// ─────────────────────────────────────────────────────────────
// PATCH 3: Model-driven compaction of ephemeral tool outputs
// After microcompact, scan for tool results whose tool_use_id
// was marked ephemeral AND that the model has already seen
// (at least 1 assistant turn old). Truncate those.
// ─────────────────────────────────────────────────────────────

const COMPACT_ANCHOR = 'c3("query_microcompact_end");';
const COMPACT_REPLACEMENT = `c3("query_microcompact_end");

/* --- REWIND PATCH: compact ephemeral Bash outputs --- */
if (typeof globalThis.__REWIND_MODE__ !== "undefined" && globalThis.__REWIND_MODE__ !== "off") {
  (function compactEphemerals(msgs) {
    var preservedIds = globalThis.__REWIND_PRESERVED_IDS__ || new Set();

    var cfg = globalThis.__REWIND_COMPACT_CFG__ || {
      keepFirstLines: 30,
      keepLastLines: 10,
      minLinesForCompaction: 50,
    };

    // Build tool_use_id -> tool_name map from assistant messages
    var toolNameById = new Map();
    for (var si = 0; si < msgs.length; si++) {
      if (msgs[si].type !== "assistant" || !Array.isArray(msgs[si].message?.content)) continue;
      for (var sb = 0; sb < msgs[si].message.content.length; sb++) {
        var sblk = msgs[si].message.content[sb];
        if (sblk.type === "tool_use" && sblk.id && sblk.name) {
          toolNameById.set(sblk.id, sblk.name);
        }
      }
    }

    // Only compact results the model has already seen (1+ assistant turns after)
    var assistantCount = 0;
    var turnAtIndex = new Map();
    for (var ti = 0; ti < msgs.length; ti++) {
      if (msgs[ti].type === "assistant") assistantCount++;
      turnAtIndex.set(ti, assistantCount);
    }
    var currentTurn = assistantCount;

    for (var mi = 0; mi < msgs.length; mi++) {
      var msg = msgs[mi];
      if (msg.type !== "user" || !Array.isArray(msg.message?.content)) continue;

      var hasToolResult = msg.message.content.some(function(c) { return c.type === "tool_result"; });
      if (!hasToolResult) continue;

      var turnCreated = turnAtIndex.get(mi) || 0;
      if (currentTurn - turnCreated < 1) continue;

      var modified = false;
      var newContent = msg.message.content.map(function(block) {
        if (block.type !== "tool_result") return block;

        // Only compact Bash outputs (ephemeral by default)
        var toolName = toolNameById.get(block.tool_use_id) || "";
        if (toolName !== "Bash") return block;

        // Model explicitly preserved this one with ephemeral:false
        if (preservedIds.has(block.tool_use_id)) return block;

        var text = typeof block.content === "string" ? block.content
          : Array.isArray(block.content) ? block.content.filter(function(x){return x.type==="text"}).map(function(x){return x.text}).join("\\n")
          : "";
        if (!text || text.startsWith("[COMPACTED")) return block;
        var lines = text.split("\\n");
        if (lines.length < cfg.minLinesForCompaction) return block;

        var truncated = lines.slice(0, cfg.keepFirstLines).concat(
          ["\\n[... " + (lines.length - cfg.keepFirstLines - cfg.keepLastLines) + " lines compacted ...]\\n"],
          lines.slice(-cfg.keepLastLines)
        ).join("\\n");
        var compacted = "[COMPACTED — ephemeral Bash output, original " + lines.length + " lines]\\n" + truncated;
        var savedChars = text.length - compacted.length;
        if (savedChars <= 0) return block;
        modified = true;

        if (globalThis.__REWIND_LOG__) {
          globalThis.__REWIND_LOG__("compact", {
            toolName: "Bash",
            toolUseId: block.tool_use_id,
            originalLines: lines.length,
            compactedLines: cfg.keepFirstLines + cfg.keepLastLines + 1,
            tokensSavedEstimate: Math.round(savedChars / 4),
            preserved: false,
          });
        }

        if (typeof block.content === "string") return { ...block, content: compacted };
        return { ...block, content: [{ type: "text", text: compacted }] };
      });
      if (modified) {
        msgs[mi] = { ...msg, message: { ...msg.message, content: newContent } };
      }
    }
  })(F);
}
/* --- END compact ephemerals --- */
`;

patch("Compact ephemeral tool outputs", COMPACT_ANCHOR, COMPACT_REPLACEMENT);


// ─────────────────────────────────────────────────────────────
// PATCH 4: Register the Rewind tool in the core tool list
// ─────────────────────────────────────────────────────────────

const TOOL_LIST_ANCHOR = "ozY??=[Iz,XP,MP,kx,Uh,c4,tp,ng8]";
const TOOL_LIST_REPLACEMENT = `ozY??=(function(){
  /* --- REWIND PATCH: register Rewind tool --- */
  var basTools = [Iz,XP,MP,kx,Uh,c4,tp,ng8];
  if (typeof globalThis.__REWIND_MODE__ === "undefined" || globalThis.__REWIND_MODE__ !== "full") return basTools;

  var rewindSessionCount = 0;
  var REWIND_MAX = 5;

  var RewindTool = q4({
    name: "Rewind",
    searchHint: "rewind conversation abandon failed approach",
    maxResultSizeChars: Infinity,
    strict: true,
    async description() {
      return "Abandon the current approach and rewind the conversation.";
    },
    async prompt() {
      return "Abandon the current approach and rewind the conversation. " +
        "Everything from turns_back ago to now is REPLACED with your summary. " +
        "\\n\\nUse when: (1) an approach has clearly failed after 2+ attempts, " +
        "(2) you are going in circles retrying similar things, " +
        "(3) you have learned enough to know WHY it failed." +
        "\\n\\nSummary must include: what was tried, why it failed, key facts or " +
        "constraints discovered. This is your only record of the pruned work — " +
        "include everything important." +
        "\\n\\nDoes NOT undo file changes. If you modified files during the pruned " +
        "segment, mention what was changed so you can fix it after rewind." +
        "\\n\\nParameters:\\n- turns_back (number, required): How many conversation turns " +
        "to rewind. Count your assistant messages back to where the failed approach started." +
        "\\n- summary (string, required): Summary of the pruned work.";
    },
    get inputSchema() {
      return {
        type: "object",
        properties: {
          turns_back: {
            type: "number",
            description: "How many conversation turns to rewind."
          },
          summary: {
            type: "string",
            description: "Summary of pruned work: what was tried, why it failed, constraints discovered, file changes made."
          }
        },
        required: ["turns_back", "summary"]
      };
    },
    userFacingName() { return "Rewind"; },
    getToolUseSummary(input) {
      return "Rewinding " + (input?.turns_back || "?") + " turns";
    },
    getActivityDescription(input) {
      return "Rewinding conversation";
    },
    isConcurrencySafe() { return false; },
    isReadOnly() { return false; },
    isEnabled() {
      return typeof globalThis.__REWIND_MODE__ !== "undefined" && globalThis.__REWIND_MODE__ === "full";
    },
    async checkPermissions(input, ctx) {
      return { behavior: "allow", updatedInput: input };
    },
    async run(input) {
      if (!input.summary || input.summary.length < 100) {
        return "Error: Summary must be at least 100 characters. Include what was tried, why it failed, and what you learned.";
      }
      if (!input.turns_back || input.turns_back < 1) {
        return "Error: turns_back must be at least 1.";
      }

      rewindSessionCount++;
      if (rewindSessionCount > REWIND_MAX) {
        return "Error: Rewind limit reached (" + REWIND_MAX + " rewinds this session). Reassess the problem fundamentally before continuing.";
      }

      globalThis.__REWIND_PENDING__ = {
        turnsBack: input.turns_back,
        summary: input.summary,
        rewindNumber: rewindSessionCount,
      };

      if (globalThis.__REWIND_LOG__) {
        globalThis.__REWIND_LOG__("rewind", {
          turnsBack: input.turns_back,
          summaryLength: input.summary.length,
          rewindNumber: rewindSessionCount,
        });
      }

      return "[REWIND ACCEPTED — " + input.turns_back + " turns will be pruned. Summary recorded. The orchestrator will now truncate the conversation.]";
    },
    renderToolUseMessage() { return undefined; },
    renderToolResultMessage() { return undefined; },
  });

  return [...basTools, RewindTool];
  /* --- END Rewind tool --- */
})()`;

patch("Register Rewind tool", TOOL_LIST_ANCHOR, TOOL_LIST_REPLACEMENT);


// ─────────────────────────────────────────────────────────────
// PATCH 5: Handle Rewind in the main loop
// ─────────────────────────────────────────────────────────────

const MAIN_LOOP_ANCHOR = 'c3("query_recursive_call"),J={messages:[...F,...X6,...P6]';
const MAIN_LOOP_REPLACEMENT = `c3("query_recursive_call");

/* --- REWIND PATCH: handle pending rewind --- */
(function handleRewind() {
  if (!globalThis.__REWIND_PENDING__) return;
  var rw = globalThis.__REWIND_PENDING__;
  globalThis.__REWIND_PENDING__ = null;

  var allMsgs = [...F,...X6,...P6];
  var assistantIndices = [];
  for (var ri = 0; ri < allMsgs.length; ri++) {
    if (allMsgs[ri].type === "assistant") assistantIndices.push(ri);
  }

  var maxRewind = assistantIndices.length - 1;
  var turnsBack = Math.min(rw.turnsBack, maxRewind);
  if (turnsBack < 1) return;

  var cutAtAssistant = assistantIndices[assistantIndices.length - turnsBack];
  var prunedCount = allMsgs.length - cutAtAssistant;

  var kept = allMsgs.slice(0, cutAtAssistant);
  kept.push(i8({
    content: "[REWIND — " + prunedCount + " messages pruned, previous approach abandoned]\\n\\n" + rw.summary,
    isMeta: true,
  }));

  F.length = 0;
  F.push(...kept);
  X6.length = 0;
  P6.length = 0;

  if (globalThis.__REWIND_LOG__) {
    globalThis.__REWIND_LOG__("rewind_applied", {
      turnsBack: turnsBack,
      messagesPruned: prunedCount,
      messagesRemaining: kept.length,
      rewindNumber: rw.rewindNumber,
    });
  }
})();
/* --- END handle rewind --- */

/* --- HAIKU MONITOR: check thinking for spiraling --- */
if (globalThis.__HAIKU_MONITOR__) {
  await (async function haikuMonitor() {
    try {
      // 1. Extract thinking text from this turn's assistant messages
      var thinkingText = "";
      for (var xi = 0; xi < X6.length; xi++) {
        var xContent = X6[xi].message?.content;
        if (!Array.isArray(xContent)) continue;
        for (var ci = 0; ci < xContent.length; ci++) {
          if (xContent[ci].type === "thinking" && xContent[ci].thinking) {
            thinkingText += xContent[ci].thinking;
          }
        }
      }

      if (thinkingText.length < 500) return;

      // 2. Heuristic: is the thinking suspicious?
      var suspicious = false;

      // 2a. Repeated 20-char substrings appearing 3+ times
      var seen = {};
      for (var si = 0; si < thinkingText.length - 20; si += 10) {
        var sub = thinkingText.substring(si, si + 20);
        seen[sub] = (seen[sub] || 0) + 1;
        if (seen[sub] >= 3) { suspicious = true; break; }
      }

      // 2b. "Going in circles" keyword frequency
      if (!suspicious) {
        var circleWords = thinkingText.match(
          /\\b(try again|let me try|another approach|actually,|wait,|hmm|let me reconsider|I was wrong|that didn't work|same error|still failing)\\b/gi
        );
        if (circleWords && circleWords.length >= 5) suspicious = true;
      }

      // 2c. High word overlap between first half and second half of thinking
      if (!suspicious && thinkingText.length > 2000) {
        var halfPt = Math.floor(thinkingText.length / 2);
        var words1 = new Set(thinkingText.slice(0, halfPt).toLowerCase().split(/\\s+/).filter(function(w){return w.length>4}));
        var words2 = thinkingText.slice(halfPt).toLowerCase().split(/\\s+/).filter(function(w){return w.length>4});
        var overlap = 0;
        for (var oi = 0; oi < words2.length; oi++) { if (words1.has(words2[oi])) overlap++; }
        if (words2.length > 0 && overlap / words2.length > 0.6) suspicious = true;
      }

      if (!suspicious) return;

      if (globalThis.__REWIND_LOG__) {
        globalThis.__REWIND_LOG__("haiku_heuristic_triggered", {
          thinkingLength: thinkingText.length,
          turnCount: m,
        });
      }

      // 3. Inject self-reflection nudge into the conversation.
      // Instead of calling a separate model, we tell the current model
      // it may be going in circles and ask it to use Rewind if it agrees.
      // Build a summary of what the last few turns did (tool names + files)
      var recentActions = [];
      var lookback = [...F.slice(-20), ...X6, ...P6];
      for (var lb = 0; lb < lookback.length; lb++) {
        var lbContent = lookback[lb].message?.content;
        if (!Array.isArray(lbContent)) continue;
        for (var lbc = 0; lbc < lbContent.length; lbc++) {
          var lbBlock = lbContent[lbc];
          if (lbBlock.type === "tool_use") {
            var detail = lbBlock.input?.command || lbBlock.input?.file_path || lbBlock.input?.pattern || "";
            if (typeof detail === "string") detail = detail.slice(-60);
            recentActions.push(lbBlock.name + ": " + detail);
          }
        }
      }
      var actionSummary = recentActions.slice(-10).join("\\n  ");

      var nudgeMsg = i8({
        content: "[CONTEXT MONITOR — turn " + m + "]\\n\\n" +
          "Your recent thinking shows signs of repeated reasoning patterns. " +
          "You may be going in circles.\\n\\n" +
          "Recent tool calls:\\n  " + actionSummary + "\\n\\n" +
          "Review your last few turns critically:\\n" +
          "- Are you retrying the same approach with minor variations?\\n" +
          "- Are you investigating the same files/functions repeatedly?\\n" +
          "- Has your hypothesis changed or are you stuck on the same one?\\n\\n" +
          "If you ARE going in circles, use the Rewind tool NOW to prune the " +
          "failed turns and record what you tried. Then take a fundamentally " +
          "different approach.\\n\\n" +
          "If you are making genuine progress, continue — but be honest with yourself.",
        isMeta: true,
      });

      // Append the nudge to P6 so it appears after the tool results
      // and before the next model turn
      P6.push(nudgeMsg);

      if (globalThis.__REWIND_LOG__) {
        globalThis.__REWIND_LOG__("monitor_nudge_injected", {
          turnCount: m,
          thinkingLength: thinkingText.length,
          recentActions: recentActions.slice(-5),
        });
      }
    } catch (haikuErr) {
      // Never let monitor failure break the main loop
      if (globalThis.__REWIND_LOG__) {
        globalThis.__REWIND_LOG__("haiku_error", {
          error: haikuErr?.message || String(haikuErr),
        });
      }
    }
  })();
}
/* --- END HAIKU MONITOR --- */

J={messages:[...F,...X6,...P6]`;

patch("Handle Rewind + Haiku monitor in main loop", MAIN_LOOP_ANCHOR, MAIN_LOOP_REPLACEMENT);


// ─────────────────────────────────────────────────────────────
// PATCH 6: Add system prompt for ephemeral + Rewind
// ─────────────────────────────────────────────────────────────

const SYSPROMPT_ANCHOR = 'var TL1="You are Claude Code, Anthropic\'s official CLI for Claude."';

const ephemeralPrompt = "\\n\\nYou have context management capabilities:" +
  "\\n\\n- Bash output is ephemeral by default: after you process a Bash result, it is automatically " +
  "compacted on the next turn to free context window space. This is the right behavior for most commands " +
  "(test runs, builds, installs, git operations, linting). If you need to preserve a Bash output to " +
  "reference it in later turns, set `ephemeral: false` on that specific Bash call. Examples of when " +
  "to set ephemeral: false: capturing a version string you will use later, output you will diff or " +
  "compare against, data you will parse across multiple steps.";

const rewindPrompt = "\\n\\n- Rewind(turns_back, summary): Abandon your current approach by rewinding the conversation. " +
  "Everything from turns_back ago to now is replaced with your summary. " +
  "Include: what you tried, why it failed, what you learned, and any file changes made (rewind does NOT undo file changes). " +
  "Use Rewind when you have failed 2+ times with the same approach or realize you are on the wrong track. " +
  "Rewind early -- do not burn context on a dead end.";

const SYSPROMPT_REPLACEMENT =
  'var TL1="You are Claude Code, Anthropic\'s official CLI for Claude."' +
  '+(typeof globalThis.__REWIND_MODE__!=="undefined"&&globalThis.__REWIND_MODE__!=="off"' +
  '?"' + ephemeralPrompt + '"' +
  '+(globalThis.__REWIND_MODE__==="full"?"' + rewindPrompt + '":"")' +
  ':"")';

patch("System prompt addition", SYSPROMPT_ANCHOR, SYSPROMPT_REPLACEMENT);


// ─────────────────────────────────────────────────────────────
// Done — write the patched file
// ─────────────────────────────────────────────────────────────

if (checkOnly) {
  console.log(`\n${patchCount} patches verified (dry run, no file written)`);
} else {
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, src);
  console.log(`\n${patchCount} patches applied → ${outPath}`);
}
