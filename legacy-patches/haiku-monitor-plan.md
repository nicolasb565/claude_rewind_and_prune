# Haiku Monitor Implementation Plan

## Patch Points (all in minified cli.js)

### 1. Thinking stream heuristic + Haiku call + compaction
**Anchor:** `c3("query_recursive_call"),J={messages:[...F,...X6,...P6]`
**Available variables:**
- `X6` — assistant messages array (thinking blocks in `.message.content` where `.type === "thinking"`)
- `F` — conversation messages at start of turn
- `P6` — tool results from this turn
- `pL()` — Anthropic client factory (call with `{maxRetries:0, model:"claude-haiku-4-5-20251001", source:"haiku_monitor"}`)
- `i8()` — user message constructor
- `v` — toolUseContext (has abortController)

### 2. Haiku model ID
`claude-haiku-4-5-20251001` (from system prompt in this conversation)

## Implementation

```javascript
// After tool execution, before building J for next iteration:

(async function haikuMonitor() {
  if (!globalThis.__HAIKU_MONITOR__) return;
  
  // 1. Extract thinking text from this turn's assistant messages
  var thinkingText = "";
  for (var xi = 0; xi < X6.length; xi++) {
    var content = X6[xi].message?.content;
    if (!Array.isArray(content)) continue;
    for (var ci = 0; ci < content.length; ci++) {
      if (content[ci].type === "thinking" && content[ci].thinking) {
        thinkingText += content[ci].thinking;
      }
    }
  }
  
  if (thinkingText.length < 500) return; // too short to analyze
  
  // 2. Heuristic: is the thinking suspicious?
  var suspicious = false;
  
  // Check for repeated substrings (20+ chars appearing 3+ times)
  var seen = {};
  for (var si = 0; si < thinkingText.length - 20; si += 10) {
    var sub = thinkingText.slice(si, si + 20);
    seen[sub] = (seen[sub] || 0) + 1;
    if (seen[sub] >= 3) { suspicious = true; break; }
  }
  
  // Check for "going in circles" keywords
  if (!suspicious) {
    var circleWords = /\b(try again|let me try|another approach|actually|wait|hmm|let me reconsider)\b/gi;
    var matches = thinkingText.match(circleWords);
    if (matches && matches.length >= 5) suspicious = true;
  }
  
  // Check for high overlap between first half and second half
  if (!suspicious && thinkingText.length > 1000) {
    var half = Math.floor(thinkingText.length / 2);
    var words1 = new Set(thinkingText.slice(0, half).toLowerCase().split(/\s+/));
    var words2 = new Set(thinkingText.slice(half).toLowerCase().split(/\s+/));
    var overlap = 0;
    words2.forEach(function(w) { if (words1.has(w)) overlap++; });
    if (overlap / words2.size > 0.6) suspicious = true;
  }
  
  if (!suspicious) return;
  
  // 3. Call Haiku
  var client = await pL({maxRetries: 0, model: "claude-haiku-4-5-20251001", source: "haiku_monitor"});
  var lastThinking = thinkingText.slice(-2000); // last ~2000 chars
  
  var response = await client.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 500,
    system: "You monitor another AI's thinking during a coding task. You are given a chunk of its internal reasoning. Determine if it is going in circles — retrying the same approach, repeating similar reasoning, or stuck. If stuck, call the stuck tool with a concise summary of what was tried and why it failed. If it is making genuine progress (new insights, different approaches, converging on a solution), respond with just OK.",
    tools: [{
      name: "stuck",
      description: "Call when the model is stuck in a loop.",
      input_schema: {
        type: "object",
        properties: {
          summary: {
            type: "string",
            description: "What was tried and why it failed. Be concise."
          }
        },
        required: ["summary"]
      }
    }],
    messages: [{role: "user", content: "Here is the AI's recent thinking:\n\n" + lastThinking}]
  });
  
  // 4. Check if Haiku called the stuck tool
  var stuckCall = response.content.find(function(b) { return b.type === "tool_use" && b.name === "stuck"; });
  if (!stuckCall) return; // Haiku says it's fine
  
  var summary = stuckCall.input.summary;
  
  // 5. Compact: find where repetition started and prune
  // Heuristic: remove the last N assistant turns where N = number of 
  // assistant messages that share similar tool calls
  var assistantIndices = [];
  var allMsgs = [...F, ...X6, ...P6];
  for (var ri = 0; ri < allMsgs.length; ri++) {
    if (allMsgs[ri].type === "assistant") assistantIndices.push(ri);
  }
  
  // Prune last 3 turns (or fewer if not enough)
  var turnsBack = Math.min(3, assistantIndices.length - 1);
  if (turnsBack < 1) return;
  
  var cutAt = assistantIndices[assistantIndices.length - turnsBack];
  var kept = allMsgs.slice(0, cutAt);
  kept.push(i8({
    content: "[CONTEXT COMPACTED — previous approach did not work]\n\n" + summary,
    isMeta: true,
  }));
  
  F.length = 0;
  F.push(...kept);
  X6.length = 0;
  P6.length = 0;
  
  if (globalThis.__REWIND_LOG__) {
    globalThis.__REWIND_LOG__("haiku_compaction", {
      thinkingLength: thinkingText.length,
      summary: summary,
      turnsPruned: turnsBack,
    });
  }
})(),

J = {messages: [...F, ...X6, ...P6], ...}
```

## Issues to resolve

1. The `await` in the IIFE — the anchor is inside a synchronous expression (`c3(...), J = {...}`). Need to restructure so the Haiku call can be awaited. Options:
   - The containing function `A$Y` is `async function*` so we can use await
   - May need to split the comma expression and add `await` before the J assignment

2. The `pL` function requires model param — need to verify Haiku model string works with the existing auth

3. Cost tracking — should log Haiku token usage

## Config

```
HAIKU_MONITOR=1          # enable/disable
HAIKU_MONITOR_PRUNE=3    # how many turns to prune when stuck
```
