#!/usr/bin/env node
/**
 * Context-aware proxy for Claude Code.
 *
 * Sits between Claude Code and the Anthropic API.
 * Intercepts requests to compact tool outputs and detect stuck reasoning.
 *
 * Usage:
 *   node proxy.mjs &
 *   ANTHROPIC_BASE_URL=http://localhost:8080 claude "your prompt"
 *
 * Environment:
 *   PROXY_PORT          — listen port (default: 8080)
 *   PROXY_UPSTREAM      — upstream API (default: https://api.anthropic.com)
 *   COMPACT_ENABLED     — enable auto-compact (default: 1)
 *   STUCK_ENABLED       — enable stuck detection (default: 1)
 *   See compact.mjs and stuck.mjs for additional config.
 */

import { createServer } from "http";
import { compact } from "./compact.mjs";
import { pruneIfStuck, resetState } from "./stuck.mjs";
import { log, logRequest } from "./log.mjs";

const PORT = parseInt(process.env.PROXY_PORT || "8080", 10);
const UPSTREAM = process.env.PROXY_UPSTREAM || "https://api.anthropic.com";
const COMPACT_ENABLED = process.env.COMPACT_ENABLED !== "0";
const STUCK_ENABLED = process.env.STUCK_ENABLED !== "0";

log("proxy_start", {
  port: PORT,
  upstream: UPSTREAM,
  compactEnabled: COMPACT_ENABLED,
  stuckEnabled: STUCK_ENABLED,
});

const server = createServer(async (req, res) => {
  // Collect request body
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  let body = Buffer.concat(chunks);

  const isMessagesEndpoint =
    req.url?.startsWith("/v1/messages") && req.method === "POST";

  if (isMessagesEndpoint) {
    try {
      const parsed = JSON.parse(body.toString());
      const originalCount = parsed.messages?.length || 0;

      // 1. Auto-compact stale Bash tool results
      if (COMPACT_ENABLED && Array.isArray(parsed.messages)) {
        parsed.messages = compact(parsed.messages, log);
      }

      // 2. Stuck detection — analyze thinking, inject nudge if needed
      if (STUCK_ENABLED && Array.isArray(parsed.messages)) {
        parsed.messages = pruneIfStuck(parsed.messages, log);
      }

      body = Buffer.from(JSON.stringify(parsed));

      logRequest(req.method, req.url, originalCount);
    } catch (e) {
      log("parse_error", { error: e.message, url: req.url });
      // Forward original body on parse error
    }
  }

  // Build upstream headers — forward everything except host
  const upstreamHeaders = {};
  for (const [key, value] of Object.entries(req.headers)) {
    if (key === "host") continue;
    if (key === "content-length") {
      upstreamHeaders[key] = body.length;
      continue;
    }
    upstreamHeaders[key] = value;
  }

  try {
    // Request uncompressed response so we can stream it cleanly
    upstreamHeaders["accept-encoding"] = "identity";

    const upstreamUrl = UPSTREAM + req.url;
    const upstreamRes = await fetch(upstreamUrl, {
      method: req.method,
      headers: upstreamHeaders,
      body: req.method === "POST" || req.method === "PUT" ? body : undefined,
      redirect: "follow",
    });

    // Forward response status and headers
    const responseHeaders = {};
    upstreamRes.headers.forEach((value, key) => {
      // Skip headers that node http server manages
      if (key === "transfer-encoding" && value === "chunked") return;
      responseHeaders[key] = value;
    });

    if (upstreamRes.status !== 200) {
      log("upstream_non200", {
        status: upstreamRes.status,
        url: req.url,
      });
    }

    res.writeHead(upstreamRes.status, responseHeaders);

    // Stream response body through
    if (upstreamRes.body) {
      const reader = upstreamRes.body.getReader();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          res.write(value);
        }
      } catch (e) {
        log("stream_error", { error: e.message });
      } finally {
        reader.releaseLock();
      }
    }

    res.end();
  } catch (e) {
    log("upstream_error", { error: e.message, url: req.url });
    res.writeHead(502, { "content-type": "application/json" });
    res.end(
      JSON.stringify({
        type: "error",
        error: {
          type: "proxy_error",
          message: `Proxy failed to reach upstream: ${e.message}`,
        },
      }),
    );
  }
});

server.listen(PORT, () => {
  console.log(`[stuck-detector-proxy] listening on :${PORT} → ${UPSTREAM}`);
  console.log(`[stuck-detector-proxy] compact=${COMPACT_ENABLED} stuck=${STUCK_ENABLED}`);
  console.log(`[stuck-detector-proxy] usage: ANTHROPIC_BASE_URL=http://localhost:${PORT} claude "..."`);
});

// Clean shutdown
process.on("SIGINT", () => {
  log("proxy_stop", {});
  process.exit(0);
});
process.on("SIGTERM", () => {
  log("proxy_stop", {});
  process.exit(0);
});
