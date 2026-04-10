#!/usr/bin/env node
/**
 * CNN-based proxy for Claude Code.
 *
 * Sits between Claude Code and the Anthropic API to detect stuck reasoning
 * via a language-agnostic CNN (2,621 params, ~57 KB JS weights).
 *
 * Usage:
 *   node proxy_cnn.mjs &
 *   ANTHROPIC_BASE_URL=http://localhost:8080 claude "your prompt"
 *
 * Environment:
 *   PROXY_PORT         listen port (default: 8080)
 *   PROXY_UPSTREAM     upstream API (default: https://api.anthropic.com)
 *   STUCK_ENABLED      enable stuck detection (default: 1)
 *   COMPACT_ENABLED    enable Bash output compaction (default: 0)
 */

import { createServer } from "http";
import { pruneIfStuck, resetState } from "./stuck_cnn.mjs";
import { log, logRequest } from "./log.mjs";
import { fetchUpstream, getStats } from "./upstream.mjs";

const PORT = parseInt(process.env.PROXY_PORT || "8080", 10);
const UPSTREAM = process.env.PROXY_UPSTREAM || "https://api.anthropic.com";
const COMPACT_ENABLED = process.env.COMPACT_ENABLED === "1";
const STUCK_ENABLED = process.env.STUCK_ENABLED !== "0";

let compact = null;
if (COMPACT_ENABLED) {
  const mod = await import("./compact.mjs");
  compact = mod.compact;
}

log("proxy_cnn_start", {
  port: PORT,
  upstream: UPSTREAM,
  compactEnabled: COMPACT_ENABLED,
  stuckEnabled: STUCK_ENABLED,
  classifier: "cnn_v2",
  ...getStats(),
});

const server = createServer(async (req, res) => {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  let body = Buffer.concat(chunks);

  if (req.url === "/stats" && req.method === "GET") {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(getStats()));
    return;
  }

  const isMessagesEndpoint =
    req.url?.startsWith("/v1/messages") && req.method === "POST";

  if (isMessagesEndpoint) {
    try {
      const parsed = JSON.parse(body.toString());
      const originalCount = parsed.messages?.length || 0;

      if (COMPACT_ENABLED && compact && Array.isArray(parsed.messages)) {
        parsed.messages = compact(parsed.messages, log);
      }

      if (STUCK_ENABLED && Array.isArray(parsed.messages)) {
        parsed.messages = pruneIfStuck(parsed.messages, log);
      }

      body = Buffer.from(JSON.stringify(parsed));
      logRequest(req.method, req.url, originalCount);
    } catch (e) {
      log("parse_error", { error: e.message, url: req.url });
    }
  }

  // Build upstream headers
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
    upstreamHeaders["accept-encoding"] = "identity";
    const upstreamUrl = UPSTREAM + req.url;
    const upstreamRes = await fetchUpstream(
      upstreamUrl,
      {
        method: req.method,
        headers: upstreamHeaders,
        body: req.method === "POST" || req.method === "PUT" ? body : undefined,
        redirect: "follow",
      },
      log,
    );

    const responseHeaders = {};
    upstreamRes.headers.forEach((value, key) => {
      if (key === "transfer-encoding" && value === "chunked") return;
      responseHeaders[key] = value;
    });

    if (upstreamRes.status !== 200) {
      log("upstream_non200", { status: upstreamRes.status, url: req.url });
    }

    res.writeHead(upstreamRes.status, responseHeaders);

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
    res.end(JSON.stringify({ error: "upstream_error", message: e.message }));
  }
});

server.listen(PORT, () => {
  log("proxy_cnn_listening", { port: PORT });
});
