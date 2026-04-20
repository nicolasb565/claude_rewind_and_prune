#!/usr/bin/env node
/**
 * Bookmark MCP server — probes whether Claude will use agent-managed
 * memory primitives when they're available.
 *
 * Exposes three tools over stdio JSON-RPC (MCP protocol):
 *   bookmark_mark(name, summary)
 *   bookmark_recall(name)
 *   bookmark_list()
 *
 * Storage is in-memory; lifetime = single Claude Code invocation (the
 * server process is spawned and killed with the session). That matches
 * what we're testing: does the agent bookmark within a session?
 *
 * Every request is appended to `$BOOKMARK_LOG_DIR/bookmarks.jsonl` so
 * we can audit *whether* the model ever called these tools and *what*
 * it chose to bookmark, without reading back the full transcript.
 *
 * Protocol: https://spec.modelcontextprotocol.io
 * We implement the minimum to expose tools — initialize, tools/list,
 * tools/call. Everything else returns "method not found".
 */

import { appendFileSync, mkdirSync, existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { randomUUID } from 'node:crypto'
import { createInterface } from 'node:readline'

const SERVER_NAME = 'bookmarks'
const SERVER_VERSION = '0.1.0'
const PROTOCOL_VERSION = '2025-06-18'

const LOG_DIR = process.env.BOOKMARK_LOG_DIR || '/tmp/bookmark_logs'
const LOG_FILE = resolve(LOG_DIR, 'bookmarks.jsonl')
const SESSION_ID = randomUUID()

if (!existsSync(LOG_DIR)) mkdirSync(LOG_DIR, { recursive: true })

function logEvent(event) {
  try {
    appendFileSync(
      LOG_FILE,
      JSON.stringify({ sessionId: SESSION_ID, timestamp: Date.now(), ...event }) + '\n',
    )
  } catch {
    // Best-effort logging; don't crash the server over a log failure.
  }
}

// ── Bookmark store ────────────────────────────────────────────────────────

/** @type {Map<string, { summary: string, created_at: number }>} */
const bookmarks = new Map()

const TOOLS = [
  {
    name: 'bookmark_mark',
    description:
      'Save an important observation, decision, or finding as a named '
      + 'bookmark. Use this proactively when you: (a) identify the root '
      + 'cause of a problem, (b) make a non-obvious design decision you '
      + 'may reference later, (c) learn a fact about the codebase you '
      + 'might need again (e.g. "build flag X is required in file Y"). '
      + 'Bookmarks survive as durable notes that you can recall later by '
      + 'name without re-deriving them. Prefer short, descriptive names '
      + '(snake_case). Summary should be 1-3 sentences capturing the key '
      + 'insight, not a full explanation.',
    inputSchema: {
      type: 'object',
      properties: {
        name: {
          type: 'string',
          description: 'Short, descriptive identifier. snake_case preferred.',
        },
        summary: {
          type: 'string',
          description: 'The observation or finding (1-3 sentences).',
        },
      },
      required: ['name', 'summary'],
    },
  },
  {
    name: 'bookmark_recall',
    description:
      'Retrieve a previously-saved bookmark by name. Use this when you '
      + 'need to reference a prior decision or finding without re-'
      + 'exploring the code. Returns the stored summary, or an error if '
      + 'the name does not exist. Call bookmark_list first if you want '
      + 'to see what bookmarks are available.',
    inputSchema: {
      type: 'object',
      properties: {
        name: { type: 'string', description: 'The bookmark name to look up.' },
      },
      required: ['name'],
    },
  },
  {
    name: 'bookmark_list',
    description:
      'List all bookmarks saved in this session with their names and '
      + 'summaries. Useful for checking what you have captured so far '
      + 'before deciding whether to bookmark something new.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'checkpoint_progress',
    description:
      'Record a checkpoint when you have CONCRETE EVIDENCE of progress. '
      + 'Two valid cases:\n\n'
      + '- milestone_achieved: you observed something specific that '
      + 'constitutes real progress. Examples: the bug reproduces with '
      + 'the expected error, you identified the specific file:line of '
      + 'a bug after reading it, a fix you wrote made a failing test '
      + 'pass, a build now succeeds.\n\n'
      + '- approach_eliminated: you tested a specific hypothesis and '
      + 'specific evidence contradicted it. Examples: "applied patch X, '
      + 'bug still reproduces with same output" — NOT "X seems '
      + 'unrelated". "Searched all callers of Y and none match the '
      + 'pattern" — NOT "Y probably isn\'t involved".\n\n'
      + 'Every checkpoint requires `evidence` — a concrete observation, '
      + 'not a feeling. If you cannot cite a specific tool output, '
      + 'file contents, test result, or build log, you are NOT ready '
      + 'to checkpoint. Keep exploring.\n\n'
      + 'Bad checkpoint: "Explored X extensively, code looks correct."\n'
      + 'Good checkpoint: "Ran the reproducer after reverting '
      + 'simplifyRecipes changes; bug still occurs, confirming that '
      + 'function is not the cause."\n\n'
      + 'When called, prior exploration is elided from future context '
      + 'and replaced with your structured summary. A wrong-shaped '
      + 'checkpoint entrenches wrong conclusions — use it only when '
      + 'you have evidence.',
    inputSchema: {
      type: 'object',
      properties: {
        progress_type: {
          type: 'string',
          enum: ['milestone_achieved', 'approach_eliminated'],
          description:
            'milestone_achieved = you confirmed something concrete. '
            + 'approach_eliminated = you ruled out a specific hypothesis '
            + 'with direct evidence.',
        },
        finding: {
          type: 'string',
          description:
            'What is now true. 1 sentence. Example: "Root cause is '
            + 'missing TYPE_UNSIGNED guard at match.pd:3421." or '
            + '"Hypothesis that simplifyRecipes mutates reduction phis '
            + 'is false."',
        },
        evidence: {
          type: 'string',
          description:
            'The concrete observation that establishes the finding. '
            + 'Must reference specific output, file contents, test '
            + 'results, or build errors. NOT "I looked at X '
            + 'extensively" or "seems correct". If you cannot cite a '
            + 'concrete observation, do not checkpoint.',
        },
        next_direction: {
          type: 'string',
          description: 'What you will do next, in 1 sentence.',
        },
      },
      required: ['progress_type', 'finding', 'evidence', 'next_direction'],
    },
  },
]

// ── Tool handlers ────────────────────────────────────────────────────────

function handleCall(name, args) {
  switch (name) {
    case 'bookmark_mark': {
      const { name: key, summary } = args ?? {}
      if (typeof key !== 'string' || !key.trim()) {
        return toolError('`name` is required and must be a non-empty string')
      }
      if (typeof summary !== 'string' || !summary.trim()) {
        return toolError('`summary` is required and must be a non-empty string')
      }
      const existed = bookmarks.has(key)
      bookmarks.set(key, { summary, created_at: Date.now() })
      logEvent({ type: 'mark', name: key, summary, overwrote: existed })
      return toolOk(
        existed
          ? `Updated bookmark "${key}".`
          : `Saved bookmark "${key}".`,
      )
    }
    case 'bookmark_recall': {
      const { name: key } = args ?? {}
      if (typeof key !== 'string' || !key.trim()) {
        return toolError('`name` is required and must be a non-empty string')
      }
      const hit = bookmarks.get(key)
      logEvent({ type: 'recall', name: key, found: !!hit })
      if (!hit) {
        const available = [...bookmarks.keys()]
        return toolError(
          `No bookmark named "${key}". `
          + (available.length
            ? `Available: ${available.join(', ')}.`
            : 'No bookmarks have been saved yet.'),
        )
      }
      return toolOk(hit.summary)
    }
    case 'bookmark_list': {
      logEvent({ type: 'list', count: bookmarks.size })
      if (bookmarks.size === 0) return toolOk('(no bookmarks saved yet)')
      const lines = []
      for (const [k, v] of bookmarks) lines.push(`- ${k}: ${v.summary}`)
      return toolOk(lines.join('\n'))
    }
    case 'checkpoint_progress': {
      const { progress_type, finding, evidence, next_direction } = args ?? {}
      const missing = []
      if (!['milestone_achieved', 'approach_eliminated'].includes(progress_type)) {
        missing.push('progress_type (must be "milestone_achieved" or "approach_eliminated")')
      }
      for (const [k, v] of [
        ['finding', finding],
        ['evidence', evidence],
        ['next_direction', next_direction],
      ]) {
        if (typeof v !== 'string' || !v.trim()) missing.push(k)
      }
      if (missing.length) {
        return toolError(
          `Missing or invalid fields: ${missing.join(', ')}. A checkpoint requires `
          + `concrete evidence. If you cannot cite a specific tool output, test `
          + `result, or build log, do not checkpoint — keep exploring.`,
        )
      }
      logEvent({ type: 'checkpoint_progress', progress_type, finding, evidence, next_direction })
      // The MCP server is just a marker. The proxy scans for this
      // tool_use on subsequent requests and performs the elision. We
      // ack so the agent knows its call landed.
      const label = progress_type === 'milestone_achieved' ? 'Milestone' : 'Approach eliminated'
      return toolOk(
        `${label}: ${finding}. Evidence: ${evidence}. Next: ${next_direction}.`
        + ` Prior exploration will be condensed for focus; the full history remains accessible via bookmark_recall.`,
      )
    }
    default:
      return toolError(`Unknown tool: ${name}`)
  }
}

function toolOk(text) {
  return { content: [{ type: 'text', text }], isError: false }
}

function toolError(text) {
  return { content: [{ type: 'text', text }], isError: true }
}

// ── JSON-RPC plumbing ────────────────────────────────────────────────────

function send(obj) {
  process.stdout.write(JSON.stringify(obj) + '\n')
}

function reply(id, result) {
  send({ jsonrpc: '2.0', id, result })
}

function replyError(id, code, message) {
  send({ jsonrpc: '2.0', id, error: { code, message } })
}

function handleRequest(req) {
  const { id, method, params } = req
  // Notifications have no id — the protocol allows us to silently accept.
  switch (method) {
    case 'initialize':
      reply(id, {
        protocolVersion: PROTOCOL_VERSION,
        capabilities: { tools: {} },
        serverInfo: { name: SERVER_NAME, version: SERVER_VERSION },
      })
      return
    case 'notifications/initialized':
      logEvent({ type: 'session_start' })
      return
    case 'tools/list':
      logEvent({ type: 'tools_list' })
      reply(id, { tools: TOOLS })
      return
    case 'tools/call': {
      const { name, arguments: args } = params ?? {}
      const result = handleCall(name, args)
      reply(id, result)
      return
    }
    case 'ping':
      reply(id, {})
      return
    default:
      if (id != null) replyError(id, -32601, `method not found: ${method}`)
  }
}

// ── Stdio loop ───────────────────────────────────────────────────────────

const rl = createInterface({ input: process.stdin, crlfDelay: Infinity })
rl.on('line', (line) => {
  const s = line.trim()
  if (!s) return
  let req
  try { req = JSON.parse(s) } catch {
    return // malformed frame — discard
  }
  try { handleRequest(req) } catch (e) {
    if (req.id != null) replyError(req.id, -32603, e.message || String(e))
  }
})
rl.on('close', () => {
  logEvent({ type: 'session_end', bookmark_count: bookmarks.size })
  process.exit(0)
})
