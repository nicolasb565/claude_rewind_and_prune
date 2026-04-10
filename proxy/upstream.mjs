/**
 * Resilient upstream fetch with retry, backoff, and concurrency limiting.
 *
 * Environment:
 *   PROXY_MAX_CONCURRENT  — max in-flight upstream requests (default: 5)
 *   PROXY_MAX_RETRIES     — max retries on 429/529 (default: 8)
 *   PROXY_BASE_DELAY_MS   — initial backoff delay in ms (default: 1000)
 *   PROXY_MAX_DELAY_MS    — max backoff delay in ms (default: 60000)
 *   PROXY_QUEUE_TIMEOUT   — max time waiting in queue in ms (default: 300000)
 */

const MAX_CONCURRENT = parseInt(process.env.PROXY_MAX_CONCURRENT || "5", 10);
const MAX_RETRIES = parseInt(process.env.PROXY_MAX_RETRIES || "8", 10);
const BASE_DELAY_MS = parseInt(process.env.PROXY_BASE_DELAY_MS || "1000", 10);
const MAX_DELAY_MS = parseInt(process.env.PROXY_MAX_DELAY_MS || "60000", 10);
const QUEUE_TIMEOUT = parseInt(process.env.PROXY_QUEUE_TIMEOUT || "300000", 10);

// ---------- Semaphore (concurrency limiter) ----------

let inFlight = 0;
const waitQueue = []; // Array of { resolve, timer }

function acquireSlot() {
  if (inFlight < MAX_CONCURRENT) {
    inFlight++;
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const entry = { resolve: null, timer: null };
    entry.timer = setTimeout(() => {
      const idx = waitQueue.indexOf(entry);
      if (idx !== -1) waitQueue.splice(idx, 1);
      reject(new Error(`Queue timeout: waited ${QUEUE_TIMEOUT}ms for a slot`));
    }, QUEUE_TIMEOUT);
    entry.resolve = () => {
      clearTimeout(entry.timer);
      resolve();
    };
    waitQueue.push(entry);
  });
}

function releaseSlot() {
  if (waitQueue.length > 0) {
    const next = waitQueue.shift();
    next.resolve();
  } else {
    inFlight--;
  }
}

// ---------- Retry with backoff ----------

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function getRetryDelay(attempt, retryAfterHeader) {
  // Respect Retry-After header if present
  if (retryAfterHeader) {
    const secs = parseFloat(retryAfterHeader);
    if (!isNaN(secs) && secs > 0) {
      // Add jitter: 0-25% extra
      return Math.min(secs * 1000 * (1 + Math.random() * 0.25), MAX_DELAY_MS);
    }
  }
  // Exponential backoff with full jitter
  const expDelay = BASE_DELAY_MS * Math.pow(2, attempt);
  return Math.min(expDelay * (0.5 + Math.random() * 0.5), MAX_DELAY_MS);
}

const RETRYABLE_STATUS = new Set([429, 529, 502, 503]);

/**
 * Fetch from upstream with concurrency limiting and retry.
 * Returns the Response object. Throws on unrecoverable errors.
 */
export async function fetchUpstream(url, options, log) {
  await acquireSlot();
  try {
    return await fetchWithRetry(url, options, log);
  } finally {
    releaseSlot();
  }
}

async function fetchWithRetry(url, options, log) {
  let lastError = null;
  let lastResponse = null;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const res = await fetch(url, options);

      if (!RETRYABLE_STATUS.has(res.status)) {
        return res; // Success or non-retryable error
      }

      // Retryable status — log and maybe retry
      lastResponse = res;
      const retryAfter = res.headers.get("retry-after");
      const delay = getRetryDelay(attempt, retryAfter);

      log?.("upstream_retry", {
        attempt: attempt + 1,
        maxRetries: MAX_RETRIES,
        status: res.status,
        retryAfter,
        delayMs: Math.round(delay),
        url,
        queueDepth: waitQueue.length,
        inFlight,
      });

      // Consume body to free the connection
      try { await res.text(); } catch {}

      if (attempt < MAX_RETRIES) {
        await sleep(delay);
      }
    } catch (e) {
      // Network errors (ECONNRESET, ETIMEDOUT, etc.)
      lastError = e;

      if (attempt < MAX_RETRIES) {
        const delay = getRetryDelay(attempt, null);
        log?.("upstream_network_retry", {
          attempt: attempt + 1,
          maxRetries: MAX_RETRIES,
          error: e.message,
          delayMs: Math.round(delay),
          url,
        });
        await sleep(delay);
      }
    }
  }

  // Exhausted retries
  if (lastResponse) {
    log?.("upstream_retries_exhausted", {
      status: lastResponse.status,
      attempts: MAX_RETRIES + 1,
      url,
    });
    return lastResponse;
  }

  throw lastError || new Error("Upstream fetch failed after retries");
}

/**
 * Get current queue/concurrency stats for diagnostics.
 */
export function getStats() {
  return {
    inFlight,
    queueDepth: waitQueue.length,
    maxConcurrent: MAX_CONCURRENT,
  };
}
