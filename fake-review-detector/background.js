// Background Service Worker (Manifest V3)
// - Receives batched review payloads from content scripts
// - Calls backend /api/predict
// - Returns predictions to content script
// - Logs results back to backend for continuous learning

const DEFAULT_BACKEND_BASE = 'http://localhost:8000'; // Dev default; non-local will be forced to HTTPS

// In-memory queue to coalesce requests across tabs quickly
let queue = [];
let flushTimer = null;
const FLUSH_INTERVAL_MS = 600; // small delay to batch quickly
const MAX_BATCH = 8;

// Rate limiter: max 5 requests per 10 seconds
const RATE_LIMIT_MAX = 5;
const RATE_LIMIT_WINDOW_MS = 10_000;
let requestTimestamps = [];

// Basic stats persisted in storage for popup analytics
async function updateStats(domain, count, avgProb) {
  const key = 'frd_stats';
  const data = (await chrome.storage.local.get(key))[key] || {
    totalReviews: 0,
    totalPredictions: 0,
    domains: {}
  };
  data.totalReviews += count;
  data.totalPredictions += count;
  if (!data.domains[domain]) {
    data.domains[domain] = { reviews: 0, avgProb: 0 };
  }
  const d = data.domains[domain];
  // Recompute running average
  const total = d.reviews + count;
  d.avgProb = total === 0 ? 0 : (d.avgProb * d.reviews + avgProb * count) / total;
  d.reviews = total;
  await chrome.storage.local.set({ [key]: data });
}

async function getBackendBase() {
  const key = 'frd_backend_base';
  const stored = await chrome.storage.local.get(key);
  const base = stored[key] || DEFAULT_BACKEND_BASE;
  return normalizeBackendBase(base);
}

function isLocalhostUrl(u) {
  try {
    const url = new URL(u);
    return (
      url.hostname === 'localhost' ||
      url.hostname === '127.0.0.1' ||
      url.hostname === '::1'
    );
  } catch { return false; }
}

function normalizeBackendBase(base) {
  // Force HTTPS for non-local URLs
  try {
    const url = new URL(base);
    if (!isLocalhostUrl(base) && url.protocol !== 'https:') {
      url.protocol = 'https:';
      return url.toString().replace(/\/$/, '');
    }
    return base.replace(/\/$/, '');
  } catch {
    return base;
  }
}

async function isLoggingEnabled() {
  const key = 'frd_logging_enabled';
  const stored = await chrome.storage.local.get(key);
  return !!stored[key];
}

async function appendRequestLog(entry) {
  const key = 'frd_request_logs';
  const stored = await chrome.storage.local.get(key);
  const logs = stored[key] || [];
  logs.push(entry);
  // Keep only the last 200 entries to bound storage
  const trimmed = logs.slice(-200);
  await chrome.storage.local.set({ [key]: trimmed });
}

function pruneOldRequests(now) {
  requestTimestamps = requestTimestamps.filter(ts => now - ts < RATE_LIMIT_WINDOW_MS);
}

async function rateLimitWait() {
  const now = Date.now();
  pruneOldRequests(now);
  if (requestTimestamps.length < RATE_LIMIT_MAX) return; // proceed
  const earliest = requestTimestamps[0];
  const delay = RATE_LIMIT_WINDOW_MS - (now - earliest);
  if (delay > 0) {
    await new Promise(res => setTimeout(res, delay));
  }
}

async function callPredictAPI(items) {
  const base = await getBackendBase();
  const url = base.replace(/\/$/, '') + '/api/predict';
  try { console.debug('[FRD:bg] calling', url, 'items', items.length); } catch {}
  const controller = new AbortController();
  const to = setTimeout(() => controller.abort('timeout'), 25000);
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      reviews: items.map(i => ({
        id: i.clientId,
        text: i.text,
        rating: i.rating,
        url: i.url,
        domain: i.domain
      }))
    }),
    // Explicit CORS mode; no credentials needed
    mode: 'cors',
    cache: 'no-store',
    signal: controller.signal
  });
  clearTimeout(to);
  if (!resp.ok) {
    let body = '';
    try { body = await resp.text(); } catch {}
    throw new Error(`Predict API failed: ${resp.status} ${body?.slice(0,200)}`);
  }
  try { console.debug('[FRD:bg] predict response ok'); } catch {}
  try {
    return await resp.json();
  } catch (e) {
    throw new Error('Predict API JSON parse error: ' + (e?.message || e));
  }
}

async function logPredictionsBack(itemsWithPred) {
  try {
    const base = await getBackendBase();
    const url = base.replace(/\/$/, '') + '/api/predict/log';
    await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ predictions: itemsWithPred })
    });
  } catch (e) {
    // Non-blocking
    console.debug('Log back failed:', e);
  }
}

async function flushQueue() {
  if (queue.length === 0) return;
  const batch = queue.splice(0, MAX_BATCH);
  try {
    await rateLimitWait();
    // mark request time just before sending
    requestTimestamps.push(Date.now());
    let result;
    try {
      result = await callPredictAPI(batch);
    } catch (err) {
      // One quick retry
      try { console.warn('[FRD:bg] retrying predict after error:', err?.message || err); } catch {}
      result = await callPredictAPI(batch);
    }
    // result should map by id: { predictions: [{ id, prediction, prob_fake }] }
    const predictions = result?.predictions || [];
    try { console.debug('[FRD:bg] got predictions', predictions.length); } catch {}

    // Send responses back to the right tabs
    const byTab = new Map();
    for (const p of predictions) {
      const item = batch.find(b => b.clientId === p.id);
      if (!item) continue;
      const key = item.tabId + ':' + item.frameId;
      if (!byTab.has(key)) byTab.set(key, []);
      byTab.get(key).push({ ...p, clientId: item.clientId });
    }

    // Update stats
    const domain = batch[0]?.domain || 'unknown';
    const avgProb = predictions.length
      ? predictions.reduce((s, p) => s + (p.prob_fake || 0), 0) / predictions.length
      : 0;
    updateStats(domain, predictions.length, avgProb);

    // Log back for continuous learning
    const payloadToLog = predictions.map(p => {
      const item = batch.find(b => b.clientId === p.id);
      return {
        id: p.id,
        text: item?.text,
        rating: item?.rating,
        url: item?.url,
        domain: item?.domain,
        prediction: p.prediction,
        prob_fake: p.prob_fake
      };
    });
    logPredictionsBack(payloadToLog);

    // Optional local logging for transparency
    if (await isLoggingEnabled()) {
      await appendRequestLog({
        ts: Date.now(),
        domain: batch[0]?.domain || 'unknown',
        count: batch.length,
        predictions: predictions.map(p => ({ id: p.id, prob_fake: p.prob_fake, prediction: p.prediction }))
      });
    }

    // Respond to content scripts
    for (const [key, list] of byTab.entries()) {
      const [tabIdStr, frameIdStr] = key.split(':');
      const tabId = Number(tabIdStr);
      const frameId = Number(frameIdStr);
      if (!Number.isFinite(tabId)) { try { console.warn('[FRD:bg] skip send, invalid tabId', key); } catch {} ; continue; }
      const opts = { frameId: Number.isFinite(frameId) ? frameId : 0 };
      try {
        chrome.tabs.sendMessage(tabId, { type: 'frd_predictions_ready', predictions: list }, opts);
      } catch (e) {
        try { console.warn('[FRD:bg] sendMessage failed', e); } catch {}
      }
    }
  } catch (e) {
    console.warn('FRD batch error:', e);
    // Notify content scripts so they can update badges to error
    const byTab = new Map();
    for (const item of batch) {
      const key = item.tabId + ':' + item.frameId;
      if (!byTab.has(key)) byTab.set(key, []);
      byTab.get(key).push({ clientId: item.clientId, error: String(e?.message || e) });
    }
    for (const [key, list] of byTab.entries()) {
      const [tabIdStr, frameIdStr] = key.split(':');
      const tabId = Number(tabIdStr);
      const frameId = Number(frameIdStr);
      if (!Number.isFinite(tabId)) { continue; }
      const opts = { frameId: Number.isFinite(frameId) ? frameId : 0 };
      try {
        chrome.tabs.sendMessage(tabId, { type: 'frd_predictions_failed', items: list }, opts);
      } catch {}
    }
  } finally {
    if (queue.length > 0) scheduleFlush();
  }
}

function scheduleFlush() {
  if (flushTimer) return;
  flushTimer = setTimeout(() => {
    flushTimer = null;
    flushQueue();
  }, FLUSH_INTERVAL_MS);
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === 'frd_batch_request') {
    const items = msg.items || [];
    try { console.debug('[FRD:bg] received batch_request', { items: items.length, fromTab: sender.tab?.id }); } catch {}
    for (const it of items) {
      queue.push({
        ...it,
        tabId: sender.tab?.id,
        frameId: sender.frameId ?? 0
      });
    }
    scheduleFlush();
    sendResponse({ ok: true, queued: items.length });
    return true;
  }
  if (msg?.type === 'frd_get_backend_base') {
    getBackendBase().then(base => sendResponse({ base }));
    return true;
  }
  if (msg?.type === 'frd_set_backend_base') {
    const normalized = normalizeBackendBase(msg.base || '');
    chrome.storage.local.set({ frd_backend_base: normalized }).then(() => sendResponse({ ok: true, base: normalized }));
    return true;
  }
  if (msg?.type === 'frd_get_logging_enabled') {
    isLoggingEnabled().then(enabled => sendResponse({ enabled }));
    return true;
  }
  if (msg?.type === 'frd_set_logging_enabled') {
    chrome.storage.local.set({ frd_logging_enabled: !!msg.enabled }).then(() => sendResponse({ ok: true }));
    return true;
  }
});
