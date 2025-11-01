// Content Script
// - Detects review blocks on Amazon/Flipkart (and generic heuristics)
// - Extracts text and rating
// - Batches requests to background for prediction
// - Renders inline badge with probability

(function () {
  const BADGE_CLASS = 'frd-badge';
  const PROCESSED_ATTR = 'data-frd-processed';
  const ID_ATTR = 'data-frd-id';
  const BATCH_INTERVAL = 400;
  const MAX_BATCH = 20;

  // Performance caps
  const MAX_INITIAL_SCAN = 300; // Max review nodes to process initially
  const MAX_MUTATION_PROCESS = 60; // Per mutation flush
  const QUERY_CAP_PER_NODE = 200; // Prevent explosive node scans

  const pending = [];
  let batchTimer = null;

  const processedIds = new Set();
  const processedElements = new WeakSet();

  function uuid() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
      const r = (Math.random() * 16) | 0,
        v = c === 'x' ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  function scheduleBatch() {
    if (batchTimer) return;
    batchTimer = setTimeout(() => {
      batchTimer = null;
      flushBatch();
    }, BATCH_INTERVAL);
  }

  function flushBatch() {
    if (pending.length === 0) return;
    const items = pending.splice(0, MAX_BATCH);
    try { console.debug('[FRD] sending batch', { count: items.length }); } catch {}
    chrome.runtime.sendMessage({ type: 'frd_batch_request', items }, (resp) => {
      if (chrome.runtime.lastError) {
        console.debug('FRD send error', chrome.runtime.lastError);
      }
    });
    if (pending.length > 0) scheduleBatch();
  }

  function buildBadge(prob) {
    const badge = document.createElement('span');
    badge.className = BADGE_CLASS;
    const pct = Math.round((prob || 0) * 100);
    badge.textContent = `🧠 ${pct}% fake`;
    badge.title = 'Fake Review Detector';
    return badge;
  }

  function insertBadge(container, badge) {
    // Prefer to append near rating or header
    const site = detectSite();
    if (site === 'flipkart') {
      const ratingRow = container.querySelector('div._3LWZlK')?.parentElement;
      if (ratingRow && ratingRow.parentElement) {
        ratingRow.parentElement.insertBefore(badge, ratingRow.nextSibling);
        return;
      }
      const title = container.querySelector('._2-N8zT, ._6K-7Co');
      if (title && title.parentElement) {
        title.parentElement.insertBefore(badge, title.nextSibling);
        return;
      }
    }
    const header = container.querySelector('h3,h4,h5,.a-row,.row');
    if (header && header.parentElement) {
      header.parentElement.insertBefore(badge, header.nextSibling);
      return;
    }
    container.appendChild(badge);
  }

  function parseRatingFromText(text) {
    // looks for patterns like "4.0 out of 5" or "4/5" or "Rated 4"
    const m = text.match(/(\d+(?:\.\d+)?)\s*(?:out of\s*)?\/?\s*5/i);
    if (m) return parseFloat(m[1]);
    return null;
  }

  function extractFromAmazon(reviewEl) {
    const textEl =
      reviewEl.querySelector('.review-text-content span') ||
      reviewEl.querySelector('.a-expander-content');
    const ratingEl = reviewEl.querySelector('.review-rating, i[data-hook="review-star-rating"]');
    const text = (textEl?.innerText || '').trim();
    let rating = null;
    if (ratingEl) {
      rating = parseRatingFromText(ratingEl.innerText || ratingEl.getAttribute('aria-label') || '');
    } else {
      rating = parseRatingFromText(reviewEl.innerText || '');
    }
    return { text, rating };
  }

  function extractFromFlipkart(reviewEl) {
    // Flipkart review text usually inside div.t-ZTKy > div
    const textEl = reviewEl.querySelector('div.t-ZTKy, div[class*="t-ZTKy"], div._6K-7Co, div._2-N8zT');
    const ratingEl = reviewEl.querySelector('div._3LWZlK, span._3LWZlK, [aria-label*="out of" i]');
    const text = (textEl?.innerText || '').trim();
    let rating = null;
    if (ratingEl) {
      const lbl = (ratingEl.innerText || ratingEl.getAttribute('aria-label') || '').trim();
      rating = parseFloat(lbl) || parseRatingFromText(lbl) || parseRatingFromText(reviewEl.innerText || '');
    } else {
      rating = parseRatingFromText(reviewEl.innerText || '');
    }
    return { text, rating };
  }

  function extractGeneric(reviewEl) {
    // Generic heuristics
    const textEl = reviewEl.querySelector('p, .content, .review-text, [data-review-text]');
    const ratingEl = reviewEl.querySelector('[aria-label*="out of" i], [class*="rating" i], .stars');
    const text = (textEl?.innerText || reviewEl.innerText || '').trim();
    const rating = ratingEl ? parseRatingFromText(ratingEl.innerText || ratingEl.getAttribute('aria-label') || '') : parseRatingFromText(reviewEl.innerText || '');
    return { text, rating };
  }

  function extractFromMyntra(reviewEl) {
    // Myntra review blocks often have text in p tags and ratings as numbers (e.g., 4) with star icons
    const textEl = reviewEl.querySelector('p, .user-review-text, .review-text');
    // Try explicit numeric rating first
    let rating = null;
    const explicit = reviewEl.querySelector('[class*="rating" i], [aria-label*="rating" i]');
    if (explicit) {
      const t = (explicit.textContent || explicit.getAttribute('aria-label') || '').trim();
      const m = t.match(/(\d+(?:\.\d+)?)/);
      if (m) rating = parseFloat(m[1]);
    }
    if (rating == null) rating = parseRatingFromText(reviewEl.innerText || '');
    const text = (textEl?.innerText || reviewEl.innerText || '').trim();
    return { text, rating };
  }

  function detectSite() {
    const host = location.hostname;
    if (/amazon\./i.test(host)) return 'amazon';
    if (/flipkart\./i.test(host)) return 'flipkart';
    if (/myntra\./i.test(host)) return 'myntra';
    return 'generic';
  }

  function getSelectorsForSite() {
    const site = detectSite();
    if (site === 'amazon') {
      return [
        '[data-hook="review"]',
        '.a-section.review',
        '.a-section.celwidget[data-hook="review"]'
      ];
    } else if (site === 'flipkart') {
      return [
        'div.col._2wzgFH',
        'div._16PBlm',
        'div[data-id^="REVIEW"]',
        'div._27M-vq',
        'div._1AtVbE'
      ];
    } else if (site === 'myntra') {
      return [
        'div[data-reviewid]',
        'li[data-reviewid]',
        'div.user-review, li.user-review',
        'div.index-reviewContainer'
      ];
    }
    return [
      '.review',
      '.user-review',
      '[class*="review" i]'
    ];
  }

  function findReviewElements() {
    const selectors = getSelectorsForSite();
    const all = [];
    try {
      for (const sel of selectors) {
        const nodeList = document.querySelectorAll(sel);
        for (let i = 0; i < nodeList.length && all.length < MAX_INITIAL_SCAN; i++) {
          all.push(nodeList[i]);
        }
        if (all.length >= MAX_INITIAL_SCAN) break;
      }
    } catch (e) {
      // ignore
    }
    // Flipkart fallback: if nothing matched, use text blocks t-ZTKy and lift to closest review container
    if (all.length === 0 && detectSite() === 'flipkart') {
      try {
        const textBlocks = document.querySelectorAll('div.t-ZTKy, div[class*="t-ZTKy"]');
        for (let i = 0; i < textBlocks.length && all.length < MAX_INITIAL_SCAN; i++) {
          const tb = textBlocks[i];
          const container = tb.closest('div.col._2wzgFH, div._16PBlm, div[data-id^="REVIEW"], div._27M-vq, div._1AtVbE') || tb.parentElement;
          if (container && !all.includes(container)) all.push(container);
        }
      } catch {}
    }
    try { console.debug('[FRD] matched elements', { selectors, count: all.length }); } catch {}
    return all;
  }

  function extractReview(el) {
    const site = detectSite();
    let data = { text: '', rating: null };
    if (site === 'amazon') data = extractFromAmazon(el);
    else if (site === 'flipkart') data = extractFromFlipkart(el);
    else if (site === 'myntra') data = extractFromMyntra(el);
    else data = extractGeneric(el);
    return data;
  }

  function processElement(el) {
    if (!el || el.getAttribute(PROCESSED_ATTR) === '1') return;
    if (processedElements.has(el)) return;
    const { text, rating } = extractReview(el);
    if (!text || text.length < 5) {
      el.setAttribute(PROCESSED_ATTR, '1');
      return; // too short
    }
    const clientId = uuid();
    el.setAttribute(ID_ATTR, clientId);
    el.setAttribute(PROCESSED_ATTR, '1');

    const item = {
      clientId,
      text,
      rating: typeof rating === 'number' ? rating : null,
      url: location.href,
      domain: location.hostname
    };
    processedIds.add(clientId);
    pending.push(item);
    scheduleBatch();

    // placeholder badge while loading
    const placeholder = buildBadge(0);
    placeholder.textContent = '🧠 analyzing…';
    placeholder.classList.add('loading');
    insertBadge(el, placeholder);
  }

  function processAll() {
    const els = findReviewElements();
    for (const el of els) {
      try {
        processElement(el);
        processedElements.add(el);
      } catch {}
    }
  }

  // Listen for prediction results
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg?.type === 'frd_predictions_ready') {
      try { console.debug('[FRD] predictions ready', msg.predictions?.length || 0); } catch {}
      for (const p of msg.predictions || []) {
        const el = document.querySelector(`[${ID_ATTR}="${p.clientId}"]`);
        if (!el) continue;
        // Remove any existing badge
        const old = el.querySelector(`.${BADGE_CLASS}`);
        if (old) old.remove();
        const badge = buildBadge(p.prob_fake);
        insertBadge(el, badge);
      }
    }
    if (msg?.type === 'frd_predictions_failed') {
      try { console.debug('[FRD] predictions failed', msg.items?.length || 0); } catch {}
      for (const f of msg.items || []) {
        const el = document.querySelector(`[${ID_ATTR}="${f.clientId}"]`);
        if (!el) continue;
        const old = el.querySelector(`.${BADGE_CLASS}`);
        if (old) old.remove();
        const badge = document.createElement('span');
        badge.className = BADGE_CLASS;
        badge.textContent = '🧠 error';
        badge.title = 'Prediction failed' + (f.error ? `: ${f.error}` : '');
        insertBadge(el, badge);
      }
    }
  });

  // Observe dynamic content
  const observer = new MutationObserver((mutations) => {
    const selectors = getSelectorsForSite();
    let processedCount = 0;
    for (const m of mutations) {
      if (m.type !== 'childList') continue;
      m.addedNodes.forEach((n) => {
        if (processedCount >= MAX_MUTATION_PROCESS) return;
        if (n.nodeType !== 1) return;
        const node = /** @type {Element} */ (n);
        // If node itself is a review
        for (const sel of selectors) {
          if (node.matches && node.matches(sel)) {
            processElement(node);
            processedElements.add(node);
            processedCount++;
            break;
          }
        }
        // Query for review children but cap results
        for (const sel of selectors) {
          if (processedCount >= MAX_MUTATION_PROCESS) break;
          let list = [];
          try { list = Array.from(node.querySelectorAll(sel)); } catch {}
          for (let i = 0; i < list.length && processedCount < MAX_MUTATION_PROCESS && i < QUERY_CAP_PER_NODE; i++) {
            const el = list[i];
            if (!processedElements.has(el)) {
              processElement(el);
              processedElements.add(el);
              processedCount++;
            }
          }
          if (processedCount >= MAX_MUTATION_PROCESS) break;
        }
      });
    }
  });

  observer.observe(document.documentElement || document.body, {
    childList: true,
    subtree: true
  });

  // Initial scan
  const safeProcessAll = () => {
    if ('requestIdleCallback' in window) {
      requestIdleCallback(() => processAll(), { timeout: 700 });
    } else {
      setTimeout(processAll, 0);
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', safeProcessAll, { once: true });
  } else {
    safeProcessAll();
  }

  try { console.debug('[FRD] content script loaded on', location.hostname); } catch {}
})();
