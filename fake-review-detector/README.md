# Fake Review Detector (Chrome Extension, MV3)

A Manifest V3 Chrome Extension that:
- Parses reviews from e-commerce sites (Amazon, Flipkart, generic).
- Sends text + rating to your backend `/api/predict`.
- Renders inline results beside each review (e.g., "🧠 84% fake").
- Logs predictions back to backend for continuous learning.
 - Optional local request logging (toggle in popup).

## Files
- manifest.json
- background.js (service worker)
- contentScript.js
- styles.css (injected for badges)
- popup.html, popup.js, popup.css
- icons/ (placeholders)

## Load in Chrome
1. Build/run your backend locally (default `http://localhost:8000`). Ensure `/api/predict` and `/api/predict/log` exist and CORS allows the extension.
2. Open chrome://extensions.
3. Toggle Developer mode.
4. Click "Load unpacked" and select this `fake-review-detector/` folder.
5. Pin the extension. Open the popup to configure Backend Base if not `http://localhost:8000`.

## Test on Amazon/Flipkart
- Navigate to an Amazon or Flipkart product page with reviews.
- The content script scans the DOM, extracts review text + rating, and shows a temporary "analyzing…" badge.
- After the background batched call returns, badges update like "🧠 84% fake".
- Works with dynamically loaded reviews via MutationObserver.

## Backend Contract
- POST `/api/predict`
  - Body: `{ reviews: [{ id, text, rating, url, domain }] }`
  - Response: `{ predictions: [{ id, prediction, prob_fake }] }`
- POST `/api/predict/log`
  - Body: `{ predictions: [{ id, text, rating, url, domain, prediction, prob_fake }] }`
  - Used for continuous learning; store in a predict table.

## Continuous Learning
- The extension logs predictions to `/api/predict/log`.
- On the backend, store rows with: timestamp, text, rating, domain, url, prediction, prob_fake, model_version.
- Periodically retrain your model from this table (e.g., cron/worker).
- Optionally, expose a feedback API for users to correct labels; incorporate into retraining.

## Notes
 - Batching: content script sends up to 20 reviews per batch with a small delay; background coalesces and calls backend.
 - Rate limiting: background enforces max 5 requests per 10 seconds globally across tabs.
 - HTTPS enforcement: non-local backends (not localhost/127.0.0.1/::1) are automatically coerced to `https://` for requests.
 - CORS: extension requests use `mode: 'cors'`; ensure backend allows the extension origin and required methods/headers.
 - Local logging: enable/disable in popup. Logs are stored under `frd_request_logs` (last 200 entries) in `chrome.storage.local`.
 - Permissions: kept minimal. `host_permissions` includes all to allow fetch to your backend; restrict if desired.
 - Icons: placeholders included in `icons/`. Replace with real PNGs and add an `icons` section in manifest.json.
