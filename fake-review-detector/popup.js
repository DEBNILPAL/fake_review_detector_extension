// Popup script
// - Shows basic analytics from chrome.storage
// - Lets user set backend base URL used by background service worker

async function getBackendBase() {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ type: 'frd_get_backend_base' }, (resp) => {
      resolve(resp?.base || 'http://localhost:8000');
    });
  });
}

async function setBackendBase(base) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ type: 'frd_set_backend_base', base }, (resp) => {
      resolve(!!resp?.ok);
    });
  });
}

async function loadStats() {
  const key = 'frd_stats';
  const data = (await chrome.storage.local.get(key))[key] || {
    totalReviews: 0,
    totalPredictions: 0,
    domains: {}
  };
  const domains = data.domains || {};
  const domainKeys = Object.keys(domains);

  document.getElementById('statPredictions').textContent = String(data.totalPredictions || 0);
  document.getElementById('statDomains').textContent = String(domainKeys.length);

  const list = document.getElementById('domainList');
  list.innerHTML = '';
  for (const d of domainKeys) {
    const row = document.createElement('div');
    row.className = 'domain-row';
    const avg = (domains[d].avgProb || 0) * 100;
    row.innerHTML = `
      <div class="left">
        <div class="name">${d}</div>
        <div class="meta">reviews: ${domains[d].reviews || 0}</div>
      </div>
      <div class="right">avg fake: ${Math.round(avg)}%</div>
    `;
    list.appendChild(row);
  }
}

async function init() {
  const base = await getBackendBase();
  const input = document.getElementById('backendBase');
  input.value = base;
  document.getElementById('saveBackend').addEventListener('click', async () => {
    const newBase = input.value.trim();
    if (!newBase) return;
    await setBackendBase(newBase);
    window.close();
  });

  await loadStats();

  // Setup logging toggle
  const toggle = document.getElementById('loggingToggle');
  chrome.runtime.sendMessage({ type: 'frd_get_logging_enabled' }, (resp) => {
    toggle.checked = !!resp?.enabled;
  });
  toggle.addEventListener('change', () => {
    chrome.runtime.sendMessage({ type: 'frd_set_logging_enabled', enabled: toggle.checked }, () => {});
  });
}

init();
