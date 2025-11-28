// Background worker keeps Instagram blocked until the local gate grants access
const GATE_URL = 'http://localhost:5000';
const INSTAGRAM_DOMAINS = ['www.instagram.com', 'instagram.com'];

async function isServerRunning() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000); // bail out quickly if server is dead
    
    const response = await fetch(`${GATE_URL}/status`, { 
      method: 'GET',
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    return response.ok;
  } catch (error) {
    return false;
  }
}

async function isVerified() {
  const serverRunning = await isServerRunning();
  if (!serverRunning) {
    return false;
  }

  const result = await chrome.storage.local.get(['age_gate_token']);
  const token = result.age_gate_token;
  
  if (!token) {
    return false;
  }

  try {
    const response = await fetch(`${GATE_URL}/api/check_verification?token=${encodeURIComponent(token)}`);
    const data = await response.json();
    return data.verified === true;
  } catch (error) {
    console.error('Verification check failed:', error);
    return false;
  }
}

async function checkOnceOnStart() {
  const result = await chrome.storage.local.get(['age_gate_token']);
  const verified = await isVerified();
  
  if (!verified) {
    chrome.tabs.query({ url: ['https://www.instagram.com/*', 'https://instagram.com/*'] }, (tabs) => {
      if (tabs.length > 0) {
        chrome.tabs.update(tabs[0].id, { url: GATE_URL });
      }
    });
  }
}

checkOnceOnStart();

chrome.webRequest.onBeforeRequest.addListener(
  async function(details) {
    const url = new URL(details.url);
    if (INSTAGRAM_DOMAINS.includes(url.hostname) && 
        (url.pathname === '/' || url.pathname.startsWith('/accounts/') || url.pathname.startsWith('/explore'))) {
      
      const verified = await isVerified();
      
      if (!verified) {
        return { redirectUrl: GATE_URL };
      }
    }
  },
  { urls: ["https://www.instagram.com/*", "https://instagram.com/*"] },
  ["blocking"]
);

chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status === 'loading' && tab.url) {
    try {
      const url = new URL(tab.url);
      
      if (INSTAGRAM_DOMAINS.includes(url.hostname) && 
          (url.pathname === '/' || url.pathname.startsWith('/accounts/') || url.pathname.startsWith('/explore'))) {
        
        const verified = await isVerified();
        
        if (!verified) {
          chrome.tabs.update(tabId, { url: GATE_URL });
        }
      }
    } catch (e) {
    }
  }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'checkVerification') {
    isVerified().then(verified => {
      sendResponse({ verified });
    });
    return true; // keep channel open for async response
  }
  
  if (request.action === 'checkServer') {
    isServerRunning().then(running => {
      sendResponse({ running });
    });
    return true;
  }
  
  if (request.action === 'clearToken') {
    chrome.storage.local.remove('age_gate_token', () => {
      sendResponse({ success: true });
    });
    return true;
  }
});

