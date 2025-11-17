// Age Gate Extension - Background Service Worker
const GATE_URL = 'http://localhost:5000';
const INSTAGRAM_DOMAINS = ['www.instagram.com', 'instagram.com'];
const CHECKED_KEY = 'age_gate_checked_once';

// Check if server is running
async function isServerRunning() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000); // 2 second timeout
    
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

// Check if user is verified
async function isVerified() {
  // First check if server is running
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

// Check verification once on browser start
async function checkOnceOnStart() {
  const result = await chrome.storage.local.get([CHECKED_KEY, 'age_gate_token']);
  
  // Only check once per browser session
  if (result[CHECKED_KEY]) {
    return;
  }
  
  // Mark as checked
  chrome.storage.local.set({ [CHECKED_KEY]: true });
  
  // Check if verified
  const verified = await isVerified();
  
  if (!verified) {
    // Find or create Instagram tab and redirect to age gate
    chrome.tabs.query({ url: ['https://www.instagram.com/*', 'https://instagram.com/*'] }, (tabs) => {
      if (tabs.length > 0) {
        // Redirect existing Instagram tab
        chrome.tabs.update(tabs[0].id, { url: GATE_URL });
      } else {
        // Create new tab with age gate
        chrome.tabs.create({ url: GATE_URL });
      }
    });
  }
}

// Run check once when extension starts
checkOnceOnStart();

// Intercept Instagram navigation - only if not verified
chrome.webRequest.onBeforeRequest.addListener(
  async function(details) {
    const url = new URL(details.url);
    
    // Only intercept Instagram main pages
    if (INSTAGRAM_DOMAINS.includes(url.hostname) && 
        (url.pathname === '/' || url.pathname.startsWith('/accounts/') || url.pathname.startsWith('/explore'))) {
      
      const verified = await isVerified();
      
      if (!verified) {
        // Check if we already checked once
        const result = await chrome.storage.local.get([CHECKED_KEY]);
        if (!result[CHECKED_KEY]) {
          // First time - redirect to age gate
          chrome.storage.local.set({ [CHECKED_KEY]: true });
          return { redirectUrl: GATE_URL };
        }
        // Already checked once, allow access (user can manually go to age gate)
      }
    }
  },
  { urls: ["https://www.instagram.com/*", "https://instagram.com/*"] },
  ["blocking"]
);

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'checkVerification') {
    isVerified().then(verified => {
      sendResponse({ verified });
    });
    return true; // Keep channel open for async response
  }
  
  if (request.action === 'checkServer') {
    isServerRunning().then(running => {
      sendResponse({ running });
    });
    return true;
  }
  
  if (request.action === 'clearToken') {
    chrome.storage.local.remove(['age_gate_token', CHECKED_KEY], () => {
      sendResponse({ success: true });
    });
    return true;
  }
  
  if (request.action === 'resetCheck') {
    chrome.storage.local.remove(CHECKED_KEY, () => {
      sendResponse({ success: true });
    });
    return true;
  }
});

