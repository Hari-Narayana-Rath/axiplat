// Age Gate Extension - Background Service Worker
const GATE_URL = 'http://localhost:5000';
const INSTAGRAM_DOMAINS = ['www.instagram.com', 'instagram.com'];

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
  const result = await chrome.storage.local.get(['age_gate_token']);
  
  // Check if verified
  const verified = await isVerified();
  
  if (!verified) {
    // Find or create Instagram tab and redirect to age gate
    chrome.tabs.query({ url: ['https://www.instagram.com/*', 'https://instagram.com/*'] }, (tabs) => {
      if (tabs.length > 0) {
        // Redirect existing Instagram tab
        chrome.tabs.update(tabs[0].id, { url: GATE_URL });
      }
    });
  }
}

// Run check once when extension starts
checkOnceOnStart();

// Intercept Instagram navigation - ALWAYS block if not verified
chrome.webRequest.onBeforeRequest.addListener(
  async function(details) {
    const url = new URL(details.url);
    
    // Only intercept Instagram main pages
    if (INSTAGRAM_DOMAINS.includes(url.hostname) && 
        (url.pathname === '/' || url.pathname.startsWith('/accounts/') || url.pathname.startsWith('/explore'))) {
      
      const verified = await isVerified();
      
      if (!verified) {
        // Always redirect to age gate if not verified
        return { redirectUrl: GATE_URL };
      }
    }
  },
  { urls: ["https://www.instagram.com/*", "https://instagram.com/*"] },
  ["blocking"]
);

// Also listen for tab updates to catch navigation
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status === 'loading' && tab.url) {
    try {
      const url = new URL(tab.url);
      
      if (INSTAGRAM_DOMAINS.includes(url.hostname) && 
          (url.pathname === '/' || url.pathname.startsWith('/accounts/') || url.pathname.startsWith('/explore'))) {
        
        const verified = await isVerified();
        
        if (!verified) {
          // Redirect to age gate
          chrome.tabs.update(tabId, { url: GATE_URL });
        }
      }
    } catch (e) {
      // Invalid URL, ignore
    }
  }
});

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
    chrome.storage.local.remove('age_gate_token', () => {
      sendResponse({ success: true });
    });
    return true;
  }
});

