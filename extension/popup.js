// Popup script
const GATE_URL = 'http://localhost:5000';

async function checkServerStatus() {
  try {
    const response = await chrome.runtime.sendMessage({ action: 'checkServer' });
    updateServerUI(response.running);
    return response.running;
  } catch (error) {
    updateServerUI(false);
    return false;
  }
}

function updateServerUI(running) {
  const serverStatusDiv = document.getElementById('serverStatus');
  const startServerBtn = document.getElementById('startServerBtn');
  const infoDiv = document.getElementById('info');
  
  if (running) {
    serverStatusDiv.textContent = '✓ Server: Online';
    serverStatusDiv.className = 'server-status online';
    startServerBtn.style.display = 'none';
    infoDiv.textContent = 'Server is running on localhost:5000';
  } else {
    serverStatusDiv.textContent = '✗ Server: Offline';
    serverStatusDiv.className = 'server-status offline';
    startServerBtn.style.display = 'block';
    infoDiv.innerHTML = 'Server is not running. Click "Start Server" to launch it.<br><small>Or run: <code>python app.py</code></small>';
  }
}

async function checkStatus() {
  const serverRunning = await checkServerStatus();
  
  if (!serverRunning) {
    const statusDiv = document.getElementById('status');
    statusDiv.textContent = '⚠ Server offline - Cannot verify';
    statusDiv.className = 'status server-down';
    return;
  }
  
  try {
    const response = await chrome.runtime.sendMessage({ action: 'checkVerification' });
    updateUI(response.verified);
  } catch (error) {
    updateUI(false);
  }
}

function updateUI(verified) {
  const statusDiv = document.getElementById('status');
  const verifyBtn = document.getElementById('verifyBtn');
  
  if (verified) {
    statusDiv.textContent = '✓ Verified - Instagram access allowed';
    statusDiv.className = 'status verified';
    verifyBtn.textContent = 'Re-verify Age';
  } else {
    statusDiv.textContent = '✗ Not verified - Instagram blocked';
    statusDiv.className = 'status unverified';
    verifyBtn.textContent = 'Verify Age Now';
  }
}

document.getElementById('verifyBtn').addEventListener('click', () => {
  chrome.tabs.create({ url: GATE_URL });
  window.close();
});

document.getElementById('startServerBtn').addEventListener('click', async () => {
  // Redirect to GitHub launcher repository
  chrome.tabs.create({ url: 'https://github.com/Hari-Narayana-Rath/axiplat-server' });
  window.close();
});

document.getElementById('clearBtn').addEventListener('click', async () => {
  try {
    await chrome.runtime.sendMessage({ action: 'clearToken' });
    updateUI(false);
  } catch (error) {
    console.error('Failed to clear token:', error);
  }
});

// Check status on load and periodically
checkStatus();
setInterval(checkStatus, 5000); // Check every 5 seconds

