// Content script that runs on the Flask age gate page
// Reads verification token from localStorage and stores it in extension storage

(function() {
  // Check for token in localStorage (set by insta.html)
  const token = localStorage.getItem('age_gate_token');
  
  if (token) {
    // Store token in extension storage
    chrome.storage.local.set({ age_gate_token: token }, () => {
      console.log('Age gate token stored in extension');
    });
  }
})();

