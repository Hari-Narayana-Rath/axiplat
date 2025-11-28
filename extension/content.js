// Pulls the issued token from the AXIPLAT page and hands it to the extension
(function() {
  const token = localStorage.getItem('age_gate_token');
  
  if (token) {
    chrome.storage.local.set({ age_gate_token: token }, () => {
      console.log('AXIPLAT token stored in extension');
    });
  }
})();

