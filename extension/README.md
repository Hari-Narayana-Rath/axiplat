# AXIPLAT Browser Extension

This Chrome/Edge extension automatically intercepts Instagram navigation and redirects users to the AXIPLAT verification page if they haven't been verified.

## Installation

1. Open Chrome/Edge and navigate to `chrome://extensions/` (or `edge://extensions/`)
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `extension` folder from this project
5. The extension will now be active

## How It Works

- When a user tries to visit Instagram, the extension checks if they have a valid verification token
- If not verified, they are redirected to `http://localhost:5000` (the AXIPLAT Flask app)
- After successful verification, a token is stored in browser storage
- The token is valid for 24 hours
- Instagram access is allowed only with a valid token

## Requirements

- The Flask app (`app.py`) must be running on `localhost:5000`
- Make sure to allow camera permissions when prompted

## Icons

The extension needs icon files (`icon16.png`, `icon48.png`, `icon128.png`). You can create simple placeholder icons or use any 16x16, 48x48, and 128x128 pixel images.

