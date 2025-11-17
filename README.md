# Age Gate Snapchat Clone

Simple Flask-based camera age gate that combines OpenCV DNN predictions with an optional TensorFlow CNN regressor, plus a dark UI flow for redirecting to Instagram when access is granted.

## Local Setup

```bash
git clone https://github.com/Hari-Narayana-Rath/axiplat.git
cd axiplat
```

Create a virtual environment (replace `python` with `python3` if needed):

```bash
python -m venv venv
```

Activate it:

```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
python app.py
```

Flask runs in debug mode on `http://127.0.0.1:5000/`. Open that URL in a browser, allow camera access, and follow the on-screen instructions. Press `Ctrl+C` in the terminal to stop the server.

## Browser Extension (Optional)

The project includes a Chrome/Edge/Brave browser extension that automatically intercepts Instagram navigation and redirects to the age gate if the user hasn't been verified.

### Setup Extension

1. Navigate to the `extension` folder
2. See `extension/README.md` for installation instructions
3. **Note:** Icon files are optional - the extension works without them

### Portable Server Setup

The extension now automatically detects if the server is running and provides easy ways to start it:

**Quick Start Options:**
- **Windows**: Double-click `start_server.bat` in the project root
- **Mac/Linux**: Run `./start_server.sh` in the project root
- **Hidden (Windows)**: Double-click `start_server_hidden.vbs` to run server in background

**Extension Features:**
- Automatically checks if server is running every 5 seconds
- Shows server status in the extension popup
- "Start Server" button appears when server is offline
- Provides instructions to launch the server

### How It Works

- When a user tries to visit Instagram, the extension checks for a valid verification token
- If not verified, they're redirected to the age gate (`localhost:5000`)
- After successful age verification, a token is stored (valid for 24 hours)
- Instagram access is only allowed with a valid token
- Extension automatically detects when server starts/stops

**No need to manually run the server in terminal** - just use the launcher scripts or the extension will guide you!

