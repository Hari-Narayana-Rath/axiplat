# AXIPLAT
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

> **Note:** The `face-recognition` package requires dlib. On Windows install the [Visual C++ build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and ensure CMake is available; on macOS/Linux make sure you have a working C/C++ toolchain.

## Run the App

```bash
python app.py
```

Flask runs in debug mode on `http://127.0.0.1:5000/`. Open that URL in a browser, allow camera access, and follow the on-screen instructions. Press `Ctrl+C` in the terminal to stop the server.

