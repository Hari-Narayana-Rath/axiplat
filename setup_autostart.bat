@echo off
REM Adds the server launcher to Windows startup

echo ========================================
echo AXIPLAT Server - Auto-Start Setup
echo ========================================
echo.

set "SCRIPT_DIR=%~dp0"
set "BAT_FILE=%SCRIPT_DIR%start_server_hidden.bat"

if not exist "%BAT_FILE%" (
    echo ERROR: start_server_hidden.bat not found in:
    echo %SCRIPT_DIR%
    echo.
    echo Please make sure you're running this from the project folder.
    pause
    exit /b 1
)

set "VBS_FILE=%SCRIPT_DIR%run_server_hidden.vbs"
(
echo Set WshShell = CreateObject^("WScript.Shell"^)
echo WshShell.Run """%BAT_FILE%""", 0, False
) > "%VBS_FILE%"

set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "STARTUP_LINK=%STARTUP_FOLDER%\AXIPLAT Server.lnk"

powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%STARTUP_LINK%'); $s.TargetPath = '%VBS_FILE%'; $s.WorkingDirectory = '%SCRIPT_DIR%'; $s.Save()"

if exist "%STARTUP_LINK%" (
    echo.
    echo ✓ Success! AXIPLAT Server has been added to Windows startup.
    echo.
    echo The server will now start automatically when you boot Windows.
    echo It will run hidden in the background.
    echo.
    echo To remove auto-start, delete this file:
    echo %STARTUP_LINK%
) else (
    echo.
    echo ✗ Failed to create startup shortcut.
    echo Please run this script as Administrator.
)

echo.
pause

