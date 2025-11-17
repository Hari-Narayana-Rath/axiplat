@echo off
REM Age Gate Server - Auto-Start Setup
REM This script adds the server to Windows startup

echo ========================================
echo Age Gate Server - Auto-Start Setup
echo ========================================
echo.

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
set "BAT_FILE=%SCRIPT_DIR%start_server_hidden.bat"

REM Check if the launcher exists
if not exist "%BAT_FILE%" (
    echo ERROR: start_server_hidden.bat not found in:
    echo %SCRIPT_DIR%
    echo.
    echo Please make sure you're running this from the project folder.
    pause
    exit /b 1
)

REM Create a VBS script that runs the batch file hidden
set "VBS_FILE=%SCRIPT_DIR%run_server_hidden.vbs"
(
echo Set WshShell = CreateObject^("WScript.Shell"^)
echo WshShell.Run """%BAT_FILE%""", 0, False
) > "%VBS_FILE%"

REM Add to Windows startup folder
set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "STARTUP_LINK=%STARTUP_FOLDER%\Age Gate Server.lnk"

REM Create shortcut in startup folder
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%STARTUP_LINK%'); $s.TargetPath = '%VBS_FILE%'; $s.WorkingDirectory = '%SCRIPT_DIR%'; $s.Save()"

if exist "%STARTUP_LINK%" (
    echo.
    echo ✓ Success! Age Gate Server has been added to Windows startup.
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

