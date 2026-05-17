@echo off
setlocal

echo ============================================================
echo   Brain Tumor Detection - Automated Setup and Start
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.9-3.11 from https://www.python.org/
    pause
    exit /b 1
)

REM Set project directories
set "ROOT_DIR=%~dp0"
set "SRC_DIR=%ROOT_DIR%src"
set "VENV_DIR=%SRC_DIR%\venv"

cd /d "%SRC_DIR%"

REM Create virtual environment if it doesn't exist
if not exist "%VENV_DIR%" (
    echo [INFO] Creating virtual environment in %VENV_DIR%...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo Try moving the project to a shorter path (e.g. C:\BTD)
        pause
        exit /b 1
    )
)

REM Install dependencies
echo [INFO] Ensuring dependencies are installed...
echo [INFO] This might take a few minutes for the first run...
"%VENV_DIR%\Scripts\pip" install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    echo This is often due to Windows Long Path limits.
    echo Please refer to RUN_GUIDE.md for troubleshooting.
    pause
    exit /b 1
)

REM Run the app
echo.
echo [OK] Everything is ready!
echo [INFO] Starting Flask server on http://localhost:5000
echo.
"%VENV_DIR%\Scripts\python" app.py

pause
