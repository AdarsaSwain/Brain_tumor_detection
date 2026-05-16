@echo off
echo ============================================================
echo   Brain Tumor Detection - Start Server
echo ============================================================
echo.

REM Activate the virtual environment at C:\btd_venv
call C:\btd_venv\Scripts\activate.bat

REM Change to the src directory
cd /d "%~dp0"

REM Step 1: Build models if they are still Git-LFS stubs (< 1 MB)
for %%F in (epoch10_sgd_acc96Point76.h5) do (
    if %%~zF LSS 1000000 (
        echo [INFO] Model files not downloaded. Building from scratch...
        echo [INFO] This downloads ImageNet weights and may take 5-10 minutes...
        python build_models.py
        if errorlevel 1 (
            echo [ERROR] Model build failed. See error above.
            pause
            exit /b 1
        )
    )
)

echo [INFO] Starting Flask server on http://localhost:5000
python app.py
pause
