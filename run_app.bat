@echo off
REM Launch Konimeter. Double-click it, or run it from a terminal; any
REM arguments are passed on (for example --port 7861).

chcp 65001 > nul 2>&1
REM Run from the repo root whatever directory this was launched from, so
REM "python -m sem_analysis_app" can find the package.
cd /d "%~dp0"

call "%~dp0scripts\activate_env.bat"
if errorlevel 1 (
    pause
    exit /b 1
)

echo.
echo Starting Konimeter. Open the address it prints below
echo ^(http://127.0.0.1:7860 unless you passed --port^). Press Ctrl+C to stop it.
echo.
python -m sem_analysis_app %*

pause
