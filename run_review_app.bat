@echo off
REM Launch the review app, for correcting and signing off a folder that a batch
REM run has already analysed. Any arguments are passed on:
REM
REM   run_review_app.bat
REM   run_review_app.bat --folder "D:\images\C1_analysis"

chcp 65001 > nul 2>&1
cd /d "%~dp0"

call "%~dp0scripts\activate_env.bat"
if errorlevel 1 (
    pause
    exit /b 1
)

echo.
echo Starting the review app at http://127.0.0.1:7870
echo Press Ctrl+C to stop it.
echo.
python -m sem_review_app %*

pause
