@echo off
REM Start the segment-and-classify demo (Windows).
REM   1. finds conda and activates the cnt-vfm environment
REM   2. starts the page at http://127.0.0.1:7861; the first time, the page
REM      offers to download the SAM ViT-B checkpoint (358 MB)
setlocal
cd /d "%~dp0"
if not "%CONDA_DEFAULT_ENV%"=="cnt-vfm" (
  call "%~dp0..\scripts\activate_env.bat" || (pause & exit /b 1)
)
python demo_app.py %*
pause
