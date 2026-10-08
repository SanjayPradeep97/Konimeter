@echo off
REM Activate the conda environment both launchers run in.
REM
REM The environment is SEM_analysis, the one the README sets up. Set SEM_ENV
REM before running to use a different one. If SEM_analysis does not exist but
REM cnt-imaging does, that is used instead.
REM
REM Conda is found wherever it was installed: through CONDA_EXE when started
REM from an Anaconda Prompt, otherwise in the usual install folders.

if not defined SEM_ENV set "SEM_ENV=SEM_analysis"

set "CONDA_ROOT="
if defined CONDA_EXE for %%I in ("%CONDA_EXE%\..\..") do set "CONDA_ROOT=%%~fI"
for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do (
    if not defined CONDA_ROOT if exist "%%~D\Scripts\activate.bat" set "CONDA_ROOT=%%~D"
)
if not defined CONDA_ROOT (
    echo ERROR: could not find a conda installation.
    echo Install Miniconda or Anaconda, or run this from an Anaconda Prompt.
    exit /b 1
)

set "ENV_DIR="
for %%E in ("%CONDA_ROOT%\envs\%SEM_ENV%" "%USERPROFILE%\.conda\envs\%SEM_ENV%") do (
    if not defined ENV_DIR if exist "%%~E\python.exe" set "ENV_DIR=%%~E"
)
if not defined ENV_DIR if "%SEM_ENV%"=="SEM_analysis" (
    if exist "%CONDA_ROOT%\envs\cnt-imaging\python.exe" set "ENV_DIR=%CONDA_ROOT%\envs\cnt-imaging"
)
if not defined ENV_DIR (
    echo ERROR: conda environment "%SEM_ENV%" not found under %CONDA_ROOT%\envs
    echo Create it as the README describes, or set SEM_ENV to the one you use.
    exit /b 1
)

echo Using %ENV_DIR%
call "%CONDA_ROOT%\Scripts\activate.bat" "%ENV_DIR%"
REM Activation can fail without saying so - two launchers started at the same
REM moment collide on conda's temporary file - and the app then dies further on
REM with "Python was not found". So check it took, and try once more if not.
if /i not "%CONDA_PREFIX%"=="%ENV_DIR%" (
    ping -n 3 127.0.0.1 > nul
    call "%CONDA_ROOT%\Scripts\activate.bat" "%ENV_DIR%"
)
if /i not "%CONDA_PREFIX%"=="%ENV_DIR%" (
    echo ERROR: could not activate %ENV_DIR%
    echo Try again, or activate it yourself in an Anaconda Prompt and run
    echo "python -m sem_analysis_app" or "python -m sem_review_app" from here.
    exit /b 1
)

REM Status lines use characters a cp1252 console cannot print.
set PYTHONIOENCODING=utf-8
REM PyTorch and NumPy each ship an OpenMP runtime; without this they refuse to
REM load into the same process on Windows.
set KMP_DUPLICATE_LIB_OK=TRUE
exit /b 0
