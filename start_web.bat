@echo off
setlocal
cd /d "%~dp0"
for %%I in ("%~dp0.") do set "WORKSPACE=%%~fI"
if exist ".venv\Scripts\python.exe" goto python_ready
for /f "delims=" %%I in ('git rev-parse --path-format^=absolute --git-common-dir 2^>nul') do set "GIT_COMMON=%%I"
if not defined GIT_COMMON goto missing_python
for %%I in ("%GIT_COMMON%\..") do set "WORKSPACE=%%~fI"
:python_ready
set "PYTHON=%WORKSPACE%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" goto missing_python
if not exist "web\out\index.html" (
    echo Build the web UI first: python tasks.py web:build
    pause
    exit /b 1
)
echo Groundwork Studio: http://127.0.0.1:8787/chat/
echo Workspace: %WORKSPACE%
echo Keep this window open while using the app. Ctrl+C stops the server.
"%PYTHON%" -m studio_api --workspace "%WORKSPACE%" --port 8787
if errorlevel 1 pause
exit /b %errorlevel%
:missing_python
echo Python environment not found. Install the project requirements into .venv first.
pause
exit /b 1
