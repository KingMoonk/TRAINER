@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "import tkinter, numpy, pandas, panda3d">nul 2>&1
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto failed
)
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0app.py"
exit /b 0
:failed
echo Setup failed. See the error above.
pause
exit /b 1
