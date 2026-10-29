@echo off
setlocal
cd /d "%~dp0"
REM Creates .venv with Pillow only (the server is stdlib). Python 3.10+.
py -3 -m venv .venv || (echo Python 3 launcher not found & pause & exit /b 1)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if not exist config.json copy config-sample.json config.json
echo Done. Edit config.json (engine folders and ports) or use Settings in the app, then run.bat
pause
