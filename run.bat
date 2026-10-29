@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
REM crispz-dataset-studio: stdlib server + Pillow. Any Python 3.10+ with Pillow works;
REM the crispz family venvs all have it, so we reuse one when present (no install).
set RUNPY=
if exist ".venv\Scripts\python.exe" set RUNPY=.venv\Scripts\python.exe
if "!RUNPY!"=="" if exist "..\crispz-klein\.venv\Scripts\python.exe" set RUNPY=..\crispz-klein\.venv\Scripts\python.exe
if "!RUNPY!"=="" if exist "..\crispz-flux3\.venv\Scripts\python.exe" set RUNPY=..\crispz-flux3\.venv\Scripts\python.exe
if "!RUNPY!"=="" set RUNPY=py -3
!RUNPY! -c "import PIL" >nul 2>&1
if errorlevel 1 (
  echo Pillow is missing for !RUNPY! : run install.bat
  pause
  exit /b 1
)
echo Python = !RUNPY!
!RUNPY! ds_server.py %*
endlocal
