@echo off
setlocal EnableDelayedExpansion
REM crispz-dataset-studio: send an image to Dataset Studio as a new project's reference.
REM For IrfanView's external editor (Shift+E): point it at this file with "%1".
REM Same Python choice as run.bat.
set APPDIR=%~dp0
set RUNPY=
if exist "%APPDIR%.venv\Scripts\python.exe" set RUNPY="%APPDIR%.venv\Scripts\python.exe"
if "!RUNPY!"=="" if exist "%APPDIR%..\crispz-klein\.venv\Scripts\python.exe" set RUNPY="%APPDIR%..\crispz-klein\.venv\Scripts\python.exe"
if "!RUNPY!"=="" if exist "%APPDIR%..\crispz-flux3\.venv\Scripts\python.exe" set RUNPY="%APPDIR%..\crispz-flux3\.venv\Scripts\python.exe"
if "!RUNPY!"=="" set RUNPY=py -3
!RUNPY! "%APPDIR%tools\send_to_studio.py" %*
if errorlevel 1 pause
endlocal
