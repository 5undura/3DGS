@echo off
rem Double-click: run the whole pipeline (extract -> register -> train) with config.ps1
cd /d "%~dp0"
if not exist config.ps1 (
  echo [!] config.ps1 not found. Copy config.example.ps1 to config.ps1 and edit the paths.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1"
pause
