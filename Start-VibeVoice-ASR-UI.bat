@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv-vibevoice-asr\Scripts\python.exe" (
  echo Virtual environment was not found.
  echo Run scripts\setup_vibevoice_asr_hf.ps1 first.
  pause
  exit /b 1
)

".venv-vibevoice-asr\Scripts\python.exe" "local_asr\transcribe_ui.py"
if errorlevel 1 pause
