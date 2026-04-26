@echo off
setlocal
cd /d "%~dp0"

echo Setting up VibeVoice ASR local environment...
echo This may take a long time on the first run.
echo.

powershell -ExecutionPolicy Bypass -File "scripts\setup_vibevoice_asr_hf.ps1" -Torch auto
if errorlevel 1 (
  echo.
  echo Setup failed.
  pause
  exit /b 1
)

echo.
echo Setup complete.
echo You can now double-click Start-VibeVoice-ASR-UI.bat.
pause
