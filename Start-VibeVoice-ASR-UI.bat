@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv-vibevoice-asr\Scripts\python.exe" (
  echo Virtual environment was not found.
  echo Run scripts\setup_vibevoice_asr_hf.ps1 first.
  pause
  exit /b 1
)

set "HF_HOME=%CD%\.hf-cache"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "HF_XET_CACHE=%HF_HOME%\xet"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "PYTHONUNBUFFERED=1"

if not exist "logs" mkdir "logs"
set "LOGFILE=logs\ui-latest.log"

echo Logging stdout/stderr to %LOGFILE%
echo === Run start: %DATE% %TIME% === > "%LOGFILE%"
".venv-vibevoice-asr\Scripts\python.exe" "local_asr\transcribe_ui.py" >> "%LOGFILE%" 2>&1
echo === Exit code: %errorlevel% === >> "%LOGFILE%"
echo.
echo Run finished. See %LOGFILE% for full output.
pause
