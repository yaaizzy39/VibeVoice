@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv-vibevoice-asr\Scripts\python.exe" (
  echo Virtual environment was not found.
  echo Run scripts\setup_vibevoice_asr_hf.ps1 first.
  pause
  exit /b 1
)

if not defined HF_HOME set "HF_HOME=D:\models\VibeVoice\hf-cache"
if not defined HF_HUB_CACHE set "HF_HUB_CACHE=%HF_HOME%\hub"
if not defined HF_XET_CACHE set "HF_XET_CACHE=%HF_HOME%\xet"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "PYTHONIOENCODING=utf-8"
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
