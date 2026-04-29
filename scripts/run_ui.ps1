param(
    [string]$VenvPath = ".venv-vibevoice-asr"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $RepoRoot "$VenvPath\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Virtual environment not found. Run scripts\setup_vibevoice_asr_hf.ps1 first."
}

if (-not $env:HF_HOME) {
    $env:HF_HOME = "D:\models\VibeVoice\hf-cache"
}
if (-not $env:HF_HUB_CACHE) {
    $env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
}
if (-not $env:HF_XET_CACHE) {
    $env:HF_XET_CACHE = Join-Path $env:HF_HOME "xet"
}
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$ui = Join-Path $RepoRoot "local_asr\transcribe_ui.py"
& $VenvPython $ui
if ($LASTEXITCODE -ne 0) {
    throw "UI exited with an error."
}
