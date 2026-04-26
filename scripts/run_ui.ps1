param(
    [string]$VenvPath = ".venv-vibevoice-asr"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $RepoRoot "$VenvPath\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Virtual environment not found. Run scripts\setup_vibevoice_asr_hf.ps1 first."
}

$env:HF_HOME = Join-Path $RepoRoot ".hf-cache"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:HF_XET_CACHE = Join-Path $env:HF_HOME "xet"

$ui = Join-Path $RepoRoot "local_asr\transcribe_ui.py"
& $VenvPython $ui
if ($LASTEXITCODE -ne 0) {
    throw "UI exited with an error."
}
