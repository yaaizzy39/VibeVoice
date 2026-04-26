param(
    [ValidateSet("auto", "cpu", "cu126", "cu128", "cu130", "none")]
    [string]$Torch = "auto",

    [string]$VenvPath = ".venv-vibevoice-asr"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$VenvFullPath = Join-Path $RepoRoot $VenvPath

function Find-Python {
    $candidates = @(
        @{ Exe = "py"; Args = @("-3.11") },
        @{ Exe = "py"; Args = @("-3.10") },
        @{ Exe = "python"; Args = @() }
    )

    foreach ($candidate in $candidates) {
        try {
            & $candidate.Exe @($candidate.Args) --version | Out-Null
            if ($LASTEXITCODE -eq 0) {
                return $candidate
            }
        } catch {
        }
    }
    throw "Python 3.10 or newer was not found. Install Python first."
}

function Invoke-Pip {
    param([string[]]$Arguments)
    & $script:VenvPython -m pip @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "pip failed: $($Arguments -join ' ')"
    }
}

$python = Find-Python
Write-Host "Using Python launcher: $($python.Exe) $($python.Args -join ' ')"

if (-not (Test-Path -LiteralPath $VenvFullPath)) {
    Write-Host "Creating virtual environment: $VenvFullPath"
    & $python.Exe @($python.Args) -m venv $VenvFullPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment."
    }
}

$script:VenvPython = Join-Path $VenvFullPath "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $script:VenvPython)) {
    throw "Virtual environment Python was not found: $script:VenvPython"
}

Invoke-Pip @("install", "--upgrade", "pip", "setuptools", "wheel")

if ($Torch -eq "auto") {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $Torch = "cu128"
    } else {
        $Torch = "cpu"
    }
    Write-Host "Auto-selected torch build: $Torch"
}

if ($Torch -ne "none") {
    $torchIndex = switch ($Torch) {
        "cpu" { "https://download.pytorch.org/whl/cpu" }
        "cu126" { "https://download.pytorch.org/whl/cu126" }
        "cu128" { "https://download.pytorch.org/whl/cu128" }
        "cu130" { "https://download.pytorch.org/whl/cu130" }
    }
    Invoke-Pip @("install", "torch", "torchvision", "torchaudio", "--index-url", $torchIndex)
}

Invoke-Pip @(
    "install",
    "transformers>=5.3.0",
    "accelerate",
    "huggingface_hub",
    "safetensors",
    "soundfile",
    "librosa",
    "av",
    "numpy"
)

Write-Host ""
Write-Host "Setup complete."
Write-Host "Run transcription with:"
Write-Host "  .\scripts\run_transcribe.ps1 -Audio `"C:\path\audio.wav`""
