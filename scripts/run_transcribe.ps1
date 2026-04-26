param(
    [Parameter(Mandatory = $true)]
    [string[]]$Audio,

    [string]$Model = "microsoft/VibeVoice-ASR-HF",
    [string]$OutputDir = "transcripts",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [ValidateSet("auto", "float32", "float16", "bfloat16")]
    [string]$Dtype = "auto",
    [string]$Prompt = "",
    [string]$PromptFile = "",
    [int]$MaxNewTokens = 32768,
    [int]$TokenizerChunkSize = 0,
    [switch]$Offline,
    [switch]$WriteSrt,
    [string]$VenvPath = ".venv-vibevoice-asr"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $RepoRoot "$VenvPath\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Virtual environment not found. Run .\scripts\setup_vibevoice_asr_hf.ps1 first."
}

$runner = Join-Path $RepoRoot "local_asr\transcribe_vibevoice_hf.py"
$argsList = @(
    $runner
) + $Audio + @(
    "--model", $Model,
    "--output-dir", $OutputDir,
    "--device", $Device,
    "--dtype", $Dtype,
    "--max-new-tokens", [string]$MaxNewTokens
)

if ($Prompt) {
    $argsList += @("--prompt", $Prompt)
}
if ($PromptFile) {
    $argsList += @("--prompt-file", $PromptFile)
}
if ($TokenizerChunkSize -gt 0) {
    $argsList += @("--tokenizer-chunk-size", [string]$TokenizerChunkSize)
}
if ($Offline) {
    $argsList += "--offline"
}
if ($WriteSrt) {
    $argsList += "--write-srt"
}

& $VenvPython @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Transcription failed."
}
