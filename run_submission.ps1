[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSUseApprovedVerbs', '', Justification='No custom functions are declared in this script.')]
param()

$ErrorActionPreference = "Stop"

$envPath = ".env"
if (-not (Test-Path $envPath)) {
    throw "Missing $envPath file in project root."
}

Get-Content $envPath | ForEach-Object {
    $line = $_.Trim()
    if ([string]::IsNullOrWhiteSpace($line)) { return }
    if ($line.StartsWith("#")) { return }

    $parts = $line -split '=', 2
    if ($parts.Count -ne 2) { return }

    $name = $parts[0].Trim()
    $value = $parts[1].Trim()
    [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
}

if (-not $env:HF_TOKEN) {
    throw "HF_TOKEN is missing. Add it to .env"
}
if (-not $env:MODEL_NAME) {
    throw "MODEL_NAME is missing. Add it to .env"
}
if (-not $env:API_BASE_URL) {
    throw "API_BASE_URL is missing. Add it to .env"
}

# Compatibility for any old script that reads OPENAI_API_KEY.
if (-not $env:OPENAI_API_KEY) {
    $env:OPENAI_API_KEY = $env:HF_TOKEN
}

$python = "d:/Meta/.venv/Scripts/python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

Write-Host "[1/3] Syntax check..."
& $python -m py_compile inference.py baseline/run_baseline.py meta_ads_env/env.py

Write-Host "[2/3] Running required inference.py..."
& $python inference.py

Write-Host "[3/3] Done."
