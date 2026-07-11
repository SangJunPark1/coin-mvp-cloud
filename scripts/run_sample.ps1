$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$PythonCommand = Get-Command python -ErrorAction SilentlyContinue

if ($PythonCommand) {
    $Python = $PythonCommand.Source
} elseif (Test-Path $BundledPython) {
    $Python = $BundledPython
} else {
    throw "Python was not found. Install Python or run this inside the Codex desktop environment."
}

Set-Location $Root
& $Python -m coin_mvp --config config.example.json --source sample --ticks 80
& $Python -m coin_mvp.report --trades data\trades.csv --events logs\events.jsonl --output reports\latest_report.html
