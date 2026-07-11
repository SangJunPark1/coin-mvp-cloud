$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
[Environment]::SetEnvironmentVariable("PATH", $null, "Process")

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
$PidFile = Join-Path $Root "logs\watch_3day.pid"
$OutFile = Join-Path $Root "logs\watch_3day.out.log"
$ErrFile = Join-Path $Root "logs\watch_3day.err.log"

if ($PythonCommand) {
    $Python = $PythonCommand.Source
} elseif (Test-Path $BundledPython) {
    $Python = $BundledPython
} else {
    throw "Python was not found. Install Python or run this inside the Codex desktop environment."
}

New-Item -ItemType Directory -Path (Join-Path $Root "logs") -Force | Out-Null

if (Test-Path $PidFile) {
    $ExistingPid = Get-Content $PidFile -ErrorAction SilentlyContinue
    if ($ExistingPid -and (Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue)) {
        Write-Host "3-day monitor is already running. PID: $ExistingPid"
        exit 0
    }
}

$Args = @(
    "-m", "coin_mvp.watch_multi",
    "--config", "config.lowload.json",
    "--top-markets", "30",
    "--ticks", "864",
    "--report-every", "1",
    "--output", "reports\latest_report.html",
    "--request-delay", "0.18"
)

$Process = Start-Process `
    -FilePath $Python `
    -ArgumentList $Args `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $OutFile `
    -RedirectStandardError $ErrFile `
    -PassThru `
    -WindowStyle Hidden

$Process.Id | Set-Content $PidFile
Write-Host "3-day low-load monitor started. PID: $($Process.Id)"
Write-Host "Polling: every 5 minutes. Duration: about 3 days."
Write-Host "Report: reports\latest_report.html"
