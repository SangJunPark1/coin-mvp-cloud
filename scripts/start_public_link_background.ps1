$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
[Environment]::SetEnvironmentVariable("PATH", $null, "Process")

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Reports = Join-Path $Root "reports"
$Logs = Join-Path $Root "logs"
$Port = 8765
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
$CloudflaredCommand = Get-Command cloudflared -ErrorAction SilentlyContinue
$WingetCloudflared = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe"
$ServerPidFile = Join-Path $Logs "report_server.pid"
$TunnelPidFile = Join-Path $Logs "cloudflared.pid"
$TunnelOutFile = Join-Path $Logs "cloudflared.out.log"
$TunnelErrFile = Join-Path $Logs "cloudflared.err.log"
$PublicUrlFile = Join-Path $Logs "public_report_url.txt"

if ($PythonCommand) {
    $Python = $PythonCommand.Source
} elseif (Test-Path $BundledPython) {
    $Python = $BundledPython
} else {
    throw "Python was not found. Install Python or run this inside the Codex desktop environment."
}

if ($CloudflaredCommand) {
    $Cloudflared = $CloudflaredCommand.Source
} elseif (Test-Path $WingetCloudflared) {
    $Cloudflared = $WingetCloudflared
} else {
    throw "cloudflared was not found."
}

New-Item -ItemType Directory -Path $Reports -Force | Out-Null
New-Item -ItemType Directory -Path $Logs -Force | Out-Null

Set-Location $Root
& $Python -m coin_mvp.report --trades data\trades.csv --events logs\events.jsonl --output reports\latest_report.html

$ExistingServer = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if (!$ExistingServer) {
    $ServerProcess = Start-Process `
        -FilePath $Python `
        -ArgumentList @("-m", "http.server", "$Port", "--bind", "127.0.0.1") `
        -WorkingDirectory $Reports `
        -PassThru `
        -WindowStyle Hidden
    $ServerProcess.Id | Set-Content $ServerPidFile
    Start-Sleep -Seconds 2
}

if (Test-Path $TunnelPidFile) {
    $ExistingTunnelPid = Get-Content $TunnelPidFile -ErrorAction SilentlyContinue
    if ($ExistingTunnelPid -and (Get-Process -Id $ExistingTunnelPid -ErrorAction SilentlyContinue)) {
        Write-Host "Cloudflare tunnel already running. PID: $ExistingTunnelPid"
        if (Test-Path $PublicUrlFile) {
            Get-Content $PublicUrlFile
        }
        exit 0
    }
}

if (Test-Path $TunnelOutFile) { Remove-Item $TunnelOutFile -Force }
if (Test-Path $TunnelErrFile) { Remove-Item $TunnelErrFile -Force }
if (Test-Path $PublicUrlFile) { Remove-Item $PublicUrlFile -Force }

$TunnelProcess = Start-Process `
    -FilePath $Cloudflared `
    -ArgumentList @("tunnel", "--url", "http://localhost:$Port", "--no-autoupdate") `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $TunnelOutFile `
    -RedirectStandardError $TunnelErrFile `
    -PassThru `
    -WindowStyle Hidden

$TunnelProcess.Id | Set-Content $TunnelPidFile

$Url = $null
for ($i = 0; $i -lt 45; $i++) {
    Start-Sleep -Seconds 1
    $Text = ""
    if (Test-Path $TunnelOutFile) { $Text += Get-Content $TunnelOutFile -Raw -ErrorAction SilentlyContinue }
    if (Test-Path $TunnelErrFile) { $Text += Get-Content $TunnelErrFile -Raw -ErrorAction SilentlyContinue }
    $Match = [regex]::Match($Text, "https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    if ($Match.Success) {
        $Url = $Match.Value
        break
    }
}

if (!$Url) {
    Write-Host "Tunnel started, but no public URL was detected yet."
    Write-Host "Check logs\cloudflared.err.log"
    exit 1
}

$ReportUrl = "$Url/latest_report.html"
$ReportUrl | Set-Content $PublicUrlFile
Write-Host $ReportUrl
