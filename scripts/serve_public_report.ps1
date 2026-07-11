$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Reports = Join-Path $Root "reports"
$Port = 8765
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
$CloudflaredCommand = Get-Command cloudflared -ErrorAction SilentlyContinue
$WingetCloudflared = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe"

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
    Write-Host "cloudflared was not found."
    Write-Host "Install it with:"
    Write-Host "  winget install --id Cloudflare.cloudflared"
    Write-Host "Then run this VSCode task again."
    exit 1
}

if (!(Test-Path $Reports)) {
    New-Item -ItemType Directory -Path $Reports | Out-Null
}

Set-Location $Root
& $Python -m coin_mvp.report --trades data\trades.csv --events logs\events.jsonl --output reports\latest_report.html

$ExistingServer = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
$ServerProcess = $null

try {
    if ($ExistingServer) {
        Write-Host "Local report server already appears to be running on port $Port."
    } else {
        Write-Host "Starting local report server on port $Port..."
        $ServerProcess = Start-Process `
            -FilePath $Python `
            -ArgumentList @("-m", "http.server", "$Port", "--bind", "127.0.0.1") `
            -WorkingDirectory $Reports `
            -PassThru `
            -WindowStyle Hidden
        Start-Sleep -Seconds 2
    }

    Write-Host ""
    Write-Host "Cloudflare Tunnel is starting."
    Write-Host "Copy the https://*.trycloudflare.com URL shown below."
    Write-Host "Open this on your phone:"
    Write-Host "  https://YOUR-TUNNEL-URL/latest_report.html"
    Write-Host ""
    Write-Host "Keep this terminal open while you want the phone link to work."
    Write-Host "Press Ctrl+C to stop the public link."
    Write-Host ""

    & $Cloudflared tunnel --url "http://localhost:$Port"
} finally {
    if ($ServerProcess -and !$ServerProcess.HasExited) {
        Stop-Process -Id $ServerProcess.Id -Force
    }
}
