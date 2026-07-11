$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Reports = Join-Path $Root "reports"
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
$Port = 8765

if ($PythonCommand) {
    $Python = $PythonCommand.Source
} elseif (Test-Path $BundledPython) {
    $Python = $BundledPython
} else {
    throw "Python was not found. Install Python or run this inside the Codex desktop environment."
}

if (!(Test-Path $Reports)) {
    New-Item -ItemType Directory -Path $Reports | Out-Null
}

$IPv4 = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object {
        $_.IPAddress -notlike "127.*" -and
        $_.IPAddress -notlike "169.254.*" -and
        $_.PrefixOrigin -ne "WellKnown"
    } |
    Select-Object -First 1 -ExpandProperty IPAddress

Set-Location $Reports
Write-Host "Your phone and PC must be on the same Wi-Fi."
Write-Host "Open on PC: http://localhost:$Port/latest_report.html"
if ($IPv4) {
    Write-Host "Open on phone: http://$IPv4`:$Port/latest_report.html"
} else {
    Write-Host "Could not detect a phone URL automatically. Check your Windows network IP."
}
Write-Host "Press Ctrl+C in this terminal to stop the server."
& $Python -m http.server $Port --bind 0.0.0.0
