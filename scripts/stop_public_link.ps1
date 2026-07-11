$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Logs = Join-Path $Root "logs"
$PidFiles = @(
    (Join-Path $Logs "cloudflared.pid"),
    (Join-Path $Logs "report_server.pid")
)

foreach ($PidFile in $PidFiles) {
    if (Test-Path $PidFile) {
        $PidValue = Get-Content $PidFile -ErrorAction SilentlyContinue
        if ($PidValue -and (Get-Process -Id $PidValue -ErrorAction SilentlyContinue)) {
            Stop-Process -Id $PidValue -Force
            Write-Host "Stopped PID: $PidValue"
        }
        Remove-Item $PidFile -Force
    }
}
