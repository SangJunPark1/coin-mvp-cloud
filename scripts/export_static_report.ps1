$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
$ExportDir = Join-Path $Root "public_report"
$IndexPath = Join-Path $ExportDir "index.html"
$ReadmePath = Join-Path $ExportDir "README.txt"

if ($PythonCommand) {
    $Python = $PythonCommand.Source
} elseif (Test-Path $BundledPython) {
    $Python = $BundledPython
} else {
    throw "Python was not found. Install Python or run this inside the Codex desktop environment."
}

if (!(Test-Path $ExportDir)) {
    New-Item -ItemType Directory -Path $ExportDir | Out-Null
}

Set-Location $Root
& $Python -m coin_mvp.report --trades data\trades.csv --events logs\events.jsonl --output $IndexPath

$Guide = @"
코인 MVP 정적 리포트

이 폴더의 index.html은 서버 없이 열리는 단일 HTML 리포트입니다.
컴퓨터를 꺼도 휴대폰에서 보려면 이 public_report 폴더 또는 index.html 파일을 외부 정적 호스팅에 올리면 됩니다.

추천 순서:
1. GitHub Pages: 저장소에 public_report/index.html을 올리고 Pages를 켭니다.
2. Netlify Drop: https://app.netlify.com/drop 에 public_report 폴더를 끌어다 놓습니다.
3. Google Drive/OneDrive: index.html을 업로드하고 공유 링크로 보관합니다.

주의:
- 리포트는 업로드 시점의 스냅샷입니다.
- 새 거래 결과를 반영하려면 이 스크립트를 다시 실행한 뒤 다시 업로드해야 합니다.
- API 키나 개인정보를 리포트에 넣지 마세요.
"@

Set-Content -Path $ReadmePath -Value $Guide -Encoding UTF8

Write-Host "정적 리포트 생성 완료:"
Write-Host "  $IndexPath"
Write-Host ""
Write-Host "PC를 꺼도 보려면 public_report 폴더를 GitHub Pages, Netlify Drop, Google Drive 같은 외부 저장소에 올리세요."
