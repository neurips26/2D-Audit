param()
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$Paths = @(
    ".\outputs\revision\reviewer_complete_pipeline\launcher\logs\01_preflight.log",
    ".\outputs\revision\reviewer_complete_pipeline\preflight_report.txt",
    ".\outputs\revision\reviewer_complete_pipeline\preflight_report.json",
    ".\outputs\revision\reviewer_complete_pipeline\fatal_error.json"
)

Write-Host ("=" * 90) -ForegroundColor Cyan
Write-Host "PREFLIGHT DIAGNOSTIC" -ForegroundColor Cyan
Write-Host ("=" * 90) -ForegroundColor Cyan

foreach ($Path in $Paths) {
    Write-Host ""
    if (Test-Path $Path) {
        Write-Host "[FOUND] $Path" -ForegroundColor Green
        Get-Content $Path
    }
    else {
        Write-Host "[MISSING] $Path" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Running direct preflight with visible output..." -ForegroundColor Cyan
py -u .\reviewer_complete_pipeline.py preflight
$Code = $LASTEXITCODE

if ($Code -eq 0) {
    Write-Host "OVERALL VERDICT: PASS" -ForegroundColor Green
}
else {
    Write-Host "OVERALL VERDICT: FAIL (exit $Code)" -ForegroundColor Red
}
exit $Code
