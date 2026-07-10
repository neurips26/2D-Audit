param(
    [string]$ZipPath = "$HOME\Downloads\AAAI_reviewer_complete_pipeline_v4_20260703.zip",
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"
$Required = @(
    "reviewer_complete_pipeline.py",
    "run_all_reviewer_fixes.ps1",
    "stage3_npo_smoke_complete_fixed.py",
    "diagnose_preflight.ps1"
)

$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$ZipPath = (Resolve-Path $ZipPath).Path

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$InstallRoot = Join-Path $ProjectRoot ".aaai_pipeline_v4_install_$Timestamp"
$ExtractRoot = Join-Path $InstallRoot "extracted"
$BackupRoot = Join-Path $ProjectRoot "outputs\revision\reviewer_complete_pipeline\install_backups\$Timestamp"
$Report = Join-Path $ProjectRoot "outputs\revision\reviewer_complete_pipeline\install_v4_report.txt"

New-Item -ItemType Directory -Force -Path $ExtractRoot | Out-Null
New-Item -ItemType Directory -Force -Path $BackupRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $Report) | Out-Null

$Lines = New-Object System.Collections.Generic.List[string]
function Log-Line([string]$Text) {
    $Lines.Add($Text)
    Write-Host $Text
}

try {
    Log-Line "[INFO] ZIP: $ZipPath"
    Log-Line "[INFO] PROJECT: $ProjectRoot"

    Expand-Archive -Path $ZipPath -DestinationPath $ExtractRoot -Force

    foreach ($Name in $Required) {
        $Matches = @(Get-ChildItem -Path $ExtractRoot -Recurse -File -Filter $Name)
        if ($Matches.Count -ne 1) {
            throw "Expected exactly one '$Name' in ZIP; found $($Matches.Count)."
        }

        $Source = $Matches[0].FullName
        $Target = Join-Path $ProjectRoot $Name

        if (Test-Path $Target) {
            Copy-Item $Target (Join-Path $BackupRoot $Name) -Force
            Log-Line "[BACKUP] $Name"
        }

        Copy-Item $Source $Target -Force
        Log-Line "[INSTALL] $Name"
    }

    $Launcher = Join-Path $ProjectRoot "run_all_reviewer_fixes.ps1"
    $Npo = Join-Path $ProjectRoot "stage3_npo_smoke_complete_fixed.py"
    $Engine = Join-Path $ProjectRoot "reviewer_complete_pipeline.py"

    if (-not (Select-String -Path $Launcher -SimpleMatch "AAAI_REVIEWER_PIPELINE_V4_20260703" -Quiet)) {
        throw "Installed launcher does not contain the V4 version marker."
    }

    if (-not (Test-Path $Npo)) {
        throw "NPO script was not installed."
    }

    Push-Location $ProjectRoot
    try {
        & py -m py_compile $Engine $Npo
        if ($LASTEXITCODE -ne 0) {
            throw "Python compilation failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }

    Log-Line "[PASS] V4 files installed and Python syntax verified."
    Log-Line "OVERALL VERDICT: PASS"
}
catch {
    Log-Line "[FAIL] $($_.Exception.Message)"
    Log-Line "OVERALL VERDICT: FAIL"
}
finally {
    $Lines | Set-Content $Report -Encoding UTF8
    Remove-Item $InstallRoot -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "REPORT: $Report"
}

if ($Lines -contains "OVERALL VERDICT: FAIL") {
    exit 1
}
exit 0
