param(
    [string]$ProjectRoot = (Get-Location).Path
)

$ErrorActionPreference = "Stop"

function Header([string]$Text) {
    Write-Host "`n================================================================================" -ForegroundColor DarkCyan
    Write-Host $Text -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor DarkCyan
}

function Status([string]$Level, [string]$Text) {
    $Color = switch ($Level) {
        "PASS" { "Green" }
        "WARN" { "Yellow" }
        "FAIL" { "Red" }
        "INFO" { "Cyan" }
        default { "White" }
    }
    Write-Host "[$Level] $Text" -ForegroundColor $Color
}

function Copy-FileRobust {
    param(
        [Parameter(Mandatory=$true)][string]$Source,
        [Parameter(Mandatory=$true)][string]$DestinationRoot,
        [Parameter(Mandatory=$true)][string]$RelativePath,
        [Parameter(Mandatory=$true)][System.Collections.ArrayList]$LongPathMap
    )

    $Target = Join-Path $DestinationRoot $RelativePath
    $TargetDir = Split-Path $Target -Parent

    try {
        New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
        Copy-Item -LiteralPath $Source -Destination $Target -Force
        return
    }
    catch {
        $Ext = [IO.Path]::GetExtension($Source)
        $Base = [IO.Path]::GetFileNameWithoutExtension($Source)
        $HashInput = [Text.Encoding]::UTF8.GetBytes($RelativePath)
        $Sha = [Security.Cryptography.SHA256]::Create()
        $Hash = ([BitConverter]::ToString($Sha.ComputeHash($HashInput))).Replace("-", "").Substring(0,16)
        $SafeBase = ($Base -replace '[^A-Za-z0-9._-]', '_')
        if ($SafeBase.Length -gt 70) { $SafeBase = $SafeBase.Substring(0,70) }
        $FlatDir = Join-Path $DestinationRoot "_flattened_long_paths"
        New-Item -ItemType Directory -Force -Path $FlatDir | Out-Null
        $FlatName = "${Hash}_${SafeBase}${Ext}"
        $FlatTarget = Join-Path $FlatDir $FlatName
        Copy-Item -LiteralPath $Source -Destination $FlatTarget -Force

        [void]$LongPathMap.Add([pscustomobject]@{
            OriginalRelativePath = $RelativePath
            FlattenedPath        = "_flattened_long_paths\$FlatName"
        })
    }
}

$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$TransferRoot = Join-Path $ProjectRoot "outputs\revision\home_transfer"
$BundleName = "TwoDimAudit_home_transfer_$Timestamp"
$BundleDir = Join-Path $TransferRoot $BundleName
$ZipPath = Join-Path $TransferRoot "$BundleName.zip"

New-Item -ItemType Directory -Force -Path $BundleDir | Out-Null

Header "TwoDimAudit home-transfer packager V3"
Write-Host "[config] ProjectRoot = $ProjectRoot"
Write-Host "[config] BundleDir   = $BundleDir"
Write-Host "[config] ZipPath     = $ZipPath"

$PipelineRoot = Join-Path $ProjectRoot "outputs\revision\reviewer_complete_pipeline"
$ManuscriptResults = Join-Path $PipelineRoot "manuscript_results"

Header "1. Verify required final outputs"

$Required = @(
    "final_report.json",
    "final_report.txt",
    "authoritative_bootstrap_intervals.csv",
    "authoritative_bootstrap_intervals.json",
    "leave_one_entity_out.csv",
    "mmunlearner_most_influential_entity.json",
    "influential_entity_excluded_pairwise.csv",
    "linear_probe_l31.csv",
    "hyperparameter_inventory.csv",
    "manuscript_audit.json",
    "sweep_crp_manifest.json",
    "blip2_logprob_results.json",
    "blip2_logprob_summary.csv"
)

$Missing = @()
foreach ($Name in $Required) {
    $P = Join-Path $ManuscriptResults $Name
    if (Test-Path -LiteralPath $P) {
        Status "PASS" "Required result: $Name"
    } else {
        $Missing += $Name
        Status "FAIL" "Missing required result: $Name"
    }
}

if ($Missing.Count -gt 0) {
    Status "FAIL" "$($Missing.Count) required result file(s) missing. Nothing was zipped."
    exit 1
}

Header "2. Copy verified results"

$LongPathMap = New-Object System.Collections.ArrayList
$ResultsDest = Join-Path $BundleDir "results"
New-Item -ItemType Directory -Force -Path $ResultsDest | Out-Null

# Copy manuscript_results flat and authoritative
Get-ChildItem -LiteralPath $ManuscriptResults -File | ForEach-Object {
    Copy-FileRobust -Source $_.FullName -DestinationRoot $ResultsDest `
        -RelativePath $_.Name -LongPathMap $LongPathMap
}
Status "PASS" "Copied manuscript_results"

# Copy result trees, excluding activation tensors and other huge binaries
$Trees = @(
    @{Name="cpu_analysis";  Path=(Join-Path $PipelineRoot "cpu_analysis")},
    @{Name="sweep_crp";     Path=(Join-Path $PipelineRoot "sweep_crp")},
    @{Name="blip2_logprob"; Path=(Join-Path $PipelineRoot "blip2_logprob")}
)

$AllowedResultExt = @(".csv", ".json", ".txt", ".md", ".tex", ".png", ".pdf")
foreach ($Tree in $Trees) {
    if (-not (Test-Path -LiteralPath $Tree.Path)) {
        Status "WARN" "Missing optional result tree: $($Tree.Name)"
        continue
    }

    $Count = 0
    Get-ChildItem -LiteralPath $Tree.Path -File -Recurse | Where-Object {
        $_.Extension.ToLowerInvariant() -in $AllowedResultExt
    } | ForEach-Object {
        $RelWithinTree = $_.FullName.Substring($Tree.Path.Length).TrimStart("\")
        $Rel = Join-Path $Tree.Name $RelWithinTree
        Copy-FileRobust -Source $_.FullName -DestinationRoot $ResultsDest `
            -RelativePath $Rel -LongPathMap $LongPathMap
        $Count++
    }
    Status "PASS" "Copied $Count files from $($Tree.Name)"
}

Header "3. Collect manuscript source only"

$ManuscriptDest = Join-Path $BundleDir "manuscript"
New-Item -ItemType Directory -Force -Path $ManuscriptDest | Out-Null

# Important: exclude outputs, checkpoints, environments, Git internals, and this transfer folder
$ExcludedFragments = @(
    "\outputs\",
    "\checkpoints\",
    "\.git\",
    "\.venv\",
    "\venv\",
    "\__pycache__\",
    "\site-packages\"
)

$ManuscriptExtensions = @(".tex", ".bib", ".cls", ".sty", ".bst", ".pdf", ".docx")
$ManuscriptFiles = Get-ChildItem -LiteralPath $ProjectRoot -File -Recurse -ErrorAction SilentlyContinue | Where-Object {
    $Full = $_.FullName
    $Excluded = $false
    foreach ($Fragment in $ExcludedFragments) {
        if ($Full -like "*$Fragment*") { $Excluded = $true; break }
    }
    (-not $Excluded) -and ($_.Extension.ToLowerInvariant() -in $ManuscriptExtensions)
}

$ManuscriptMap = New-Object System.Collections.ArrayList
$ManuscriptCount = 0
foreach ($File in $ManuscriptFiles) {
    $Rel = $File.FullName.Substring($ProjectRoot.Length).TrimStart("\")
    Copy-FileRobust -Source $File.FullName -DestinationRoot $ManuscriptDest `
        -RelativePath $Rel -LongPathMap $ManuscriptMap
    $ManuscriptCount++
}

if ($ManuscriptCount -gt 0) {
    Status "PASS" "Copied $ManuscriptCount manuscript/source files"
} else {
    Status "WARN" "No manuscript files found outside outputs/checkpoints"
}

Header "4. Collect essential root-level code and configuration"

$CodeDest = Join-Path $BundleDir "code"
New-Item -ItemType Directory -Force -Path $CodeDest | Out-Null

$RootCode = Get-ChildItem -LiteralPath $ProjectRoot -File | Where-Object {
    $_.Extension.ToLowerInvariant() -in @(".py", ".ps1", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".txt", ".md") -or
    $_.Name -like "requirements*" -or
    $_.Name -like "README*"
}

foreach ($File in $RootCode) {
    Copy-Item -LiteralPath $File.FullName -Destination $CodeDest -Force
}
Status "PASS" "Copied $($RootCode.Count) root-level code/config files"

Header "5. Save checkpoint inventory without model weights"

$CheckpointRoot = Join-Path $ProjectRoot "checkpoints"
$CheckpointRows = @()
if (Test-Path -LiteralPath $CheckpointRoot) {
    Get-ChildItem -LiteralPath $CheckpointRoot -File -Recurse | ForEach-Object {
        $CheckpointRows += [pscustomobject]@{
            RelativePath = $_.FullName.Substring($ProjectRoot.Length).TrimStart("\")
            Bytes        = $_.Length
            ModifiedUTC  = $_.LastWriteTimeUtc.ToString("o")
        }
    }
    $CheckpointRows | Export-Csv -NoTypeInformation -Encoding UTF8 `
        -Path (Join-Path $BundleDir "checkpoint_file_inventory.csv")
    Status "PASS" "Indexed $($CheckpointRows.Count) checkpoint files"
} else {
    Status "WARN" "No checkpoint directory found"
}

Header "6. Save long-path mappings"

$AllMaps = @()
if ($LongPathMap.Count -gt 0) {
    $LongPathMap | ForEach-Object {
        $AllMaps += [pscustomobject]@{
            Section = "results"
            OriginalRelativePath = $_.OriginalRelativePath
            FlattenedPath = $_.FlattenedPath
        }
    }
}
if ($ManuscriptMap.Count -gt 0) {
    $ManuscriptMap | ForEach-Object {
        $AllMaps += [pscustomobject]@{
            Section = "manuscript"
            OriginalRelativePath = $_.OriginalRelativePath
            FlattenedPath = $_.FlattenedPath
        }
    }
}

if ($AllMaps.Count -gt 0) {
    $AllMaps | Export-Csv -NoTypeInformation -Encoding UTF8 `
        -Path (Join-Path $BundleDir "long_path_map.csv")
    Status "INFO" "Flattened $($AllMaps.Count) overlong path(s); mapping saved"
} else {
    Status "PASS" "No path flattening required"
}

Header "7. Generate result digest"

$Digest = Join-Path $BundleDir "FINAL_RESULTS_DIGEST.md"
$Lines = New-Object System.Collections.Generic.List[string]
$Lines.Add("# TwoDimAudit Final Results Digest")
$Lines.Add("")
$Lines.Add("Created: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$Lines.Add("")
$Lines.Add("## Verified authoritative files")
foreach ($Name in $Required) {
    $Lines.Add("- PASS: ``results\$Name``")
}
$Lines.Add("")
$Lines.Add("## Confirmed corrected findings")
$Lines.Add("- Most influential MMUnlearner entity: ``malala_yousafzai``.")
$Lines.Add("- Excluding it changes MMUnlearner LB-CKA by ``+0.015475``.")
$Lines.Add("- All 3 key MMUnlearner pairwise intervals continue to exclude zero.")
$Lines.Add("- NPO stronger configurations produce collapse or substantial retain degradation rather than clean selective forgetting.")
$Lines.Add("")
$Lines.Add("## Manuscript safety notes")
$Lines.Add("- Do not reuse the obsolete ``entity_12`` or ``+0.0523`` statement.")
$Lines.Add("- Keep NPO sweep conclusions specific to NPO; do not claim a universal GradDiff ceiling.")
$Lines.Add("- Read exact probe, likelihood and CRP trajectory values from the copied CSV/JSON files before writing claims.")
$Lines | Set-Content -LiteralPath $Digest -Encoding UTF8
Status "PASS" "Generated FINAL_RESULTS_DIGEST.md"

Header "8. Scan manuscript for obsolete claims"

$AuditRows = @()
$RiskTerms = @("entity_12", "+0.0523", "0.0523", "practical ceiling")
foreach ($File in $ManuscriptFiles | Where-Object { $_.Extension -in @(".tex", ".txt", ".md") }) {
    $Text = Get-Content -LiteralPath $File.FullName -Raw -ErrorAction SilentlyContinue
    foreach ($Term in $RiskTerms) {
        if ($Text -match [regex]::Escape($Term)) {
            $AuditRows += [pscustomobject]@{
                File = $File.FullName.Substring($ProjectRoot.Length).TrimStart("\")
                Term = $Term
                Action = "REVIEW_AND_REPLACE"
            }
        }
    }
}

$AuditPath = Join-Path $BundleDir "manuscript_stale_claim_audit.csv"
$AuditRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $AuditPath
if ($AuditRows.Count -eq 0) {
    Status "PASS" "No known obsolete claims found"
} else {
    Status "WARN" "$($AuditRows.Count) obsolete/risky manuscript occurrence(s) found; see manuscript_stale_claim_audit.csv"
}

Header "9. Write README, inventory and checksums"

@"
# TwoDimAudit home-transfer bundle

This bundle is sufficient for manuscript writing from home.

Included:
- verified final result files;
- CPU/bootstrap/LOEO outputs;
- strength-sweep CRP summaries and manifests;
- BLIP-2 likelihood summaries;
- manuscript source found outside outputs/checkpoints;
- essential root code/configuration;
- checkpoint inventory without large checkpoint weights;
- long-path mapping where Windows paths had to be flattened;
- SHA256 checksums.

Start with:
1. FINAL_RESULTS_DIGEST.md
2. results\linear_probe_l31.csv
3. results\blip2_logprob_summary.csv
4. results\sweep_crp_manifest.json
5. results\authoritative_bootstrap_intervals.csv
6. results\influential_entity_excluded_pairwise.csv

No further GPU run is required for normal manuscript integration.
"@ | Set-Content -LiteralPath (Join-Path $BundleDir "README_HOME_TRANSFER.md") -Encoding UTF8

$Inventory = Get-ChildItem -LiteralPath $BundleDir -File -Recurse | ForEach-Object {
    [pscustomobject]@{
        RelativePath = $_.FullName.Substring($BundleDir.Length).TrimStart("\")
        Bytes        = $_.Length
        ModifiedUTC  = $_.LastWriteTimeUtc.ToString("o")
    }
}
$Inventory | Export-Csv -NoTypeInformation -Encoding UTF8 `
    -Path (Join-Path $BundleDir "file_inventory.csv")

$Checksums = Get-ChildItem -LiteralPath $BundleDir -File -Recurse | ForEach-Object {
    $H = Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName
    [pscustomobject]@{
        RelativePath = $_.FullName.Substring($BundleDir.Length).TrimStart("\")
        SHA256 = $H.Hash
    }
}
$Checksums | Export-Csv -NoTypeInformation -Encoding UTF8 `
    -Path (Join-Path $BundleDir "sha256_checksums.csv")

Status "PASS" "Generated inventory and SHA256 checksums"

Header "10. Create ZIP"

if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}

# tar handles long paths more reliably than Compress-Archive on Windows.
$Tar = Get-Command tar.exe -ErrorAction SilentlyContinue
if ($Tar) {
    Push-Location $TransferRoot
    & tar.exe -a -c -f "$BundleName.zip" "$BundleName"
    $TarExit = $LASTEXITCODE
    Pop-Location
    if ($TarExit -ne 0) {
        Status "FAIL" "tar.exe returned exit code $TarExit"
        exit 1
    }
} else {
    Compress-Archive -LiteralPath $BundleDir -DestinationPath $ZipPath -CompressionLevel Optimal
}

if (-not (Test-Path -LiteralPath $ZipPath)) {
    Status "FAIL" "ZIP was not created"
    exit 1
}

$ZipSizeMB = [math]::Round((Get-Item -LiteralPath $ZipPath).Length / 1MB, 2)

@"
OVERALL VERDICT: PASS
HOME TRANSFER ZIP: $ZipPath
ZIP SIZE MB: $ZipSizeMB
BUNDLE DIRECTORY: $BundleDir
CREATED: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
"@ | Set-Content -LiteralPath (Join-Path $TransferRoot "LATEST_HOME_TRANSFER.txt") -Encoding UTF8

Write-Host "`nOVERALL VERDICT: PASS" -ForegroundColor Green
Write-Host "HOME TRANSFER ZIP: $ZipPath" -ForegroundColor Cyan
Write-Host "ZIP SIZE: $ZipSizeMB MB" -ForegroundColor Cyan
Write-Host "You can now copy this ZIP to your home computer." -ForegroundColor Green
exit 0
