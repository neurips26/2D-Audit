param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$OutputRoot = "",
    [switch]$IncludeLogs
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Text) {
    Write-Host "`n================================================================================" -ForegroundColor DarkCyan
    Write-Host $Text -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor DarkCyan
}

function Add-Result([string]$Status, [string]$Item, [string]$Detail) {
    $script:Results += [pscustomobject]@{ Status=$Status; Item=$Item; Detail=$Detail }
    $colour = switch ($Status) { "PASS" {"Green"} "WARN" {"Yellow"} "FAIL" {"Red"} default {"White"} }
    Write-Host "[$Status] $Item - $Detail" -ForegroundColor $colour
}

function Copy-TreeFiltered([string]$Source,[string]$Destination) {
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $PathMap = @()
    Get-ChildItem -LiteralPath $Source -File -Recurse | Where-Object {
        $_.Extension -in @('.csv','.json','.txt','.md','.png','.pdf')
    } | ForEach-Object {
        $relative=$_.FullName.Substring($Source.Length).TrimStart('\')
        $target=Join-Path $Destination $relative

        # Windows PowerShell 5.1 commonly fails around MAX_PATH.  Preserve short paths
        # normally; flatten only paths that would be too long and record a mapping.
        if ($target.Length -ge 235) {
            $sha = [System.Security.Cryptography.SHA256]::Create()
            try {
                $bytes = [System.Text.Encoding]::UTF8.GetBytes($relative)
                $hash = ([System.BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-','').Substring(0,16).ToLowerInvariant()
            } finally {
                $sha.Dispose()
            }
            $safeLeaf = ($_.BaseName -replace '[^A-Za-z0-9._-]','_')
            if ($safeLeaf.Length -gt 70) { $safeLeaf = $safeLeaf.Substring(0,70) }
            $flatDir = Join-Path $Destination '_flattened_long_paths'
            New-Item -ItemType Directory -Force -Path $flatDir | Out-Null
            $target = Join-Path $flatDir ("{0}_{1}{2}" -f $hash,$safeLeaf,$_.Extension)
            $PathMap += [pscustomobject]@{ OriginalRelativePath=$relative; CopiedRelativePath=$target.Substring($Destination.Length).TrimStart('\') }
        } else {
            New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
        }

        Copy-Item -LiteralPath $_.FullName -Destination $target -Force
    }

    if ($PathMap.Count -gt 0) {
        $PathMap | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $Destination '_long_path_map.csv')
        Write-Host "[INFO] Flattened $($PathMap.Count) long path(s); mapping saved to _long_path_map.csv" -ForegroundColor Yellow
    }
}

$ProjectRoot=(Resolve-Path -LiteralPath $ProjectRoot).Path
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot=Join-Path $ProjectRoot 'outputs\revision\home_transfer'
}
$Timestamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$BundleName="TwoDimAudit_home_transfer_$Timestamp"
$BundleDir=Join-Path $OutputRoot $BundleName
$ZipPath="$BundleDir.zip"
$Results=@()
$PipelineRoot=Join-Path $ProjectRoot 'outputs\revision\reviewer_complete_pipeline'
$ManuscriptResults=Join-Path $PipelineRoot 'manuscript_results'

Write-Step 'TwoDimAudit final home-transfer packager'
Write-Host "[config] ProjectRoot = $ProjectRoot"
Write-Host "[config] BundleDir   = $BundleDir"
New-Item -ItemType Directory -Force -Path $BundleDir | Out-Null

Write-Step '1. Verify final experiment outputs'
$Required=@(
 'final_report.json','final_report.txt','authoritative_bootstrap_intervals.csv',
 'authoritative_bootstrap_intervals.json','leave_one_entity_out.csv',
 'mmunlearner_most_influential_entity.json','influential_entity_excluded_pairwise.csv',
 'linear_probe_l31.csv','hyperparameter_inventory.csv','manuscript_audit.json',
 'sweep_crp_manifest.json','blip2_logprob_results.json','blip2_logprob_summary.csv'
)
$MissingRequired=@()
foreach($name in $Required){
    $p=Join-Path $ManuscriptResults $name
    if(Test-Path -LiteralPath $p){ Add-Result 'PASS' 'Required result' $name }
    else { $MissingRequired+=$name; Add-Result 'FAIL' 'Required result' "Missing: $name" }
}

Write-Step '2. Copy manuscript-ready results'
$ResultsDest=Join-Path $BundleDir 'results'
New-Item -ItemType Directory -Force -Path $ResultsDest | Out-Null
if(Test-Path -LiteralPath $ManuscriptResults){
    Copy-Item -LiteralPath (Join-Path $ManuscriptResults '*') -Destination $ResultsDest -Recurse -Force
    Add-Result 'PASS' 'Manuscript results' $ManuscriptResults
}else{ Add-Result 'FAIL' 'Manuscript results' 'Folder missing' }

foreach($pair in @(
    @{S=(Join-Path $PipelineRoot 'cpu_analysis');D=(Join-Path $BundleDir 'results\cpu_analysis')},
    @{S=(Join-Path $PipelineRoot 'sweep_crp');D=(Join-Path $BundleDir 'results\sweep_crp')},
    @{S=(Join-Path $PipelineRoot 'blip2_logprob');D=(Join-Path $BundleDir 'results\blip2_logprob')}
)){
    if(Test-Path -LiteralPath $pair.S){ Copy-TreeFiltered $pair.S $pair.D; Add-Result 'PASS' 'Additional result tree' $pair.S }
    else { Add-Result 'WARN' 'Additional result tree' "Missing: $($pair.S)" }
}

Write-Step '3. Collect manuscript source'
$ManuscriptDest=Join-Path $BundleDir 'manuscript'
New-Item -ItemType Directory -Force -Path $ManuscriptDest | Out-Null
$ManuscriptCandidates=Get-ChildItem -LiteralPath $ProjectRoot -File -Recurse -ErrorAction SilentlyContinue | Where-Object {
    $_.Extension -in @('.tex','.bib','.cls','.sty','.bst','.pdf','.docx') -and
    $_.FullName -notmatch '\\checkpoints\\|\\\.venv\\|\\venv\\|\\outputs\\revision\\home_transfer\\'
}
foreach($f in $ManuscriptCandidates){
    $rel=$f.FullName.Substring($ProjectRoot.Length).TrimStart('\')
    $target=Join-Path $ManuscriptDest $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
    Copy-Item -LiteralPath $f.FullName -Destination $target -Force
}
if($ManuscriptCandidates.Count -gt 0){ Add-Result 'PASS' 'Manuscript source' "$($ManuscriptCandidates.Count) files copied" }
else { Add-Result 'WARN' 'Manuscript source' 'No manuscript files found' }

Write-Step '4. Collect essential code and configuration'
$CodeDest=Join-Path $BundleDir 'code'; New-Item -ItemType Directory -Force -Path $CodeDest | Out-Null
$CodeFiles=Get-ChildItem -LiteralPath $ProjectRoot -File -ErrorAction SilentlyContinue | Where-Object {
    $_.Extension -in @('.py','.ps1','.json','.yaml','.yml','.toml','.ini','.cfg','.txt') -or $_.Name -like 'README*'
}
foreach($f in $CodeFiles){ Copy-Item -LiteralPath $f.FullName -Destination $CodeDest -Force }
Add-Result 'PASS' 'Root code/config' "$($CodeFiles.Count) files copied"

Write-Step '5. Record checkpoint metadata without copying weights'
$CheckpointRoot=Join-Path $ProjectRoot 'checkpoints'
$CheckpointManifest=@()
if(Test-Path -LiteralPath $CheckpointRoot){
    Get-ChildItem -LiteralPath $CheckpointRoot -Directory -Recurse | ForEach-Object {
        $files=Get-ChildItem -LiteralPath $_.FullName -File -ErrorAction SilentlyContinue
        if($files.Count -gt 0){
            $CheckpointManifest += [pscustomobject]@{
                RelativePath=$_.FullName.Substring($ProjectRoot.Length).TrimStart('\')
                FileCount=$files.Count
                TotalBytes=($files|Measure-Object Length -Sum).Sum
                ConfigFiles=(($files|Where-Object {$_.Extension -in @('.json','.txt','.yaml','.yml')}).Name -join '; ')
            }
        }
    }
    $CheckpointManifest|Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $BundleDir 'checkpoint_manifest.csv')
    Add-Result 'PASS' 'Checkpoint manifest' "$($CheckpointManifest.Count) directories indexed"
}else{ Add-Result 'WARN' 'Checkpoint manifest' 'No checkpoint directory found' }

Write-Step '6. Generate digest'
$Digest=Join-Path $BundleDir 'FINAL_RESULTS_DIGEST.md'
$lines=@('# TwoDimAudit Final Results Digest','',"Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",'')
foreach($name in $Required){ $lines += "- $($(if(Test-Path (Join-Path $ResultsDest $name)){'PASS'}else{'MISSING'})): ``$name``" }
$lines += '','## Key confirmed conclusions','',
'- Correct estimator-consistent entity-cluster bootstrap and LOEO completed.',
'- Most influential MMUnlearner entity: `malala_yousafzai`.',
'- MMUnlearner LB-CKA change after exclusion: `+0.015475`.',
'- All three key pairwise intervals remain separated after exclusion.',
'- NPO strength sweep shows collapse or retain damage at stronger settings.',
'- Linear probe, sweep CRP, and BLIP-2 likelihood analyses completed.',
'', 'Use the CSV/JSON files in `results` as the authoritative numerical source.'
$lines|Set-Content -LiteralPath $Digest -Encoding UTF8
Add-Result 'PASS' 'Final results digest' $Digest

Write-Step '7. Scan manuscript for stale claims'
$AuditTerms=@('entity_12','+0.0523','0.0523','practical ceiling')
$AuditRows=@()
foreach($f in ($ManuscriptCandidates|Where-Object {$_.Extension -in @('.tex','.md','.txt')})){
    $content=Get-Content -LiteralPath $f.FullName -Raw -ErrorAction SilentlyContinue
    foreach($term in $AuditTerms){
        if($content -match [regex]::Escape($term)){
            $AuditRows += [pscustomobject]@{File=$f.FullName.Substring($ProjectRoot.Length).TrimStart('\');Term=$term;Status='REVIEW'}
        }
    }
}
$AuditRows|Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $BundleDir 'manuscript_stale_claim_audit.csv')
if($AuditRows.Count -eq 0){ Add-Result 'PASS' 'Stale-claim scan' 'No known obsolete claim found' }
else { Add-Result 'WARN' 'Stale-claim scan' "$($AuditRows.Count) item(s) require review" }

if($IncludeLogs){
    Write-Step '8. Include logs'
    foreach($logRoot in @((Join-Path $PipelineRoot 'launcher\logs'),(Join-Path $PipelineRoot 'logs'))){
        if(Test-Path -LiteralPath $logRoot){
            $dest=Join-Path $BundleDir ('logs\'+(Split-Path $logRoot -Leaf))
            New-Item -ItemType Directory -Force -Path $dest|Out-Null
            Copy-Item -LiteralPath (Join-Path $logRoot '*') -Destination $dest -Recurse -Force
        }
    }
    Add-Result 'PASS' 'Logs' 'Included'
}else{ Add-Result 'PASS' 'Logs' 'Skipped; use -IncludeLogs to include' }

Write-Step '9. Inventory, checksums, README'
$inventory=Get-ChildItem -LiteralPath $BundleDir -File -Recurse|ForEach-Object{
    [pscustomobject]@{RelativePath=$_.FullName.Substring($BundleDir.Length).TrimStart('\');Bytes=$_.Length;ModifiedUTC=$_.LastWriteTimeUtc.ToString('o')}
}
$inventory|Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $BundleDir 'file_inventory.csv')
Get-ChildItem -LiteralPath $BundleDir -File -Recurse|ForEach-Object{
    $h=Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName
    [pscustomobject]@{RelativePath=$_.FullName.Substring($BundleDir.Length).TrimStart('\');SHA256=$h.Hash}
}|Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $BundleDir 'sha256_checksums.csv')

@"
# TwoDimAudit home-transfer bundle

Source project: $ProjectRoot
Created: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

Open `FINAL_RESULTS_DIGEST.md` first. Use `results` as the authoritative source for all numerical claims.
Do not reuse obsolete claims involving `entity_12` or `0.0523`.
Correct influential entity: `malala_yousafzai`; LB-CKA change: `+0.015475`.
Keep the NPO conclusion method-specific; do not claim it proves a universal GradDiff ceiling.
Large checkpoint weights are intentionally excluded.
"@|Set-Content -LiteralPath (Join-Path $BundleDir 'README_HOME_TRANSFER.md') -Encoding UTF8
$Results|Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $BundleDir 'packager_report.csv')
$Results|ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $BundleDir 'packager_report.json') -Encoding UTF8

Write-Step '10. Create ZIP'
if(Test-Path -LiteralPath $ZipPath){Remove-Item -LiteralPath $ZipPath -Force}
Compress-Archive -LiteralPath $BundleDir -DestinationPath $ZipPath -CompressionLevel Optimal
if(Test-Path -LiteralPath $ZipPath){
    $size=[math]::Round((Get-Item $ZipPath).Length/1MB,2)
    Add-Result 'PASS' 'ZIP archive' "$ZipPath ($size MB)"
}else{ Add-Result 'FAIL' 'ZIP archive' 'ZIP creation failed' }

$FinalStatus=if($MissingRequired.Count -gt 0 -or -not(Test-Path $ZipPath)){'FAIL'}elseif(($Results|Where-Object Status -eq 'WARN').Count -gt 0){'WARN'}else{'PASS'}
$StatusFile=Join-Path $OutputRoot 'LATEST_HOME_TRANSFER.txt'
@"
OVERALL VERDICT: $FinalStatus
BUNDLE DIRECTORY: $BundleDir
ZIP FILE: $ZipPath
REQUIRED FILES MISSING: $($MissingRequired.Count)
WARNINGS: $(($Results|Where-Object Status -eq 'WARN').Count)
"@|Set-Content -LiteralPath $StatusFile -Encoding UTF8

Write-Host "`nOVERALL VERDICT: $FinalStatus" -ForegroundColor $(if($FinalStatus -eq 'PASS'){'Green'}elseif($FinalStatus -eq 'WARN'){'Yellow'}else{'Red'})
Write-Host "HOME TRANSFER ZIP: $ZipPath" -ForegroundColor Cyan
Write-Host "STATUS FILE:       $StatusFile" -ForegroundColor Cyan
if($FinalStatus -eq 'FAIL'){exit 1}else{exit 0}
