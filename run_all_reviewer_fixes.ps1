param(
    [ValidateSet("Preflight", "CPU", "GPU", "All", "Final")]
    [string]$Mode = "All",

    [switch]$Resume,

    [switch]$SkipTraining,

    [int]$BootstrapReplicates = 2000
)

$ErrorActionPreference = "Stop"
$PIPELINE_VERSION = "AAAI_REVIEWER_PIPELINE_V5_20260703"
Set-Location $PSScriptRoot
Write-Host "[pipeline] VERSION=$PIPELINE_VERSION" -ForegroundColor Green

$PythonScript = ".\reviewer_complete_pipeline.py"
$OutputRoot = ".\outputs\revision\reviewer_complete_pipeline"
$LauncherDir = Join-Path $OutputRoot "launcher"
$LogDir = Join-Path $LauncherDir "logs"
$ReportJson = Join-Path $LauncherDir "launcher_report.json"
$ReportTxt = Join-Path $LauncherDir "launcher_report.txt"

New-Item -ItemType Directory -Force -Path $LogDir

$Stages = New-Object System.Collections.Generic.List[object]
$Overall = "PASS"

function Add-Stage {
    param(
        [string]$Name,
        [string]$Verdict,
        [string]$Detail,
        [string]$Log = ""
    )

    $Stages.Add([pscustomobject]@{
        name = $Name
        verdict = $Verdict
        detail = $Detail
        log = $Log
        timestamp = (Get-Date).ToString("o")
    })

    if ($Verdict -eq "FAIL") {
        $script:Overall = "FAIL"
    }
    elseif ($Verdict -eq "WARN" -and $script:Overall -eq "PASS") {
        $script:Overall = "WARN"
    }

    Write-Host "[$Verdict] $Name - $Detail"
}

function Quote-NativeArgument {
    param([string]$Value)

    if ($null -eq $Value) {
        return '""'
    }

    if ($Value -notmatch '[\s"]') {
        return $Value
    }

    $Escaped = $Value -replace '(\\*)"', '$1$1\"'
    $Escaped = $Escaped -replace '(\\+)$', '$1$1'
    return '"' + $Escaped + '"'
}

function Invoke-LoggedStage {
    param(
        [string]$Name,
        [string[]]$Command,
        [string]$LogName
    )

    $Log = Join-Path $LogDir $LogName
    $StdoutLog = "$Log.stdout"
    $StderrLog = "$Log.stderr"

    Write-Host ""
    Write-Host ("=" * 90) -ForegroundColor Cyan
    Write-Host $Name -ForegroundColor Cyan
    Write-Host ("=" * 90) -ForegroundColor Cyan

    $Executable = $Command[0]
    $Arguments = @()
    if ($Command.Count -gt 1) {
        $Arguments = @($Command[1..($Command.Count - 1)])
    }

    Remove-Item $StdoutLog, $StderrLog -Force -ErrorAction SilentlyContinue

    $ArgumentLine = ($Arguments | ForEach-Object {
        Quote-NativeArgument ([string]$_)
    }) -join " "

    try {
        $Process = Start-Process `
            -FilePath $Executable `
            -ArgumentList $ArgumentLine `
            -WorkingDirectory $PSScriptRoot `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $StdoutLog `
            -RedirectStandardError $StderrLog

        $Code = [int]$Process.ExitCode
    }
    catch {
        $Code = 1
        $_ | Out-String | Set-Content $StderrLog -Encoding UTF8
    }

    $Stdout = if (Test-Path $StdoutLog) {
        Get-Content $StdoutLog -Raw
    }
    else {
        ""
    }

    $Stderr = if (Test-Path $StderrLog) {
        Get-Content $StderrLog -Raw
    }
    else {
        ""
    }

    $Combined = @(
        "COMMAND: $Executable $ArgumentLine"
        "EXIT CODE: $Code"
        ""
        "STDOUT"
        $Stdout
        ""
        "STDERR"
        $Stderr
    ) -join [Environment]::NewLine

    $Combined | Set-Content $Log -Encoding UTF8

    if ($Stdout) {
        Write-Host $Stdout
    }

    if ($Stderr) {
        Write-Host "[captured stderr]" -ForegroundColor Yellow
        Write-Host $Stderr
    }

    if ($Code -ne 0 -or $Combined -match "OVERALL VERDICT:\s*FAIL") {
        Add-Stage $Name "FAIL" "Exit code $Code. See $Log" $Log
    }
    elseif ($Combined -match "OVERALL VERDICT:\s*WARN") {
        Add-Stage $Name "WARN" "Completed with warnings. See $Log" $Log
    }
    else {
        Add-Stage $Name "PASS" "Completed successfully. See $Log" $Log
    }

    $script:LastStageExitCode = $Code
}

function Save-LauncherReport {
    $Payload = [pscustomobject]@{
        pipeline_version = $PIPELINE_VERSION
        mode = $Mode
        resume = [bool]$Resume
        skip_training = [bool]$SkipTraining
        bootstrap_replicates = $BootstrapReplicates
        project_root = (Get-Location).Path
        overall_verdict = $Overall
        completed_at = (Get-Date).ToString("o")
        stages = $Stages
    }

    $Payload | ConvertTo-Json -Depth 8 | Set-Content $ReportJson -Encoding UTF8

    @(
        "AAAI REVIEWER COMPLETE PIPELINE"
        ("=" * 90)
        "Version: $PIPELINE_VERSION"
        "Mode: $Mode"
        "Overall: $Overall"
        ""
    ) + ($Stages | ForEach-Object {
        "[$($_.verdict)] $($_.name): $($_.detail)"
    }) + @(
        ""
        "OVERALL VERDICT: $Overall"
        "JSON REPORT: $ReportJson"
    ) | Set-Content $ReportTxt -Encoding UTF8

    Get-Content $ReportTxt
}

if (-not (Test-Path $PythonScript)) {
    Add-Stage "Python runner" "FAIL" "Missing $PythonScript"
    Save-LauncherReport
    exit 1
}

py -m py_compile $PythonScript
if ($LASTEXITCODE -ne 0) {
    Add-Stage "Python syntax" "FAIL" "reviewer_complete_pipeline.py did not compile."
    Save-LauncherReport
    exit 1
}
Add-Stage "Python syntax" "PASS" "reviewer_complete_pipeline.py compiled."

$RunPreflight = $Mode -in @("Preflight", "CPU", "GPU", "All")
$RunCPU = $Mode -in @("CPU", "All")
$RunGPU = $Mode -in @("GPU", "All")
$RunFinal = $Mode -in @("Final", "All")

if ($RunPreflight) {
    Invoke-LoggedStage `
        -Name "Preflight" `
        -Command @("py", "-u", $PythonScript, "preflight") `
        -LogName "01_preflight.log"

    if ($Overall -eq "FAIL") {
        Save-LauncherReport
        exit 1
    }
}

if ($RunCPU) {
    Invoke-LoggedStage `
        -Name "Authoritative CPU analyses" `
        -Command @(
            "py", "-u", $PythonScript, "cpu",
            "--n-bootstrap", "$BootstrapReplicates",
            "--tex-root", "."
        ) `
        -LogName "02_cpu_analysis.log"
}

if ($RunGPU) {
    if (-not $SkipTraining) {
        # GradDiff trajectory. A single run saves 5/10/20/50-step checkpoints.
        $GradScript = ".\train_graddiff_llava.py"
        $GradCheckpointCandidates = @(
            ".\checkpoints\graddiff\lr0p0001_lambda1_seed42\graddiff_llava_50steps",
            ".\checkpoints\graddiff\graddiff_llava_50steps"
        )
        $ExistingGradFinal = $GradCheckpointCandidates |
            Where-Object { Test-Path $_ } |
            Select-Object -First 1

        if (-not (Test-Path $GradScript)) {
            Add-Stage "GradDiff strength trajectory" "FAIL" "Missing $GradScript"
        }
        elseif ($ExistingGradFinal) {
            Add-Stage "GradDiff strength trajectory" "PASS" "Existing final checkpoint found: $ExistingGradFinal"
        }
        else {
            Invoke-LoggedStage `
                -Name "GradDiff strength trajectory" `
                -Command @(
                    "py", "-u", $GradScript,
                    "--steps", "50",
                    "--save_steps", "5", "10", "20", "50",
                    "--lambda_retain", "1.0",
                    "--lr", "1e-4",
                    "--seed", "42"
                ) `
                -LogName "03_graddiff_training.log"
        }

        # Prefer the latest corrected NPO implementation available locally.
        $NpoCandidates = @(
            ".\stage3_npo_smoke_complete_fixed.py",
            ".\stage3_npo_smoke_authoritative_fixed.py",
            ".\stage3_npo_smoke_final_fixed.py",
            ".\stage3_npo_smoke.py"
        )
        $NpoScript = $NpoCandidates |
            Where-Object { Test-Path $_ } |
            Select-Object -First 1

        if (-not $NpoScript) {
            Add-Stage "NPO strength sweep" "WARN" "No NPO sweep script found; NPO sweep skipped."
        }
        else {
            $NpoArgs = @("py", "-u", $NpoScript, "--sweep")
            if ($Resume) {
                $NpoArgs += "--resume"
            }

            Invoke-LoggedStage `
                -Name "NPO strength sweep" `
                -Command $NpoArgs `
                -LogName "04_npo_sweep.log"
        }
    }
    else {
        Add-Stage "Model training" "WARN" "Skipped by -SkipTraining. Existing checkpoints will be analysed."
    }

    $SweepArgs = @("py", "-u", $PythonScript, "sweep-crp")
    if ($Resume) {
        $SweepArgs += "--resume"
    }

    Invoke-LoggedStage `
        -Name "Strength-sweep CRP extraction" `
        -Command $SweepArgs `
        -LogName "05_sweep_crp.log"

    $BlipArgs = @(
        "py", "-u", $PythonScript,
        "blip2-logprob", "--device", "cuda"
    )
    if ($Resume) {
        $BlipArgs += "--resume"
    }

    Invoke-LoggedStage `
        -Name "BLIP-2 reference-answer likelihood audit" `
        -Command $BlipArgs `
        -LogName "06_blip2_logprob.log"
}

if ($RunFinal) {
    Invoke-LoggedStage `
        -Name "Final artifact audit" `
        -Command @("py", "-u", $PythonScript, "final") `
        -LogName "07_final_audit.log"
}

Save-LauncherReport

if ($Overall -eq "FAIL") {
    exit 1
}
exit 0
