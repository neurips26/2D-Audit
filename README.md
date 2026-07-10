# AAAI Reviewer Pipeline V4

The latest console output shows that the project was still executing an older
launcher: the NPO script was absent and PowerShell still emitted
`NativeCommandError`. V4 removes ambiguity by printing a unique version marker
and includes a self-verifying installer.

## Install

From the `2DUnl` project root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\install_pipeline_v4.ps1 `
  -ZipPath "$HOME\Downloads\AAAI_reviewer_complete_pipeline_v4_20260703.zip" `
  -ProjectRoot "."
```

Expected final line:

```text
OVERALL VERDICT: PASS
```

## Verify

```powershell
Select-String `
  -Path .\run_all_reviewer_fixes.ps1 `
  -SimpleMatch "AAAI_REVIEWER_PIPELINE_V4_20260703"

Test-Path .\stage3_npo_smoke_complete_fixed.py

py -m py_compile `
  .\reviewer_complete_pipeline.py `
  .\stage3_npo_smoke_complete_fixed.py
```

## Continue from GPU only

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\run_all_reviewer_fixes.ps1 `
  -Mode GPU `
  -Resume
```

Then:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  .\run_all_reviewer_fixes.ps1 `
  -Mode Final
```

The previous `Set-ExecutionPolicy` warning is not a project failure. The
commands above apply bypass only to the new PowerShell process.
