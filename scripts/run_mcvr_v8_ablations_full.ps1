[CmdletBinding()]
param(
    [string]$PythonPath = "",
    [string]$Device = "cuda:0",
    [string]$AssetRepoRoot = "E:\3dconformergenerationcode\4dadapter-v8",
    [string]$OutputRepoRoot = "E:\3dconformergenerationcode\4dadapter-v8"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot
if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    if (-not [string]::IsNullOrWhiteSpace($env:MCVR_PYTHON)) {
        $PythonPath = $env:MCVR_PYTHON
    }
    elseif (Test-Path -LiteralPath "E:\miniconda\envs\etflow-5080-v2\python.exe") {
        $PythonPath = "E:\miniconda\envs\etflow-5080-v2\python.exe"
    }
    else {
        $PythonPath = (Get-Command python -ErrorAction Stop).Source
    }
}
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "MCVR Python executable does not exist: $PythonPath"
}

$Branch = (& git branch --show-current).Trim()
$Head = (& git rev-parse HEAD).Trim()
$Worktree = (& git status --short)
if ($Branch -ne "mcvr-v8-ablation") {
    throw "Expected branch mcvr-v8-ablation, found $Branch"
}
if (-not [string]::IsNullOrWhiteSpace(($Worktree -join "`n"))) {
    throw "Ablation worktree must be clean before formal training"
}
& git merge-base --is-ancestor 4df21d766afadab169ecc7208477a6ca6ffe384a $Head
if ($LASTEXITCODE -ne 0) {
    throw "Ablation HEAD is not descended from frozen 4df21d7"
}

$Sampler = Join-Path $AssetRepoRoot "reports/ecir_mvr/v8_full_v1/formal_large_stratified_manifest.json"
$ReportDir = Join-Path $OutputRepoRoot "reports/ecir_mvr/v8_full_v1/ablations"
$RunRoot = Join-Path $OutputRepoRoot "diagnostics/ecir_mvr/v8_full_v1"
$Registry = Join-Path $ReportDir "EXPERIMENT_REGISTRY.json"
& $PythonPath scripts/preflight_mcvr_v8_ablations.py `
    --sampler-manifest $Sampler `
    --output $Registry | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Formal ablation identity preflight failed"
}

$Runs = @(
    [ordered]@{Id="NO_CONSTRAINT"; Output="no_constraint_seed43_attempt2"; Config="configs/ecir_mvr_v8_ablation_no_constraint_formal_large_200k.yaml"},
    [ordered]@{Id="NO_CONFIDENCE"; Output="no_confidence_seed43"; Config="configs/ecir_mvr_v8_ablation_no_confidence_formal_large_200k.yaml"},
    [ordered]@{Id="NO_ERROR_STATE"; Output="no_error_state_seed43"; Config="configs/ecir_mvr_v8_ablation_no_error_state_formal_large_200k.yaml"},
    [ordered]@{Id="NO_TYPE_NORMALIZATION"; Output="no_type_normalization_seed43"; Config="configs/ecir_mvr_v8_ablation_no_type_normalization_formal_large_200k.yaml"}
)

function Test-PristineLauncherDirectory {
    param([Parameter(Mandatory=$true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $true }
    $Entries = @(Get-ChildItem -LiteralPath $Path -Force)
    foreach ($Entry in $Entries) {
        if ($Entry.PSIsContainer -or $Entry.Name -notin @("stdout.log", "stderr.log") -or $Entry.Length -ne 0) {
            return $false
        }
    }
    return $true
}

$Results = @()
foreach ($Run in $Runs) {
    $OutputDir = Join-Path $RunRoot ("ablations/" + $Run.Output)
    if (-not (Test-PristineLauncherDirectory -Path $OutputDir)) {
        throw "$($Run.Id) output exists and will not be deleted or overwritten: $OutputDir"
    }
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    $Stdout = Join-Path $OutputDir "stdout.log"
    $Stderr = Join-Path $OutputDir "stderr.log"
    Write-Host "Starting $($Run.Id) at $(Get-Date -Format o)"
    $Arguments = @(
        "scripts/train_ecir_mvr_v8.py",
        "--config", (Join-Path $RepoRoot $Run.Config),
        "--sampler-manifest", $Sampler,
        "--output-dir", $OutputDir,
        "--steps", "200000",
        "--validation-batches", "625",
        "--device", $Device
    )
    $Process = Start-Process `
        -FilePath $PythonPath `
        -ArgumentList $Arguments `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr `
        -Wait `
        -PassThru
    $ExitCode = $Process.ExitCode
    if ($ExitCode -ne 0) {
        throw "$($Run.Id) failed with exit code $ExitCode; subsequent ablations will not start"
    }
    $Verification = Join-Path $OutputDir "automation_verification.json"
    & $PythonPath scripts/verify_mcvr_v8_ablation_run.py `
        --output-dir $OutputDir `
        --ablation-id $Run.Id `
        --exit-code $ExitCode `
        --output $Verification | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "$($Run.Id) completed-process verification failed"
    }
    $Result = Get-Content -LiteralPath $Verification -Raw | ConvertFrom-Json
    $Results += $Result
    Write-Host "$($Run.Id) completed and verified at $(Get-Date -Format o)"
}

& $PythonPath scripts/report_mcvr_v8_ablations.py `
    --run-root $RunRoot `
    --output-dir $ReportDir | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Ablation summary generation failed"
}

$Complete = [ordered]@{
    schema_version = "mcvr-v8-ablation-complete-v1"
    status = "MCVR_V8_ABLATIONS_COMPLETED"
    branch = $Branch
    git_head = $Head
    run_order = @($Runs | ForEach-Object { $_.Id })
    results = $Results
    formal_test_records_read = 0
    frozen_holdout_records_read = 0
    completed_at = (Get-Date).ToUniversalTime().ToString("o")
}
$CompletePath = Join-Path $ReportDir "ABLATION_COMPLETE.json"
$Temporary = "$CompletePath.tmp"
$Complete | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Temporary -Encoding utf8
Move-Item -LiteralPath $Temporary -Destination $CompletePath -Force
Write-Host "MCVR_V8_ABLATIONS_COMPLETED"
