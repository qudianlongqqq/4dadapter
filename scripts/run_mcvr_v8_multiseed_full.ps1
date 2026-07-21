[CmdletBinding()]
param(
    [string]$PythonPath = "",
    [string]$Device = "cuda:0"
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

$ExpectedBranch = "mcvr-v8-multiseed"
$FrozenBase = "4df21d766afadab169ecc7208477a6ca6ffe384a"
$Branch = (& git branch --show-current).Trim()
$Head = (& git rev-parse HEAD).Trim()
$Worktree = (& git status --short)
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect git working tree"
}
if ($Branch -ne $ExpectedBranch) {
    throw "Expected git branch $ExpectedBranch, found $Branch"
}
if (-not [string]::IsNullOrWhiteSpace(($Worktree -join "`n"))) {
    throw "Working tree must be clean before multi-seed execution"
}
& git merge-base --is-ancestor $FrozenBase $Head
if ($LASTEXITCODE -ne 0) {
    throw "Current HEAD is not descended from frozen HEAD $FrozenBase"
}

$RegistryPath = Join-Path $RepoRoot "reports/ecir_mvr/v8_full_v1/multiseed/EXPERIMENT_REGISTRY.json"
if (-not (Test-Path -LiteralPath $RegistryPath)) {
    throw "Frozen multi-seed registry is missing"
}
$FrozenRegistry = Get-Content -LiteralPath $RegistryPath -Raw | ConvertFrom-Json
$PreflightOutput = Join-Path ([System.IO.Path]::GetTempPath()) ("mcvr-v8-preflight-" + [guid]::NewGuid() + ".json")
try {
    & $PythonPath "scripts/preflight_mcvr_v8_multiseed.py" --output $PreflightOutput | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Multi-seed identity preflight failed"
    }
    $LiveRegistry = Get-Content -LiteralPath $PreflightOutput -Raw | ConvertFrom-Json
}
finally {
    if (Test-Path -LiteralPath $PreflightOutput) {
        Remove-Item -LiteralPath $PreflightOutput -Force
    }
}

if ($LiveRegistry.branch -ne $ExpectedBranch) {
    throw "Preflight branch identity changed"
}
foreach ($Seed in @(12, 48)) {
    $Key = [string]$Seed
    if ($LiveRegistry.configs.$Key.config_file_sha256 -ne $FrozenRegistry.configs.$Key.config_file_sha256) {
        throw "Seed$Seed config file SHA256 changed"
    }
    if ($LiveRegistry.configs.$Key.inherited_resolved_config_sha256 -ne $FrozenRegistry.configs.$Key.inherited_resolved_config_sha256) {
        throw "Seed$Seed inherited resolved config SHA256 changed"
    }
}
$LiveIdentities = $LiveRegistry.identities | ConvertTo-Json -Depth 20 -Compress
$FrozenIdentities = $FrozenRegistry.identities | ConvertTo-Json -Depth 20 -Compress
if ($LiveIdentities -ne $FrozenIdentities) {
    throw "D1, dataset, sampler, scales, or baseline cache identity changed"
}
if ($FrozenRegistry.isolation.formal_test_records_read -ne 0 -or
    $FrozenRegistry.isolation.frozen_holdout_records_read -ne 0) {
    throw "Frozen registry violates test/holdout isolation"
}

$RunRoot = Join-Path $RepoRoot "diagnostics/ecir_mvr/v8_full_v1/formal_large_200k"
$ReportDir = Join-Path $RepoRoot "reports/ecir_mvr/v8_full_v1/multiseed"
$Seed43Evaluation = Join-Path $RepoRoot "diagnostics/ecir_mvr/v8_full_v1/formal_large_200k/full_seed43/validation_cache/step012500/full/evaluation.json"
if (-not (Test-Path -LiteralPath $Seed43Evaluation)) {
    throw "Frozen Seed43 FULL10K evaluation is missing"
}

function Test-PristineLauncherDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Path
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return $true
    }
    $Entries = @(Get-ChildItem -LiteralPath $Path -Force)
    foreach ($Entry in $Entries) {
        if ($Entry.PSIsContainer -or
            $Entry.Name -notin @("stdout.log", "stderr.log") -or
            $Entry.Length -ne 0) {
            return $false
        }
    }
    return $true
}

function Invoke-MCVRSeed {
    param(
        [Parameter(Mandatory = $true)][int]$Seed
    )
    $Config = Join-Path $RepoRoot "configs/ecir_mvr_v8_full_v1_formal_large_200k_seed$Seed.yaml"
    $OutputDir = Join-Path $RunRoot "full_seed$Seed"
    if (-not (Test-PristineLauncherDirectory -Path $OutputDir)) {
        throw "Seed$Seed output already exists; it will not be deleted or overwritten: $OutputDir"
    }
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    # These launcher filenames are explicitly permitted by the runner's
    # otherwise fail-closed nonempty-output check.
    $Stdout = Join-Path $OutputDir "stdout.log"
    $Stderr = Join-Path $OutputDir "stderr.log"
    Write-Host "Starting MCVR V8 Seed$Seed on $Device at $(Get-Date -Format o)"
    # The resolved config retains the original 200K horizon. Its pre-registered
    # stop request is materialized before optimizer step 1 and stops exactly at 12500.
    & $PythonPath "scripts/train_ecir_mvr_v8.py" `
        --config $Config `
        --output-dir $OutputDir `
        --steps 200000 `
        --validation-batches 625 `
        --device $Device `
        1> $Stdout 2> $Stderr
    $ReturnCode = $LASTEXITCODE
    if ($ReturnCode -ne 0) {
        throw "Seed$Seed training failed with exit code $ReturnCode; Seed48/reporting will not start"
    }
    $Verification = Join-Path $OutputDir "automation_verification.json"
    & $PythonPath "scripts/verify_mcvr_v8_multiseed_run.py" `
        --output-dir $OutputDir `
        --seed $Seed `
        --exit-code $ReturnCode `
        --output $Verification | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Seed$Seed completed process verification failed"
    }
    $Result = Get-Content -LiteralPath $Verification -Raw | ConvertFrom-Json
    if ($Result.status -ne "COMPLETED" -or $Result.exit_code -ne 0) {
        throw "Seed$Seed did not produce normalized COMPLETED/exit-code-0 evidence"
    }
    Write-Host "Seed$Seed completed and verified at $(Get-Date -Format o)"
    return $Result
}

$Seed12Output = Join-Path $RunRoot "full_seed12"
$Seed48Output = Join-Path $RunRoot "full_seed48"
if (-not (Test-PristineLauncherDirectory -Path $Seed12Output) -or
    -not (Test-PristineLauncherDirectory -Path $Seed48Output)) {
    throw "A Seed12/48 output directory already exists; no failed or partial result was removed"
}

$TotalTimer = [System.Diagnostics.Stopwatch]::StartNew()
$Seed12 = Invoke-MCVRSeed -Seed 12
$Seed48 = Invoke-MCVRSeed -Seed 48

$Seed12Evaluation = Join-Path $Seed12Output "validation_cache/step012500/full/evaluation.json"
$Seed48Evaluation = Join-Path $Seed48Output "validation_cache/step012500/full/evaluation.json"
& $PythonPath "scripts/report_mcvr_v8_multiseed.py" `
    --seed12-evaluation $Seed12Evaluation `
    --seed43-evaluation $Seed43Evaluation `
    --seed48-evaluation $Seed48Evaluation `
    --output-dir $ReportDir
if ($LASTEXITCODE -ne 0) {
    throw "Multi-seed summary generation failed"
}

$TotalTimer.Stop()
$SummaryCsv = Join-Path $ReportDir "V8_MULTI_SEED_SUMMARY.csv"
$SummaryJson = Join-Path $ReportDir "V8_MULTI_SEED_SUMMARY.json"
$SummaryMd = Join-Path $ReportDir "V8_MULTI_SEED_SUMMARY.md"
foreach ($Path in @($SummaryCsv, $SummaryJson, $SummaryMd)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required summary artifact is missing: $Path"
    }
}

$Complete = [ordered]@{
    schema_version = "mcvr-v8-multiseed-complete-v1"
    status = "MCVR_V8_MULTI_SEED_COMPLETED"
    branch = $Branch
    git_head = $Head
    seed12_status = $Seed12.status
    seed48_status = $Seed48.status
    checkpoint_sha256 = [ordered]@{
        seed12 = $Seed12.checkpoint_sha256
        seed48 = $Seed48.checkpoint_sha256
    }
    validation_sha256 = [ordered]@{
        seed12 = $Seed12.validation_sha256
        seed48 = $Seed48.validation_sha256
    }
    runtime = [ordered]@{
        seed12_seconds = $Seed12.runtime_seconds
        seed48_seconds = $Seed48.runtime_seconds
        automation_total_seconds = $TotalTimer.Elapsed.TotalSeconds
    }
    exit_code = [ordered]@{
        seed12 = $Seed12.exit_code
        seed48 = $Seed48.exit_code
    }
    summary_sha256 = [ordered]@{
        csv = (Get-FileHash -LiteralPath $SummaryCsv -Algorithm SHA256).Hash.ToLower()
        json = (Get-FileHash -LiteralPath $SummaryJson -Algorithm SHA256).Hash.ToLower()
        md = (Get-FileHash -LiteralPath $SummaryMd -Algorithm SHA256).Hash.ToLower()
    }
    formal_test_records_read = 0
    frozen_holdout_records_read = 0
    completed_at = (Get-Date).ToUniversalTime().ToString("o")
}
$CompletePath = Join-Path $ReportDir "MULTISEED_COMPLETE.json"
$TemporaryComplete = "$CompletePath.tmp"
$Complete | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $TemporaryComplete -Encoding utf8
Move-Item -LiteralPath $TemporaryComplete -Destination $CompletePath -Force

Write-Host "MCVR_V8_MULTI_SEED_COMPLETED"
