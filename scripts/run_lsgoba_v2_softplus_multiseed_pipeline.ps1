param(
    [string]$Python = "E:\miniconda\envs\etflow-5080-v2\python.exe"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $PSScriptRoot "run_lsgoba_v2_softplus_multiseed.py"
$Report = Join-Path $Repo "reports\ecir_mvr\lsgoba_v2_softplus_multiseed"
$Status = Join-Path $Report "PIPELINE_STATUS.json"
New-Item -ItemType Directory -Force -Path $Report | Out-Null

function Write-PipelineStatus {
    param([string]$Stage, [string]$State, [int]$Seed = 0, [string]$ErrorMessage = "")
    $payload = [ordered]@{
        schema_version = "lsgoba-v2-softplus-multiseed-pipeline-status-v1"
        status = $State
        stage = $Stage
        active_seed = $(if ($Seed -eq 0) { $null } else { $Seed })
        supervisor_pid = $PID
        heartbeat = [DateTimeOffset]::Now.ToString("o")
        run_order = @(331, 353)
        seed307_retrained = $false
        formal_test_records_read = 0
        frozen_holdout_records_read = 0
        xtb_started = $false
        orca_started = $false
        docking_started = $false
    }
    if ($ErrorMessage) { $payload.error = $ErrorMessage }
    $temporary = "$Status.tmp.$PID"
    [IO.File]::WriteAllText($temporary, ($payload | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $Status -Force
}

function Invoke-Phase {
    param([int]$Seed, [string]$Phase)
    $seedReport = Join-Path $Report "seed$Seed"
    New-Item -ItemType Directory -Force -Path $seedReport | Out-Null
    $stdout = Join-Path $seedReport ("{0}_STDOUT.log" -f $Phase.ToUpperInvariant())
    $stderr = Join-Path $seedReport ("{0}_STDERR.log" -f $Phase.ToUpperInvariant())
    Write-PipelineStatus -Stage ("SEED{0}_{1}" -f $Seed, $Phase.ToUpperInvariant()) -State "RUNNING" -Seed $Seed
    $process = Start-Process -FilePath $Python -ArgumentList @($Runner, $Phase, "--seed", "$Seed") `
        -WorkingDirectory $Repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
        -WindowStyle Hidden -PassThru -Wait
    if ($process.ExitCode -ne 0) {
        throw "seed$Seed $Phase failed with exit code $($process.ExitCode); see $stderr"
    }
}

try {
    $seed331Status = Join-Path $Report "seed331\STATUS.json"
    $seed331PreflightPassed = $false
    if (Test-Path -LiteralPath $seed331Status) {
        $current = Get-Content -Raw -LiteralPath $seed331Status | ConvertFrom-Json
        $seed331PreflightPassed = $current.status -eq "PASS" -and $current.stage -eq "PREFLIGHT"
    }
    if (-not $seed331PreflightPassed) { Invoke-Phase -Seed 331 -Phase "preflight" }
    Invoke-Phase -Seed 331 -Phase "train"
    Invoke-Phase -Seed 331 -Phase "verify"

    Invoke-Phase -Seed 353 -Phase "preflight"
    Invoke-Phase -Seed 353 -Phase "train"
    Invoke-Phase -Seed 353 -Phase "verify"

    Write-PipelineStatus -Stage "REPLICATE_TRAINING_COMPLETE" -State "PASS"
}
catch {
    Write-PipelineStatus -Stage "FAILED" -State "FAILED" -ErrorMessage $_.Exception.Message
    throw
}
