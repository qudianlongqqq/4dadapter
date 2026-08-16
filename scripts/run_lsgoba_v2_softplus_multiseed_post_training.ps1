param(
    [Parameter(Mandatory = $true)][int]$TrainingSupervisorPid,
    [string]$Python = "E:\miniconda\envs\etflow-5080-v2\python.exe"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Report = Join-Path $Repo "reports\ecir_mvr\lsgoba_v2_softplus_multiseed"
$TrainingStatus = Join-Path $Report "PIPELINE_STATUS.json"
$Status = Join-Path $Report "POST_TRAINING_STATUS.json"
$Evaluator = Join-Path $PSScriptRoot "evaluate_lsgoba_v2_softplus_multiseed.py"
$External = Join-Path $PSScriptRoot "evaluate_lsgoba_v2_softplus_multiseed_external.py"
New-Item -ItemType Directory -Force -Path $Report | Out-Null

function Write-Status {
    param([string]$Stage, [string]$State, [string]$ErrorMessage = "")
    $payload = [ordered]@{
        schema_version = "lsgoba-v2-softplus-multiseed-post-training-status-v1"
        status = $State
        stage = $Stage
        worker_pid = $PID
        training_supervisor_pid = $TrainingSupervisorPid
        heartbeat = [DateTimeOffset]::Now.ToString("o")
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

function Invoke-Evaluator {
    param([string]$Name, [string]$Script, [string[]]$Arguments)
    $stdout = Join-Path $Report ("POST_{0}_STDOUT.log" -f $Name)
    $stderr = Join-Path $Report ("POST_{0}_STDERR.log" -f $Name)
    Write-Status -Stage $Name -State "RUNNING"
    $process = Start-Process -FilePath $Python -ArgumentList (@($Script) + $Arguments) `
        -WorkingDirectory $Repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
        -WindowStyle Hidden -PassThru -Wait
    if ($process.ExitCode -ne 0) {
        throw "$Name failed with exit code $($process.ExitCode); see $stderr"
    }
}

try {
    Write-Status -Stage "WAITING_FOR_REPLICATE_TRAINING" -State "WAITING"
    Wait-Process -Id $TrainingSupervisorPid
    $training = Get-Content -Raw -LiteralPath $TrainingStatus | ConvertFrom-Json
    if ($training.status -ne "PASS" -or $training.stage -ne "REPLICATE_TRAINING_COMPLETE") {
        throw "replicate training pipeline did not complete successfully"
    }
    Invoke-Evaluator -Name "PREFLIGHT" -Script $Evaluator -Arguments @("preflight")
    Invoke-Evaluator -Name "COORDINATES" -Script $Evaluator -Arguments @("generate")
    Invoke-Evaluator -Name "EXTERNAL_ENDPOINTS" -Script $External -Arguments @()
    Invoke-Evaluator -Name "FIDELITY" -Script $Evaluator -Arguments @("fidelity")
    Invoke-Evaluator -Name "SUMMARY" -Script $Evaluator -Arguments @("summarize")
    Write-Status -Stage "COMPLETE" -State "PASS"
}
catch {
    Write-Status -Stage "FAILED" -State "FAILED" -ErrorMessage $_.Exception.Message
    throw
}
