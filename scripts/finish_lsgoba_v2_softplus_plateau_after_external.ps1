param(
    [Parameter(Mandatory = $true)]
    [int]$ExternalPid
)

$ErrorActionPreference = 'Stop'
$Root = 'E:\3dconformergenerationcode\4dadapter-lsgoba-v2-softplus-seed307'
$Report = Join-Path $Root 'reports\ecir_mvr\lsgoba_v2_softplus_seed307\training_plateau_evaluation'
$Artifact = Join-Path $Root 'artifacts\ecir_mvr\lsgoba_v2_softplus_seed307\training_plateau_evaluation'
$PipelineStatus = Join-Path $Report 'PIPELINE_STATUS.json'
$EtflowPython = 'E:\miniconda\envs\etflow-5080-v2\python.exe'

function Write-State([string]$Status, [string]$Stage, [string]$Message = '') {
    $payload = [ordered]@{
        schema_version = 'lsgoba-v2-softplus-plateau-direct-external-pipeline-v1'
        status = $Status
        stage = $Stage
        supervisor_pid = $PID
        external_pid = $ExternalPid
        updated_at = (Get-Date).ToString('o')
        formal_test_records_read = 0
        frozen_holdout_records_read = 0
        xtb_started = $false
        training_started = $false
        message = $Message
    }
    $temp = "$PipelineStatus.tmp.$PID"
    $payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temp -Encoding utf8
    Move-Item -LiteralPath $temp -Destination $PipelineStatus -Force
}

try {
    Write-State 'RUNNING' 'WAIT_DIRECT_EXTERNAL'
    if (Get-Process -Id $ExternalPid -ErrorAction SilentlyContinue) {
        Wait-Process -Id $ExternalPid
    }
    $completionPath = Join-Path $Artifact 'ENDPOINT_COMPLETION.json'
    $completion = Get-Content -LiteralPath $completionPath -Raw | ConvertFrom-Json
    if ($completion.status -ne 'COMPLETE' -or $completion.methods.Count -ne 10) {
        throw 'direct external endpoint evaluation did not complete all 10 methods'
    }
    Write-State 'RUNNING' 'FIDELITY'
    & $EtflowPython (Join-Path $Root 'scripts\evaluate_lsgoba_v2_softplus_training_plateau.py') fidelity `
        1> (Join-Path $Report 'FIDELITY_STDOUT.log') `
        2> (Join-Path $Report 'FIDELITY_STDERR.log')
    if ($LASTEXITCODE -ne 0) { throw "fidelity failed with exit code $LASTEXITCODE" }
    Write-State 'RUNNING' 'SUMMARIZE'
    & $EtflowPython (Join-Path $Root 'scripts\evaluate_lsgoba_v2_softplus_training_plateau.py') summarize `
        1> (Join-Path $Report 'SUMMARIZE_STDOUT.log') `
        2> (Join-Path $Report 'SUMMARIZE_STDERR.log')
    if ($LASTEXITCODE -ne 0) { throw "summarize failed with exit code $LASTEXITCODE" }
    Write-State 'COMPLETE' 'COMPLETE'
}
catch {
    Write-State 'FAILED' 'FAILED' $_.Exception.Message
    throw
}
