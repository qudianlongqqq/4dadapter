param(
    [Parameter(Mandatory = $true)]
    [int]$GenerationPid
)

$ErrorActionPreference = 'Stop'
$Root = 'E:\3dconformergenerationcode\4dadapter-lsgoba-v2-softplus-seed307'
$Report = Join-Path $Root 'reports\ecir_mvr\lsgoba_v2_softplus_seed307\training_plateau_evaluation'
$EvalStatus = Join-Path $Report 'STATUS.json'
$PipelineStatus = Join-Path $Report 'PIPELINE_STATUS.json'
$EtflowPython = 'E:\miniconda\envs\etflow-5080-v2\python.exe'
$ExternalPython = 'E:\miniconda\envs\external-validity\python.exe'

function Write-PipelineStatus([string]$Status, [string]$Stage, [string]$Message = '') {
    $payload = [ordered]@{
        schema_version = 'lsgoba-v2-softplus-plateau-pipeline-v1'
        status = $Status
        stage = $Stage
        supervisor_pid = $PID
        generation_pid = $GenerationPid
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
    Write-PipelineStatus 'RUNNING' 'WAIT_GENERATION'
    if (Get-Process -Id $GenerationPid -ErrorAction SilentlyContinue) {
        Wait-Process -Id $GenerationPid
    }
    $state = Get-Content -LiteralPath $EvalStatus -Raw | ConvertFrom-Json
    if ($state.status -ne 'PASS' -or $state.stage -ne 'COORDINATES_FROZEN') {
        throw "coordinate generation did not freeze successfully: $($state.status)/$($state.stage)"
    }

    Write-PipelineStatus 'RUNNING' 'EXTERNAL_ENDPOINTS'
    & $ExternalPython (Join-Path $Root 'scripts\evaluate_lsgoba_v2_softplus_training_plateau_external.py') `
        1> (Join-Path $Report 'EXTERNAL_STDOUT.log') `
        2> (Join-Path $Report 'EXTERNAL_STDERR.log')
    if ($LASTEXITCODE -ne 0) { throw "external endpoint evaluator failed with exit code $LASTEXITCODE" }

    Write-PipelineStatus 'RUNNING' 'FIDELITY'
    & $EtflowPython (Join-Path $Root 'scripts\evaluate_lsgoba_v2_softplus_training_plateau.py') fidelity `
        1> (Join-Path $Report 'FIDELITY_STDOUT.log') `
        2> (Join-Path $Report 'FIDELITY_STDERR.log')
    if ($LASTEXITCODE -ne 0) { throw "fidelity evaluator failed with exit code $LASTEXITCODE" }

    Write-PipelineStatus 'RUNNING' 'SUMMARIZE'
    & $EtflowPython (Join-Path $Root 'scripts\evaluate_lsgoba_v2_softplus_training_plateau.py') summarize `
        1> (Join-Path $Report 'SUMMARIZE_STDOUT.log') `
        2> (Join-Path $Report 'SUMMARIZE_STDERR.log')
    if ($LASTEXITCODE -ne 0) { throw "summarizer failed with exit code $LASTEXITCODE" }

    Write-PipelineStatus 'COMPLETE' 'COMPLETE'
}
catch {
    Write-PipelineStatus 'FAILED' 'FAILED' $_.Exception.Message
    throw
}
