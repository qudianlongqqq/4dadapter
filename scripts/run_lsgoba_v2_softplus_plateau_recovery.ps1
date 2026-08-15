$ErrorActionPreference = 'Stop'
$Root = 'E:\3dconformergenerationcode\4dadapter-lsgoba-v2-softplus-seed307'
$Report = Join-Path $Root 'reports\ecir_mvr\lsgoba_v2_softplus_seed307\training_plateau_evaluation'
$PipelineStatus = Join-Path $Report 'PIPELINE_STATUS.json'
$EtflowPython = 'E:\miniconda\envs\etflow-5080-v2\python.exe'
$ExternalPython = 'E:\miniconda\envs\external-validity\python.exe'
$Methods = @(
    'STEP12500_PROPOSAL', 'STEP12500_FINAL',
    'STEP15000_PROPOSAL', 'STEP15000_FINAL',
    'STEP17500_PROPOSAL', 'STEP17500_FINAL',
    'STEP20000_PROPOSAL', 'STEP20000_FINAL',
    'STEP22500_PROPOSAL', 'STEP22500_FINAL'
)

function Write-State([string]$Status, [string]$Stage, [string]$Message = '', [int]$Completed = 0) {
    $payload = [ordered]@{
        schema_version = 'lsgoba-v2-softplus-plateau-pipeline-v2'
        status = $Status
        stage = $Stage
        completed_methods = $Completed
        expected_methods = 10
        supervisor_pid = $PID
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
    $completed = 0
    foreach ($method in $Methods) {
        Write-State 'RUNNING' "EXTERNAL_$method" '' $completed
        $stdout = Join-Path $Report "EXTERNAL_${method}_STDOUT.log"
        $stderr = Join-Path $Report "EXTERNAL_${method}_STDERR.log"
        & $ExternalPython (Join-Path $Root 'scripts\evaluate_lsgoba_v2_softplus_training_plateau_external_chunked.py') chunk $method `
            1> $stdout 2> $stderr
        if ($LASTEXITCODE -ne 0) { throw "$method failed with exit code $LASTEXITCODE" }
        $completed += 1
    }
    Write-State 'RUNNING' 'EXTERNAL_MERGE' '' $completed
    & $ExternalPython (Join-Path $Root 'scripts\evaluate_lsgoba_v2_softplus_training_plateau_external_chunked.py') merge `
        1> (Join-Path $Report 'EXTERNAL_MERGE_STDOUT.log') `
        2> (Join-Path $Report 'EXTERNAL_MERGE_STDERR.log')
    if ($LASTEXITCODE -ne 0) { throw "external merge failed with exit code $LASTEXITCODE" }

    Write-State 'RUNNING' 'FIDELITY' '' $completed
    & $EtflowPython (Join-Path $Root 'scripts\evaluate_lsgoba_v2_softplus_training_plateau.py') fidelity `
        1> (Join-Path $Report 'FIDELITY_STDOUT.log') `
        2> (Join-Path $Report 'FIDELITY_STDERR.log')
    if ($LASTEXITCODE -ne 0) { throw "fidelity failed with exit code $LASTEXITCODE" }

    Write-State 'RUNNING' 'SUMMARIZE' '' $completed
    & $EtflowPython (Join-Path $Root 'scripts\evaluate_lsgoba_v2_softplus_training_plateau.py') summarize `
        1> (Join-Path $Report 'SUMMARIZE_STDOUT.log') `
        2> (Join-Path $Report 'SUMMARIZE_STDERR.log')
    if ($LASTEXITCODE -ne 0) { throw "summarize failed with exit code $LASTEXITCODE" }
    Write-State 'COMPLETE' 'COMPLETE' '' $completed
}
catch {
    Write-State 'FAILED' 'FAILED' $_.Exception.Message $completed
    throw
}
