param(
    [int]$WaitPid = 34460,
    [string]$CudaPython = 'E:\miniconda\envs\etflow-5080-v2\python.exe'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$report = Join-Path $repo 'reports\ecir_mvr\sixs_final_matched_ablation'
$status = Join-Path $report 'STATUS.json'
New-Item -ItemType Directory -Force -Path $report | Out-Null

function Write-PipelineStatus([string]$stage, [string]$state, [object]$waiting, [string]$cross, [string]$next, [int]$exitCode = 0, [string]$exception = $null) {
    $value = [ordered]@{
        schema_version = 'sixs-post-cross-upstream-ablation-pipeline-v1'
        status = $state
        stage = $stage
        pipeline_pid = $PID
        waiting_for_pid = $waiting
        cross_upstream_status = $cross
        next_automatic_stage = $next
        output_root = $report
        updated_at = (Get-Date).ToString('o')
        no_repeated_polling = $true
        no_busy_waiting = $true
        no_continuous_log_tail = $true
        EXIT_CODE = $exitCode
        EXCEPTION = $exception
        traceback_log = (Join-Path $report 'PIPELINE_TRACEBACK.txt')
    }
    $temporary = "$status.tmp.$PID"
    $encoded = $value | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText($temporary, $encoded, (New-Object System.Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporary -Destination $status -Force
}

try {
    Write-PipelineStatus 'STAGE_0_WAIT_CURRENT_CROSS_UPSTREAM' 'RUNNING' $WaitPid 'WAITING_FOR_CURRENT_SUPERVISOR' 'STAGE_1_VALIDATE_CROSS_UPSTREAM'
    $current = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
    if ($null -ne $current) {
        Wait-Process -Id $WaitPid
        $alreadyFinished = $false
    } else {
        $alreadyFinished = $true
    }
    Write-PipelineStatus 'STAGE_1_VALIDATE_CROSS_UPSTREAM' 'RUNNING' $null 'VALIDATING' 'STAGE_2_FREEZE_CROSS_UPSTREAM'
    $env:SIXS_CURRENT_PROCESS_ALREADY_FINISHED = if ($alreadyFinished) { 'YES' } else { 'NO' }
    & $CudaPython (Join-Path $PSScriptRoot 'run_sixs_post_cross_upstream_ablation_pipeline.py') run
    $workerExit = $LASTEXITCODE
    if ($workerExit -ne 0) { throw "Python pipeline exited with code $workerExit" }
    exit 0
} catch {
    $failureCode = if ($null -ne $workerExit -and $workerExit -ne 0) { $workerExit } else { 1 }
    Write-PipelineStatus 'PIPELINE_WRAPPER_FAILURE' 'FAIL' $null 'VALIDATION_OR_PIPELINE_FAILED' 'NONE' $failureCode $_.Exception.Message
    exit $failureCode
}
