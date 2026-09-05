param(
    [Parameter(Mandatory = $true)]
    [int]$EvalPid
)

$ErrorActionPreference = 'Stop'
$repo = 'E:\3dconformergenerationcode\4dadapter-lsgoba-musigma-reliability-factorial'
$report = Join-Path $repo 'reports\ecir_mvr\sixs_musigma_reliability_factorial_cuda'
$armRoot = Join-Path $repo 'artifacts\ecir_mvr\sixs_musigma_reliability_factorial_cuda\J0_R1'
$supervisorStatus = Join-Path $report 'J0_R1_CPU_EVAL_SUPERVISOR_STATUS.json'

function Write-SupervisorStatus {
    param([hashtable]$Payload)
    $Payload['updated_at'] = (Get-Date).ToString('o')
    $Payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $supervisorStatus -Encoding utf8
}

try {
    Write-SupervisorStatus @{ status = 'WAITING'; eval_pid = $EvalPid; wait_mode = 'BLOCKING_PROCESS_WAIT' }
    $evalProcess = Get-Process -Id $EvalPid -ErrorAction Stop
    $evalProcess.WaitForExit()
    $exitCode = $evalProcess.ExitCode
    if ($exitCode -ne 0) {
        throw "J0-R1 CPU evaluation exited with code $exitCode"
    }

    $done = Join-Path $armRoot 'DONE.json'
    $result = Join-Path $armRoot 'dev_evaluation\RESULT.json'
    if (-not (Test-Path -LiteralPath $done) -or -not (Test-Path -LiteralPath $result)) {
        throw 'J0-R1 evaluation exited without frozen DONE.json and RESULT.json'
    }

    $existing = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python' -and
        $_.ExecutablePath -eq 'E:\miniconda\envs\etflow-5080-v2\python.exe' -and
        $_.CommandLine -like '*run_sixs_musigma_reliability_factorial.py*pipeline*'
    })
    if ($existing.Count -gt 0) {
        throw 'A CUDA factorial pipeline is already running'
    }

    $env:SIXS_FACTORIAL_RUN_NAMESPACE = 'sixs_musigma_reliability_factorial_cuda'
    $env:SIXS_FACTORIAL_DEVICE = 'cuda'
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $cudaProcess = Start-Process `
        -FilePath 'E:\miniconda\envs\etflow-5080-v2\python.exe' `
        -ArgumentList @('-u', 'scripts\run_sixs_musigma_reliability_factorial.py', 'pipeline') `
        -WorkingDirectory $repo `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $report "PIPELINE_POST_EVAL_${stamp}_STDOUT.log") `
        -RedirectStandardError (Join-Path $report "PIPELINE_POST_EVAL_${stamp}_STDERR.log") `
        -PassThru
    Write-SupervisorStatus @{
        status = 'CUDA_RESUMED'
        eval_pid = $EvalPid
        eval_exit_code = $exitCode
        cuda_pid = $cudaProcess.Id
        resume_boundary = 'J0-R1_DONE_TO_J1-R0'
    }
}
catch {
    Write-SupervisorStatus @{ status = 'FAIL_CLOSED'; eval_pid = $EvalPid; error = $_.Exception.Message }
    exit 1
}
