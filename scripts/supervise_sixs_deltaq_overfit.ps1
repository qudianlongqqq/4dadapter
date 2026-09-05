param(
    [Parameter(Mandatory = $true)][string]$Repo,
    [Parameter(Mandatory = $true)][string]$Python,
    [string]$ReadyEventName = '',
    [Parameter(Mandatory = $true)][string]$GitCommit,
    [Parameter(Mandatory = $true)][string]$ConfigSha256,
    [Parameter(Mandatory = $true)][string]$DataManifestSha256,
    [Parameter(Mandatory = $true)][string]$ScriptSha256,
    [ValidateSet('production', 'engineering-smoke')][string]$WorkerMode = 'production'
)

$ErrorActionPreference = 'Stop'
$isSmoke = $WorkerMode -eq 'engineering-smoke'
$runtime = if ($isSmoke) {
    Join-Path $Repo 'reports\ecir_mvr\sixs_deltaq_prototype\runtime\engineering_smoke'
} else {
    Join-Path $Repo 'reports\ecir_mvr\sixs_deltaq_prototype\runtime'
}
$statusPath = Join-Path $runtime 'OVERFIT_STATUS.json'
$logPath = Join-Path $runtime 'OVERFIT_STDOUT.log'
$stderrPath = Join-Path $runtime 'OVERFIT_STDERR.log'
$checkpointPath = if ($isSmoke) {
    Join-Path $Repo 'artifacts\ecir_mvr\sixs_deltaq_prototype\engineering_smoke\SMALL_OVERFIT_CHECKPOINT.pt'
} else {
    Join-Path $Repo 'artifacts\ecir_mvr\sixs_deltaq_prototype\SMALL_OVERFIT_CHECKPOINT.pt'
}
$resultPath = if ($isSmoke) {
    Join-Path $runtime '07_SMALL_OVERFIT_RESULTS.csv'
} else {
    Join-Path $Repo 'reports\ecir_mvr\sixs_deltaq_prototype\07_SMALL_OVERFIT_RESULTS.csv'
}
$completedPath = Join-Path $runtime 'OVERFIT_COMPLETED.json'
$failedPath = Join-Path $runtime 'OVERFIT_FAILED.json'
$workerScript = if ($isSmoke) {
    Join-Path $Repo 'scripts\smoke_sixs_deltaq_overfit_engineering.py'
} else {
    Join-Path $Repo 'scripts\run_sixs_deltaq_prototype.py'
}
$nativeGuard = Join-Path $Repo 'scripts\run_sixs_deltaq_overfit_native_guard.py'
$workerArguments = if ($isSmoke) {
    @($nativeGuard, $workerScript)
} else {
    @($nativeGuard, $workerScript, 'overfit')
}

New-Item -ItemType Directory -Path $runtime -Force | Out-Null

function Write-AtomicJson([string]$Path, [hashtable]$Payload) {
    $temporary = "$Path.tmp.$PID"
    $json = $Payload | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText($temporary, $json + [Environment]::NewLine)
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

$started = [DateTimeOffset]::Now.ToString('o')
$readyEvent = if ($ReadyEventName) {
    [System.Threading.EventWaitHandle]::OpenExisting($ReadyEventName)
} else {
    $null
}
$worker = $null

try {
    # Launching an environment's python.exe directly does not activate the Conda
    # DLL search path.  Use an explicit, minimal PATH so native libraries cannot
    # be resolved from an unrelated interactive shell environment.
    $environmentRoot = Split-Path -Parent $Python
    $windowsRoot = [Environment]::GetFolderPath('Windows')
    $gitDirectory = Split-Path -Parent (Get-Command git.exe -ErrorAction Stop).Source
    $cleanPath = (@(
        $environmentRoot,
        (Join-Path $environmentRoot 'Library\mingw-w64\bin'),
        (Join-Path $environmentRoot 'Library\usr\bin'),
        (Join-Path $environmentRoot 'Library\bin'),
        (Join-Path $environmentRoot 'Scripts'),
        $gitDirectory,
        (Join-Path $windowsRoot 'System32'),
        $windowsRoot,
        (Join-Path $windowsRoot 'System32\Wbem'),
        (Join-Path $windowsRoot 'System32\WindowsPowerShell\v1.0')
    ) -join ';')

    # Use ProcessStartInfo directly.  Start-Process returned a stale zero
    # ExitCode for both a native Windows crash and a Python traceback in the
    # failed-run forensic reproduction.
    $argumentText = ($workerArguments | ForEach-Object {
        '"' + ([string]$_).Replace('"', '\"') + '"'
    }) -join ' '
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $Python
    $startInfo.Arguments = $argumentText
    $startInfo.WorkingDirectory = $Repo
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.EnvironmentVariables['CONDA_PREFIX'] = $environmentRoot
    $startInfo.EnvironmentVariables['CONDA_DEFAULT_ENV'] = Split-Path -Leaf $environmentRoot
    $startInfo.EnvironmentVariables['PYTHONNOUSERSITE'] = '1'
    $startInfo.EnvironmentVariables['PATH'] = $cleanPath
    [System.IO.File]::WriteAllText($logPath, '')
    [System.IO.File]::WriteAllText($stderrPath, '')
    $worker = New-Object System.Diagnostics.Process
    $worker.StartInfo = $startInfo
    if (-not $worker.Start()) { throw 'native Python worker failed to start' }
    $stdoutTask = $worker.StandardOutput.ReadToEndAsync()
    $stderrTask = $worker.StandardError.ReadToEndAsync()

    $running = @{
        schema_version = 'sixs-v2-deltaq-overfit-runtime-v1'
        stage = if ($isSmoke) { 'ENGINEERING_SMOKE_GPU' } else { 'SMALL_OVERFIT_GPU' }
        status = 'RUNNING'
        worker_pid = [int]$worker.Id
        supervisor_pid = [int]$PID
        start_time = $started
        end_time = $null
        exit_code = $null
        log_path = $logPath
        stderr_path = $stderrPath
        checkpoint_path = $checkpointPath
        result_path = $resultPath
        config_sha256 = $ConfigSha256
        data_manifest_sha256 = $DataManifestSha256
        script_sha256 = $ScriptSha256
        git_commit = $GitCommit
        model_parameter_count = 766800
        seed = 307
        train_molecule_count = if ($isSmoke) { 2 } else { 32 }
        worker_mode = $WorkerMode
        dll_search_environment = 'CLEAN_CONDA_PREFIX'
        native_runtime_guard = $nativeGuard
        training_entrypoint = $workerScript
        local_restart_safe_supervisor = 'YES'
        previous_failed_run = 'ENGINEERING_PROCESS_LIFETIME_FAILURE'
        scientific_result_from_failed_run = 'NONE'
        failed_run_used_for_model_selection = 'NO'
        scientific_configuration_changed = 'NO'
        formal_test_read = 'NO'
    }
    Write-AtomicJson $statusPath $running
    if ($null -ne $readyEvent) { [void]$readyEvent.Set() }

    $worker.WaitForExit()
    $stdoutTask.Wait()
    $stderrTask.Wait()
    $worker.Refresh()
    $exitCode = [int]$worker.ExitCode
    [System.IO.File]::WriteAllText($logPath, $stdoutTask.Result)
    [System.IO.File]::WriteAllText($stderrPath, $stderrTask.Result)
    $ended = [DateTimeOffset]::Now.ToString('o')
    $checkpointExists = Test-Path -LiteralPath $checkpointPath
    $resultExists = Test-Path -LiteralPath $resultPath
    $success = ($exitCode -eq 0 -and $checkpointExists -and $resultExists)
    $terminal = $running.Clone()
    $terminal.status = if ($success) { 'COMPLETED' } else { 'FAILED' }
    $terminal.stage = if ($success) {
        if ($isSmoke) { 'ENGINEERING_SMOKE_COMPLETE' } else { 'SMALL_OVERFIT_COMPLETE' }
    } else {
        if ($isSmoke) { 'ENGINEERING_SMOKE_FAILED' } else { 'SMALL_OVERFIT_FAILED' }
    }
    $terminal.end_time = $ended
    $terminal.exit_code = $exitCode
    $terminal.checkpoint_exists = $checkpointExists
    $terminal.result_exists = $resultExists
    $terminal.completion_marker = if ($success) { $completedPath } else { $failedPath }
    Write-AtomicJson $statusPath $terminal
    Write-AtomicJson $terminal.completion_marker $terminal
    if (-not $success) {
        # Preserve a nonzero supervisor exit even when a native worker crash is
        # misreported as zero by the platform. Missing required artifacts are
        # always an engineering failure.
        exit $(if ($exitCode -ne 0) { $exitCode } else { 2 })
    }
}
catch {
    $failed = @{
        schema_version = 'sixs-v2-deltaq-overfit-runtime-v1'
        stage = 'SUPERVISOR_FAILURE'
        status = 'FAILED'
        worker_pid = if ($null -ne $worker) { [int]$worker.Id } else { $null }
        supervisor_pid = [int]$PID
        start_time = $started
        end_time = [DateTimeOffset]::Now.ToString('o')
        exit_code = $null
        error = $_.Exception.Message
        log_path = $logPath
        stderr_path = $stderrPath
        checkpoint_path = $checkpointPath
        result_path = $resultPath
        config_sha256 = $ConfigSha256
        data_manifest_sha256 = $DataManifestSha256
        script_sha256 = $ScriptSha256
        git_commit = $GitCommit
        model_parameter_count = 766800
        seed = 307
        train_molecule_count = if ($isSmoke) { 2 } else { 32 }
        worker_mode = $WorkerMode
        dll_search_environment = 'CLEAN_CONDA_PREFIX'
        native_runtime_guard = $nativeGuard
        training_entrypoint = $workerScript
        local_restart_safe_supervisor = 'YES'
        scientific_configuration_changed = 'NO'
        formal_test_read = 'NO'
    }
    Write-AtomicJson $statusPath $failed
    Write-AtomicJson $failedPath $failed
    try { [void]$readyEvent.Set() } catch {}
    exit 1
}
finally {
    if ($null -ne $readyEvent) { $readyEvent.Dispose() }
}
