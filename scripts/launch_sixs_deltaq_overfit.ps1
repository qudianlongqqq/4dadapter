param(
    [string]$Repo = 'E:\3dconformergenerationcode\4dadapter-lsgoba-musigma-reliability-factorial',
    [string]$Python = 'E:\miniconda\envs\etflow-5080-v2\python.exe'
)

$ErrorActionPreference = 'Stop'
$configPath = Join-Path $Repo 'configs\sixs_deltaq_prototype.json'
$workerScript = Join-Path $Repo 'scripts\run_sixs_deltaq_prototype.py'
$supervisorScript = Join-Path $Repo 'scripts\supervise_sixs_deltaq_overfit.ps1'
$runtime = Join-Path $Repo 'reports\ecir_mvr\sixs_deltaq_prototype\runtime'
$statusPath = Join-Path $runtime 'OVERFIT_STATUS.json'
$identityPath = Join-Path $runtime 'OVERFIT_IDENTITY.json'
$supervisorOut = Join-Path $runtime 'SUPERVISOR_STDOUT.log'
$supervisorErr = Join-Path $runtime 'SUPERVISOR_STDERR.log'

New-Item -ItemType Directory -Path $runtime -Force | Out-Null

function Write-AtomicJson([string]$Path, [hashtable]$Payload) {
    $temporary = "$Path.tmp.$PID"
    [System.IO.File]::WriteAllText(
        $temporary,
        ($Payload | ConvertTo-Json -Depth 8) + [Environment]::NewLine
    )
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

$branch = (& git -C $Repo branch --show-current).Trim()
$gitCommit = (& git -C $Repo rev-parse HEAD).Trim()
if ($branch -ne 'experiment/sixs-v2-source-conditioned-deltaq') {
    throw "unexpected branch: $branch"
}

& git -C $Repo diff --quiet HEAD -- `
    'configs/sixs_deltaq_prototype.json' `
    'etflow/ecir/source_conditioned_deltaq.py' `
    'scripts/run_sixs_deltaq_prototype.py' `
    'tests/test_source_conditioned_deltaq.py'
if ($LASTEXITCODE -ne 0) {
    throw 'frozen overfit scientific implementation differs from the failed-run identity'
}

$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$configSha = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
$scriptSha = (Get-FileHash -LiteralPath $workerScript -Algorithm SHA256).Hash.ToLowerInvariant()
$dataManifestPath = [string]$config.data.train_manifest
$dataManifestSha = (Get-FileHash -LiteralPath $dataManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($dataManifestSha -ne [string]$config.data.train_manifest_sha256) {
    throw 'TRAIN manifest SHA256 differs from the frozen configuration'
}

$cudaJson = & $Python -c "import json,torch; print(json.dumps({'available':torch.cuda.is_available(),'count':torch.cuda.device_count(),'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'cuda':torch.version.cuda}))"
$cuda = $cudaJson | ConvertFrom-Json
if (-not $cuda.available -or [int]$cuda.count -lt 1 -or [string]$cuda.name -ne 'NVIDIA GeForce RTX 5080') {
    throw "frozen GPU unavailable: $cudaJson"
}

$identity = @{
    schema_version = 'sixs-v2-deltaq-overfit-identity-v1'
    frozen_overfit_identity_match = 'YES'
    git_commit = $gitCommit
    branch = $branch
    config_sha256 = $configSha
    data_manifest_path = $dataManifestPath
    data_manifest_sha256 = $dataManifestSha
    script_sha256 = $scriptSha
    model_parameter_count = 766800
    seed = [int]$config.seed
    train_molecule_count = [int]$config.small_overfit.molecules
    optimizer = [string]$config.training.optimizer
    backbone_learning_rate = [double]$config.training.backbone_learning_rate
    head_learning_rate = [double]$config.training.head_learning_rate
    batch_molecules = [int]$config.small_overfit.molecules
    optimizer_steps = [int]$config.small_overfit.optimizer_steps
    deltaq_target = [string]$config.model.target
    sigma_semantics = [string]$config.model.sigma_semantics
    objective = [string]$config.objective.total
    cuda_available = [bool]$cuda.available
    cuda_device_name = [string]$cuda.name
    pytorch_cuda_version = [string]$cuda.cuda
    previous_failed_run = 'ENGINEERING_PROCESS_LIFETIME_FAILURE'
    scientific_result_from_failed_run = 'NONE'
    failed_run_used_for_model_selection = 'NO'
    scientific_configuration_changed = 'NO'
    formal_test_read = 'NO'
}
Write-AtomicJson $identityPath $identity

$eventName = "Local\SIXSDeltaQOverfitReady_$([Guid]::NewGuid().ToString('N'))"
$created = $false
$readyEvent = New-Object System.Threading.EventWaitHandle($false, [System.Threading.EventResetMode]::ManualReset, $eventName, [ref]$created)
try {
    $arguments = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $supervisorScript,
        '-Repo', $Repo, '-Python', $Python, '-ReadyEventName', $eventName,
        '-GitCommit', $gitCommit, '-ConfigSha256', $configSha,
        '-DataManifestSha256', $dataManifestSha, '-ScriptSha256', $scriptSha
    )
    $supervisor = Start-Process -FilePath 'powershell.exe' `
        -ArgumentList $arguments `
        -WorkingDirectory $Repo `
        -RedirectStandardOutput $supervisorOut `
        -RedirectStandardError $supervisorErr `
        -WindowStyle Hidden `
        -PassThru
    if (-not $readyEvent.WaitOne(30000)) {
        throw 'supervisor did not signal worker readiness within 30 seconds'
    }
    $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    $processes = Get-CimInstance Win32_Process -Filter "ProcessId=$($status.worker_pid) OR ProcessId=$($status.supervisor_pid)"
    $workerExists = @($processes | Where-Object ProcessId -eq ([int]$status.worker_pid)).Count -eq 1
    $supervisorExists = @($processes | Where-Object ProcessId -eq ([int]$status.supervisor_pid)).Count -eq 1
    $gpuProcesses = & nvidia-smi --query-compute-apps=pid,name --format=csv,noheader,nounits
    $gpuWorkerExists = [bool]($gpuProcesses | Select-String -SimpleMatch ([string]$status.worker_pid))
    $logExists = Test-Path -LiteralPath ([string]$status.log_path)
    $safe = ($status.status -eq 'RUNNING' -and $workerExists -and $supervisorExists -and $logExists)
    [ordered]@{
        DELTAQ_OVERFIT_STATUS = [string]$status.status
        FROZEN_OVERFIT_IDENTITY_MATCH = 'YES'
        WORKER_PID = [int]$status.worker_pid
        SUPERVISOR_PID = [int]$status.supervisor_pid
        WORKER_EXISTS = $workerExists
        SUPERVISOR_EXISTS = $supervisorExists
        GPU_WORKER_VISIBLE_AT_ONE_SHOT_CHECK = $gpuWorkerExists
        LOG_PATH = [string]$status.log_path
        STDERR_PATH = [string]$status.stderr_path
        STATUS_PATH = $statusPath
        CHECKPOINT_PATH = [string]$status.checkpoint_path
        CODEX_CAN_EXIT_SAFELY = if ($safe) { 'YES' } else { 'NO' }
        FORMAL_TEST_READ = 'NO'
    } | ConvertTo-Json -Depth 4
}
finally {
    $readyEvent.Dispose()
}
