param(
    [Parameter(Mandatory=$true)][string]$Repo,
    [Parameter(Mandatory=$true)][string]$Python,
    [string]$ReadyEventName='',
    [int]$OverfitWorkerPid=42000,
    [int]$OverfitSupervisorPid=2152,
    [Parameter(Mandatory=$true)][string]$GitCommit,
    [Parameter(Mandatory=$true)][string]$ConfigSha256,
    [Parameter(Mandatory=$true)][string]$DataManifestSha256
)

$ErrorActionPreference='Stop'
$runtime=Join-Path $Repo 'reports\ecir_mvr\sixs_deltaq_single_seed_pilot\runtime'
$statusPath=Join-Path $runtime 'PIPELINE_STATUS.json'
$stageRunner=Join-Path $Repo 'scripts\run_sixs_deltaq_single_seed_stage.py'
$nativeGuard=Join-Path $Repo 'scripts\run_sixs_deltaq_overfit_native_guard.py'
$trainRunner=Join-Path $Repo 'scripts\run_sixs_deltaq_prototype.py'
$pilotCheckpoint=Join-Path $Repo 'artifacts\ecir_mvr\sixs_deltaq_single_seed_pilot\DELTAQ_SEED307_FULL.pt'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null

function Write-AtomicJson([string]$Path,[hashtable]$Payload) {
    $tmp="$Path.tmp.$PID"; [IO.File]::WriteAllText($tmp,($Payload|ConvertTo-Json -Depth 12)+[Environment]::NewLine); Move-Item -LiteralPath $tmp -Destination $Path -Force
}

$state=@{
    schema_version='sixs-v2-deltaq-single-seed-pipeline-runtime-v1'; pipeline_status='WAITING_FOR_OVERFIT'; current_stage='00_OVERFIT_GATE'
    worker_pid=$OverfitWorkerPid; supervisor_pid=[int]$PID; overfit_supervisor_pid=$OverfitSupervisorPid
    stage_start_time=[DateTimeOffset]::Now.ToString('o'); stage_end_time=$null; stage_exit_code=$null
    git_commit=$GitCommit; config_sha256=$ConfigSha256; data_manifest_sha256=$DataManifestSha256; seed=307
    checkpoint_path=$pilotCheckpoint; stdout_path=(Join-Path $runtime '00_OVERFIT_GATE_STDOUT.log'); stderr_path=(Join-Path $runtime '00_OVERFIT_GATE_STDERR.log')
    failure_reason=$null; formal_test_read='NO'; avg_old_final_used_for_tuning='NO'; no_repeated_polling='YES'
}
Write-AtomicJson $statusPath $state
$ready=if($ReadyEventName){[Threading.EventWaitHandle]::OpenExisting($ReadyEventName)}else{$null}
if($null-ne $ready){[void]$ready.Set();$ready.Dispose()}

function Test-Reusable([string]$Stage) {
    $path=Join-Path $runtime "${Stage}_COMPLETED.json"
    if(-not(Test-Path -LiteralPath $path)){return $false}
    try {
        $m=Get-Content -LiteralPath $path -Raw|ConvertFrom-Json
        if($m.status-ne'COMPLETED'-or$m.pipeline_identity_sha256-ne$ConfigSha256){return $false}
        foreach($p in $m.outputs.PSObject.Properties){if(-not(Test-Path -LiteralPath $p.Name)){return $false}}
        return $true
    } catch {return $false}
}

function Start-Stage([string]$Stage,[string[]]$Arguments) {
    if(Test-Reusable $Stage){return}
    $out=Join-Path $runtime "${Stage}_STDOUT.log"; $err=Join-Path $runtime "${Stage}_STDERR.log"
    [IO.File]::WriteAllText($out,'');[IO.File]::WriteAllText($err,'')
    $state.pipeline_status='RUNNING';$state.current_stage=$Stage;$state.stage_start_time=[DateTimeOffset]::Now.ToString('o');$state.stage_end_time=$null;$state.stage_exit_code=$null;$state.stdout_path=$out;$state.stderr_path=$err;$state.failure_reason=$null
    $envRoot=Split-Path -Parent $Python;$windowsRoot=[Environment]::GetFolderPath('Windows');$gitDir=Split-Path -Parent (Get-Command git.exe).Source
    $cleanPath=@($envRoot,(Join-Path $envRoot 'Library\mingw-w64\bin'),(Join-Path $envRoot 'Library\usr\bin'),(Join-Path $envRoot 'Library\bin'),(Join-Path $envRoot 'Scripts'),$gitDir,(Join-Path $windowsRoot 'System32'),$windowsRoot,(Join-Path $windowsRoot 'System32\Wbem'),(Join-Path $windowsRoot 'System32\WindowsPowerShell\v1.0'))-join';'
    $si=New-Object Diagnostics.ProcessStartInfo;$si.FileName=$Python;$si.Arguments=($Arguments|ForEach-Object{'"'+([string]$_).Replace('"','\"')+'"'})-join' ';$si.WorkingDirectory=$Repo;$si.UseShellExecute=$false;$si.CreateNoWindow=$true;$si.RedirectStandardOutput=$true;$si.RedirectStandardError=$true
    $si.EnvironmentVariables['PATH']=$cleanPath;$si.EnvironmentVariables['CONDA_PREFIX']=$envRoot;$si.EnvironmentVariables['PYTHONNOUSERSITE']='1'
    $p=New-Object Diagnostics.Process;$p.StartInfo=$si;if(-not$p.Start()){throw "failed to start $Stage"};$state.worker_pid=[int]$p.Id;Write-AtomicJson $statusPath $state
    $ot=$p.StandardOutput.ReadToEndAsync();$et=$p.StandardError.ReadToEndAsync();$p.WaitForExit();$ot.Wait();$et.Wait();$p.Refresh();[IO.File]::WriteAllText($out,$ot.Result);[IO.File]::WriteAllText($err,$et.Result)
    $state.stage_end_time=[DateTimeOffset]::Now.ToString('o');$state.stage_exit_code=[int]$p.ExitCode;Write-AtomicJson $statusPath $state
    if($p.ExitCode-ne0){throw "stage $Stage exited $($p.ExitCode)"}
    if(-not(Test-Reusable $Stage)){throw "stage $Stage lacks a valid completion marker"}
}

try {
    # Exactly one process lookup followed by a blocking kernel wait; no polling.
    $existing=Get-Process -Id $OverfitWorkerPid -ErrorAction SilentlyContinue
    if($null-ne$existing){$existing.WaitForExit()}
    Start-Stage '00_OVERFIT_GATE' @($stageRunner,'gate')
    Start-Stage '01_FULL_TRAIN' @($nativeGuard,$trainRunner,'train')
    # Materialize the canonical pilot checkpoint and training summaries.
    if(-not(Test-Reusable '01_FULL_TRAIN')){throw 'unreachable marker check'}
    # The native training worker writes its own prototype marker; finalize is
    # idempotent and creates the pipeline marker expected above.
}
catch {
    # If training completed but its pipeline marker is not yet materialized,
    # allow one deterministic finalization before declaring failure.
    if($state.current_stage-eq'01_FULL_TRAIN'-and(Test-Path -LiteralPath (Join-Path $Repo 'artifacts\ecir_mvr\sixs_deltaq_prototype\STEP17500_CHECKPOINT.pt'))){
        try { Start-Stage '01_FULL_TRAIN' @($stageRunner,'finalize-train') } catch {}
    }
    if(-not(Test-Reusable '01_FULL_TRAIN')-or$state.current_stage-eq'00_OVERFIT_GATE'){
        $state.pipeline_status='FAILED';$state.stage_end_time=[DateTimeOffset]::Now.ToString('o');$state.failure_reason=$_.Exception.Message;Write-AtomicJson $statusPath $state;exit 1
    }
}

try {
    # The training process and finalizer share a stage; at a fresh run there is
    # no marker until this call.  On restart the final checkpoint is reused.
    if(-not(Test-Reusable '01_FULL_TRAIN')){Start-Stage '01_FULL_TRAIN' @($stageRunner,'finalize-train')}
    Start-Stage '02_ETFLOW_EVAL' @($stageRunner,'etflow')
    Start-Stage '03_AVGFLOW_EVAL' @($stageRunner,'avgflow')
    Start-Stage '04_DITMC_EVAL' @($stageRunner,'ditmc')
    Start-Stage '05_XTB_EVAL' @($stageRunner,'xtb')
    Start-Stage '06_AGGREGATION' @($stageRunner,'aggregate')
    $state.pipeline_status='COMPLETED';$state.current_stage='COMPLETED';$state.worker_pid=$null;$state.stage_end_time=[DateTimeOffset]::Now.ToString('o');$state.stage_exit_code=0;$state.failure_reason=$null;Write-AtomicJson $statusPath $state
}
catch {
    $state.pipeline_status='FAILED';$state.stage_end_time=[DateTimeOffset]::Now.ToString('o');$state.failure_reason=$_.Exception.Message;Write-AtomicJson $statusPath $state;exit 1
}
