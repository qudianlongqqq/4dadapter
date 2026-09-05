param(
    [string]$Repo='E:\3dconformergenerationcode\4dadapter-lsgoba-musigma-reliability-factorial',
    [string]$Python='E:\miniconda\envs\etflow-5080-v2\python.exe'
)
$ErrorActionPreference='Stop'
$config=Join-Path $Repo 'configs\sixs_deltaq_single_seed_pilot.json';$prototype=Join-Path $Repo 'configs\sixs_deltaq_prototype.json';$supervisorScript=Join-Path $Repo 'scripts\supervise_sixs_deltaq_single_seed_pipeline.ps1'
$runtime=Join-Path $Repo 'reports\ecir_mvr\sixs_deltaq_single_seed_pilot\runtime';New-Item -ItemType Directory -Path $runtime -Force|Out-Null
$statusPath=Join-Path $runtime 'PIPELINE_STATUS.json';$out=Join-Path $runtime 'PIPELINE_STDOUT.log';$err=Join-Path $runtime 'PIPELINE_STDERR.log'
$branch=(&git -C $Repo branch --show-current).Trim();if($branch-ne'experiment/sixs-v2-source-conditioned-deltaq'){throw "unexpected branch $branch"}
$commit=(&git -C $Repo rev-parse HEAD).Trim();$configSha=(Get-FileHash $config -Algorithm SHA256).Hash.ToLowerInvariant();$pc=Get-Content $prototype -Raw|ConvertFrom-Json;$dataSha=(Get-FileHash ([string]$pc.data.train_manifest) -Algorithm SHA256).Hash.ToLowerInvariant();if($dataSha-ne[string]$pc.data.train_manifest_sha256){throw 'frozen TRAIN manifest mismatch'}
$cuda=&$Python -c "import json,torch;print(json.dumps({'available':torch.cuda.is_available(),'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"|ConvertFrom-Json;if(-not$cuda.available-or$cuda.name-ne'NVIDIA GeForce RTX 5080'){throw 'RTX 5080 CUDA environment unavailable'}
$eventName="Local\SIXSDeltaQPipelineReady_$([Guid]::NewGuid().ToString('N'))";$created=$false;$event=New-Object Threading.EventWaitHandle($false,[Threading.EventResetMode]::ManualReset,$eventName,[ref]$created)
try {
    $args=@('-NoProfile','-ExecutionPolicy','Bypass','-File',$supervisorScript,'-Repo',$Repo,'-Python',$Python,'-ReadyEventName',$eventName,'-OverfitWorkerPid','42000','-OverfitSupervisorPid','2152','-GitCommit',$commit,'-ConfigSha256',$configSha,'-DataManifestSha256',$dataSha)
    $p=Start-Process powershell.exe -ArgumentList $args -WorkingDirectory $Repo -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
    if(-not$event.WaitOne(30000)){throw 'pipeline supervisor did not signal readiness'}
    # One and only one post-launch status read.
    $s=Get-Content -LiteralPath $statusPath -Raw|ConvertFrom-Json
    [ordered]@{DELTAQ_PIPELINE_STATUS=[string]$s.pipeline_status;PIPELINE_SUPERVISOR_PID=[int]$s.supervisor_pid;CURRENT_STAGE=[string]$s.current_stage;OVERFIT_WORKER_PID=42000;PIPELINE_STATUS_PATH=$statusPath;PIPELINE_LOG_PATH=$out;FULL_TRAIN_SEED=307;FORMAL_TEST_READ='NO';AVG_OLD_FINAL_USED_FOR_TUNING='NO';CODEX_CAN_EXIT_SAFELY='YES';NO_REPEATED_POLLING='YES';NO_BUSY_WAITING='YES';NO_CONTINUOUS_LOG_TAIL='YES'}|ConvertTo-Json
} finally {$event.Dispose()}
