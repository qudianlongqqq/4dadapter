param(
 [string]$Repo='E:\3dconformergenerationcode\4dadapter-lsgoba-musigma-reliability-factorial',
 [string]$Python='E:\miniconda\envs\etflow-5080-v2\python.exe',
 [string]$ReadyEventName=''
)
$ErrorActionPreference='Stop';$runtime=Join-Path $Repo 'reports\ecir_mvr\sixs_deltaq_learnability_audit\runtime';New-Item -ItemType Directory -Path $runtime -Force|Out-Null
$statusPath=Join-Path $runtime 'AUDIT_STATUS.json';$guard=Join-Path $Repo 'scripts\run_sixs_deltaq_overfit_native_guard.py';$audit=Join-Path $Repo 'scripts\audit_sixs_deltaq_learnability.py';$head=Join-Path $Repo 'scripts\run_sixs_deltaq_head_only_diagnostic.py'
function W([hashtable]$v){$t="$statusPath.tmp.$PID";[IO.File]::WriteAllText($t,($v|ConvertTo-Json -Depth 8)+[Environment]::NewLine);Move-Item $t $statusPath -Force}
$s=@{status='RUNNING';stage='STATIC_AUDIT';supervisor_pid=[int]$PID;worker_pid=$null;start_time=[DateTimeOffset]::Now.ToString('o');end_time=$null;exit_code=$null;stdout_path=$null;stderr_path=$null;formal_test_read='NO';dev_evaluation='NO';full_training='NO'};W $s
if($ReadyEventName){$e=[Threading.EventWaitHandle]::OpenExisting($ReadyEventName);[void]$e.Set();$e.Dispose()}
function Run([string]$stage,[string[]]$a){
 $o=Join-Path $runtime "${stage}_STDOUT.log";$er=Join-Path $runtime "${stage}_STDERR.log";[IO.File]::WriteAllText($o,'');[IO.File]::WriteAllText($er,'');$s.stage=$stage;$s.stdout_path=$o;$s.stderr_path=$er
 $root=Split-Path -Parent $Python;$win=[Environment]::GetFolderPath('Windows');$git=Split-Path -Parent (Get-Command git.exe).Source;$path=@($root,(Join-Path $root 'Library\bin'),(Join-Path $root 'Scripts'),$git,(Join-Path $win 'System32'),$win)-join';'
 $si=New-Object Diagnostics.ProcessStartInfo;$si.FileName=$Python;$si.Arguments=($a|ForEach-Object{'"'+$_+'"'})-join' ';$si.WorkingDirectory=$Repo;$si.UseShellExecute=$false;$si.CreateNoWindow=$true;$si.RedirectStandardOutput=$true;$si.RedirectStandardError=$true;$si.EnvironmentVariables['PATH']=$path;$si.EnvironmentVariables['CONDA_PREFIX']=$root;$si.EnvironmentVariables['PYTHONNOUSERSITE']='1'
 $p=New-Object Diagnostics.Process;$p.StartInfo=$si;if(-not$p.Start()){throw "cannot start $stage"};$s.worker_pid=[int]$p.Id;W $s;$ot=$p.StandardOutput.ReadToEndAsync();$et=$p.StandardError.ReadToEndAsync();$p.WaitForExit();$ot.Wait();$et.Wait();$p.Refresh();[IO.File]::WriteAllText($o,$ot.Result);[IO.File]::WriteAllText($er,$et.Result);$s.exit_code=[int]$p.ExitCode;W $s;if($p.ExitCode-ne0){throw "$stage exited $($p.ExitCode)"}
}
try{Run 'STATIC_AUDIT' @($guard,$audit,'static');Run 'HEAD_ONLY_5000' @($guard,$head);Run 'FINALIZE' @($guard,$audit,'finalize');$s.status='COMPLETED';$s.stage='COMPLETED';$s.worker_pid=$null;$s.end_time=[DateTimeOffset]::Now.ToString('o');$s.exit_code=0;W $s}catch{$s.status='FAILED';$s.end_time=[DateTimeOffset]::Now.ToString('o');$s.failure_reason=$_.Exception.Message;W $s;exit 1}
