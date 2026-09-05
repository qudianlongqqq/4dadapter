$ErrorActionPreference = "Stop"
$Repo = "E:\3dconformergenerationcode\4dadapter-lsgoba-musigma-reliability-factorial"
$Python = "E:\miniconda\envs\etflow-5080-v2\python.exe"
$Supervisor = Join-Path $Repo "scripts\supervise_sixs_final_cross_upstream_unrestricted.py"
$Report = Join-Path $Repo "reports\ecir_mvr\sixs_final_cross_upstream_unrestricted"
New-Item -ItemType Directory -Force -Path $Report | Out-Null
$stdout = Join-Path $Report "SUPERVISOR_STDOUT.log"
$stderr = Join-Path $Report "SUPERVISOR_STDERR.log"
$process = Start-Process -FilePath $Python -ArgumentList @($Supervisor) -WorkingDirectory $Repo `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$payload = [ordered]@{
    status = "STARTED"
    supervisor_pid = $process.Id
    started_at = [DateTimeOffset]::Now.ToString("o")
    output_dir = "E:\3dconformergenerationcode\dataset\sixs_final_cross_upstream_unrestricted"
    report_dir = $Report
    no_repeated_polling = $true
}
$payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Report "LAUNCH_STATUS.json") -Encoding UTF8
$payload | ConvertTo-Json -Compress
