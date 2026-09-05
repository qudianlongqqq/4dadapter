param(
  [string]$Repo = 'E:\3dconformergenerationcode\4dadapter-lsgoba-musigma-reliability-factorial',
  [switch]$ResumeFromReferenceXtb
)
$ErrorActionPreference = 'Stop'
$python = 'E:\miniconda\envs\etflow-5080-v2\python.exe'
$externalPython = 'E:\miniconda\envs\external-validity\python.exe'
$report = Join-Path $Repo 'reports\ecir_mvr\final_evidence_closure'
$asset = 'E:\3dconformergenerationcode\dataset\sixs_final_evidence_closure_v1'
New-Item -ItemType Directory -Force -Path $report | Out-Null
New-Item -ItemType Directory -Force -Path $asset | Out-Null
$arguments = @(
  (Join-Path $Repo 'scripts\supervise_sixs_final_evidence_closure.py'),
  '--repo', $Repo,
  '--protocol', (Join-Path $Repo 'reports\ecir_mvr\sixs_primary_final_evaluation\00_FROZEN_FINAL_EVALUATION_PROTOCOL.json'),
  '--primary-manifest', (Join-Path $Repo 'reports\ecir_mvr\sixs_step2d_primary_final_2500\04_PRIMARY_FINAL_2500_MANIFEST.json'),
  '--primary-asset', 'E:\3dconformergenerationcode\dataset\sixs_primary_final_evaluation_v1',
  '--cross-asset', 'E:\3dconformergenerationcode\dataset\sixs_final_cross_upstream_unrestricted',
  '--topology-cache', 'E:\3dconformergenerationcode\dataset\sixs_primary_final_evaluation_v1\topology_reference_cache',
  '--source-xtb', 'E:\3dconformergenerationcode\dataset\sixs_primary_final_evaluation_v1\xtb_single_point\SOURCE_XTB.csv',
  '--asset-dir', $asset,
  '--report-dir', $report,
  '--cuda-python', $python,
  '--external-python', $externalPython
)
if ($ResumeFromReferenceXtb) {
  $arguments += '--resume-from-reference-xtb'
}
$stdout = Join-Path $report 'SUPERVISOR_STDOUT.log'
$stderr = Join-Path $report 'SUPERVISOR_STDERR.log'
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $Repo -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$launch = [ordered]@{
  status = 'STARTED'
  supervisor_pid = $process.Id
  command = $python + ' ' + ($arguments -join ' ')
  report_dir = $report
  asset_dir = $asset
  stdout = $stdout
  stderr = $stderr
  repeated_polling = $false
}
$launchJson = $launch | ConvertTo-Json -Depth 5
[System.IO.File]::WriteAllText((Join-Path $report 'LAUNCH_STATUS.json'), $launchJson + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
$launch | ConvertTo-Json -Depth 5
