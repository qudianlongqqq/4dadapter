$ErrorActionPreference = "Stop"

$repo = "E:\3dconformergenerationcode\4dadapter-lsgoba-musigma-reliability-factorial"
$python = "E:\miniconda\envs\etflow-5080-v2\python.exe"
$externalPython = "E:\miniconda\envs\external-validity\python.exe"
$report = Join-Path $repo "reports\ecir_mvr\sixs_primary_final_evaluation"
$output = "E:\3dconformergenerationcode\dataset\sixs_primary_final_evaluation_v1"
$protocol = Join-Path $report "00_FROZEN_FINAL_EVALUATION_PROTOCOL.json"
$primary = Join-Path $repo "reports\ecir_mvr\sixs_step2d_primary_final_2500\04_PRIMARY_FINAL_2500_MANIFEST.json"
$sourceManifest = "E:\3dconformergenerationcode\dataset\prospective_final_etflow_nfe10_seed42\SOURCE_RECORD_MANIFEST.jsonl"
$sourceFreeze = Join-Path $repo "reports\ecir_mvr\sixs_step3a_etflow_source_generation\07_SOURCE_ASSET_FREEZE.json"
$stdout = Join-Path $report "SUPERVISOR_STDOUT.log"
$stderr = Join-Path $report "SUPERVISOR_STDERR.log"

New-Item -ItemType Directory -Force -Path $report | Out-Null
New-Item -ItemType Directory -Force -Path $output | Out-Null

$arguments = @(
    (Join-Path $repo "scripts\supervise_sixs_primary_final_evaluation.py"),
    "--repo", $repo,
    "--protocol", $protocol,
    "--primary", $primary,
    "--source-manifest", $sourceManifest,
    "--source-asset-freeze", $sourceFreeze,
    "--output-dir", $output,
    "--report-dir", $report,
    "--cuda-python", $python,
    "--external-python", $externalPython
)

$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$launch = [ordered]@{
    schema_version = "sixs-primary-final-evaluation-launch-v1"
    status = "STARTED"
    supervisor_pid = $process.Id
    output_dir = $output
    protocol = $protocol
    stdout = $stdout
    stderr = $stderr
    no_repeated_polling = $true
}
$launch | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $report "LAUNCH_STATUS.json") -Encoding UTF8
$launch | ConvertTo-Json -Depth 5
