$ErrorActionPreference = 'Stop'

$Repo = 'E:\3dconformergenerationcode\4dadapter-lsgoba-musigma-reliability-factorial'
$Python = 'E:\miniconda\envs\etflow-5080-v2\python.exe'
$Runner = Join-Path $Repo 'scripts\generate_sixs_prospective_etflow_nfe10.py'
$Manifest = Join-Path $Repo 'reports\ecir_mvr\sixs_step2d_primary_final_2500\04_PRIMARY_FINAL_2500_MANIFEST.json'
$Protocol = Join-Path $Repo 'reports\ecir_mvr\sixs_step3a_etflow_source_generation\03_PROSPECTIVE_ETFLOW_NFE10_PROTOCOL.json'
$Config = Join-Path $Repo 'configs\etflow_drugs_o3_prospective_nfe10.yaml'
$Checkpoint = 'E:\3dconformergenerationcode\.cache\etflow_official\drugs-o3.ckpt'
$EtflowRoot = 'E:\3dconformergenerationcode\ETFlow'
$SourceRoot = 'E:\3dconformergenerationcode\dataset\prospective_source_assets\torsional_diffusion_official\full_extract'
$OutputDir = 'E:\3dconformergenerationcode\dataset\prospective_final_etflow_nfe10_seed42'
$StatusFile = Join-Path $OutputDir 'SUPERVISOR_STATUS.json'
$LogFile = Join-Path $OutputDir 'generation.log'

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function Write-SupervisorStatus {
    param([hashtable]$Payload)
    $Temporary = "$StatusFile.$PID.tmp"
    $Payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Temporary -Encoding utf8
    Move-Item -LiteralPath $Temporary -Destination $StatusFile -Force
}

$Command = @(
    $Runner,
    '--mode', 'formal',
    '--manifest', $Manifest,
    '--protocol', $Protocol,
    '--config', $Config,
    '--checkpoint', $Checkpoint,
    '--etflow-root', $EtflowRoot,
    '--source-root', $SourceRoot,
    '--output-dir', $OutputDir
)

$StartedAt = (Get-Date).ToUniversalTime().ToString('o')
Write-SupervisorStatus @{
    schema_version = 'sixs-prospective-etflow-supervisor-v1'
    status = 'RUNNING'
    supervisor_pid = $PID
    started_at_utc = $StartedAt
    output_dir = $OutputDir
    log_file = $LogFile
    resume_supported = $true
}

try {
    $env:PYTHONUNBUFFERED = '1'
    & $Python @Command *>> $LogFile
    $Code = $LASTEXITCODE
    Write-SupervisorStatus @{
        schema_version = 'sixs-prospective-etflow-supervisor-v1'
        status = $(if ($Code -eq 0) { 'COMPLETE' } else { 'FAILED' })
        supervisor_pid = $PID
        started_at_utc = $StartedAt
        finished_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        exit_code = $Code
        output_dir = $OutputDir
        log_file = $LogFile
        resume_supported = $true
    }
    exit $Code
}
catch {
    Write-SupervisorStatus @{
        schema_version = 'sixs-prospective-etflow-supervisor-v1'
        status = 'SUPERVISOR_FAILURE'
        supervisor_pid = $PID
        started_at_utc = $StartedAt
        finished_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        error = $_.Exception.Message
        output_dir = $OutputDir
        log_file = $LogFile
        resume_supported = $true
    }
    throw
}
