param(
    [Parameter(Mandatory = $true)][string]$CleanWorktree,
    [Parameter(Mandatory = $true)][string]$Workspace,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$FormalRoot,
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [string]$Device = 'cuda:0'
)

$ErrorActionPreference = 'Stop'
$started = [DateTimeOffset]::UtcNow
$output = [IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $output) {
    throw "Refusing to overwrite audit reproduction: $output"
}
New-Item -ItemType Directory -Path $output | Out-Null
New-Item -ItemType Directory -Path (Join-Path $output 'logs') | Out-Null

$runner = Join-Path $CleanWorktree 'scripts\run_ecir_mvr_v7_10k_validation.py'
$reporter = Join-Path $CleanWorktree 'scripts\report_ecir_mvr_v7_10k_validation.py'
$manifest = Join-Path $Workspace 'diagnostics\ecir_mvr\v7_10k\manifests'
$checkpoint = Join-Path $Workspace 'diagnostics\ecir_mvr\v2_bac_recovery\runs\d1_pilot_1000step_seed43018\checkpoint_final.ckpt'
$v5Config = Join-Path $Workspace 'diagnostics\ecir_mvr\v5_constraint_hybrid\runs\v5_b_pilot_seed43018\config.resolved.yaml'
$v7Config = Join-Path $Workspace 'diagnostics\ecir_mvr\v7_constraint_specific\runs\v7_constraint_specific_pilot_seed43018\config.resolved.yaml'
$runs = Join-Path $output 'runs'

try {
    foreach ($method in @('D1', 'V5-B', 'V7')) {
        $safe = $method.ToLower().Replace('-', '_')
        $stdout = Join-Path $output "logs\$safe.stdout.log"
        $stderr = Join-Path $output "logs\$safe.stderr.log"
        $arguments = @(
            $runner,
            '--method', $method,
            '--formal-root', $FormalRoot,
            '--source-cache-root', $SourceRoot,
            '--manifest-dir', $manifest,
            '--d1-checkpoint', $checkpoint,
            '--v5-config', $v5Config,
            '--v7-config', $v7Config,
            '--output-dir', $runs,
            '--molecules-per-chunk', '250',
            '--batch-size', '64',
            '--device', $Device
        )
        $savedPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $Python @arguments 1>> $stdout 2>> $stderr
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $savedPreference
        if ($exitCode -ne 0) {
            throw "$method reproduction failed with exit code $exitCode"
        }
    }
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Python $reporter `
        --manifest-dir $manifest `
        --runs-dir $runs `
        --output-dir $output `
        --report (Join-Path $output 'MCVR_V7_10K_REPRODUCTION_REPORT.md') `
        1>> (Join-Path $output 'logs\report.stdout.log') `
        2>> (Join-Path $output 'logs\report.stderr.log')
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($exitCode -ne 0) {
        throw "10K reproduction report failed with exit code $exitCode"
    }
    $status = @{
        status = 'COMPLETED'
        started_at = $started.ToString('o')
        completed_at = [DateTimeOffset]::UtcNow.ToString('o')
        clean_worktree_commit = '52ae6a89d3a3c8d038058ba4d52ed8c377931de0'
        methods = @('D1', 'V5-B', 'V7')
        test_records_read = 0
        test_assets_opened = $false
        frozen_holdout_records_opened = 0
        formal_test_run = $false
        training_performed = $false
    }
    $status | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output 'orchestrator_status.json') -Encoding utf8
}
catch {
    $status = @{
        status = 'FAILED'
        started_at = $started.ToString('o')
        failed_at = [DateTimeOffset]::UtcNow.ToString('o')
        error = $_.Exception.Message
        test_records_read = 0
        test_assets_opened = $false
        frozen_holdout_records_opened = 0
        formal_test_run = $false
        training_performed = $false
    }
    $status | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output 'orchestrator_status.json') -Encoding utf8
    throw
}
