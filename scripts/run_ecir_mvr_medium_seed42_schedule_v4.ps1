param([string]$ExpectedCommit = "")

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Python = "E:\miniconda\envs\etflow-5080-v2\python.exe"
$Config = "configs/ecir_mvr_medium_5k_500_run_a_seed42_schedule_v4_10k.yaml"
$Output = "logs_ecir_mvr/medium/run_a_seed42_schedule_v4_10k"
$Diagnostics = "diagnostics/ecir_mvr/medium/run_a_seed42_schedule_v4"
$Preflight = "$Diagnostics/preflight.json"
$Selection = "$Diagnostics/checkpoint_selection.json"

function Invoke-PythonChecked {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "python exited with code ${LASTEXITCODE}: $($Arguments -join ' ')" }
}

if (-not (Test-Path $Python)) { throw "audited Python environment missing" }
$Branch = (git branch --show-current).Trim()
if ($Branch -ne "feat/ecir-mvr-progressive") { throw "branch mismatch: $Branch" }
$Head = (git rev-parse HEAD).Trim()
if ([string]::IsNullOrWhiteSpace($ExpectedCommit)) { $ExpectedCommit = $Head }
if ($Head -ne $ExpectedCommit) { throw "commit mismatch: expected $ExpectedCommit, got $Head" }
if (Test-Path $Output) {
    if (@(Get-ChildItem $Output -Force).Count -ne 0) { throw "V4 output already contains artifacts" }
}
if (Test-Path $Diagnostics) {
    if (@(Get-ChildItem $Diagnostics -Force).Count -ne 0) { throw "V4 diagnostics already contains artifacts" }
}
New-Item -ItemType Directory -Force -Path $Output, $Diagnostics | Out-Null
Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark pipeline_start

Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark preflight_start
Invoke-PythonChecked scripts/audit_ecir_mvr_medium_preflight.py `
    --config $Config --output $Preflight --report docs/MCVR_MEDIUM_SCHEDULE_V4_PREFLIGHT.md
Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark preflight_end

$Monitor = Start-Process -FilePath $Python `
    -ArgumentList @("scripts/monitor_ecir_mvr_heartbeat.py", "--output-dir", $Output, "--interval", "45") `
    -WorkingDirectory $Root -WindowStyle Hidden -PassThru
Invoke-PythonChecked scripts/train_ecir_mvr_medium_20k.py --config $Config

Invoke-PythonChecked scripts/select_ecir_mvr_medium_schedule_v4_checkpoint.py `
    --comparison $Diagnostics/checkpoint_comparison.csv `
    --run-metadata $Output/run_metadata.json --output $Selection
$Selected = (Get-Content $Selection -Raw | ConvertFrom-Json).selected_checkpoint

Invoke-PythonChecked scripts/evaluate_ecir_mvr_medium_seed42.py `
    --config $Config --checkpoint $Selected --preflight $Preflight `
    --run_a_checkpoint logs_ecir_mvr/stage2b/run_a_rigid_only_seed42_5k/checkpoints/best_noninferior_validity.ckpt `
    --output_dir $Diagnostics --bootstrap_draws 1000 --timing_dir $Output

Invoke-PythonChecked scripts/report_ecir_mvr_medium_schedule_v4.py `
    --config $Config --diagnostics-dir $Diagnostics --selection $Selection --output-dir $Output
Invoke-PythonChecked scripts/update_ecir_mvr_sha256_inventory.py `
    --inventory reports/ecir_mvr/SHA256SUMS.txt `
    $Config $Diagnostics docs/MCVR_MEDIUM_SCHEDULE_V4_PREFLIGHT.md `
    docs/MCVR_MEDIUM_SEED42_SCHEDULE_V4_REPORT.md `
    docs/MCVR_MEDIUM_SEED42_SCHEDULE_V4_GATE2.md `
    docs/MCVR_MEDIUM_TRAINING_SCHEDULE_ANALYSIS.md `
    reports/ecir_mvr/progressive_state.json `
    scripts/train_ecir_mvr_medium_rescue_v2.py `
    scripts/evaluate_ecir_mvr_medium_seed42.py `
    scripts/select_ecir_mvr_medium_schedule_v4_checkpoint.py `
    scripts/report_ecir_mvr_medium_schedule_v4.py `
    scripts/run_ecir_mvr_medium_seed42_schedule_v4.ps1 $Output

if (-not $Monitor.HasExited) {
    Wait-Process -Id $Monitor.Id -Timeout 90 -ErrorAction SilentlyContinue
}
