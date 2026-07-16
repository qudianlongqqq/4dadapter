param([string]$ExpectedCommit = "")

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Python = "E:\miniconda\envs\etflow-5080-v2\python.exe"
$Config = "configs/ecir_mvr_medium_5k_500_run_a_seed42_20k_rescue_v3.yaml"
$Output = "logs_ecir_mvr/medium/run_a_seed42_rescue_v3_20k"
$Diagnostics = "diagnostics/ecir_mvr/medium/run_a_seed42_rescue_v3"
$Preflight = "$Diagnostics/preflight.json"
$Evaluation = "$Diagnostics/gate2"
$Selection = "$Diagnostics/checkpoint_selection.json"
$Phase = "controller_start"

function Invoke-PythonChecked {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "python exited with code ${LASTEXITCODE}: $($Arguments -join ' ')" }
}

try {
    if (-not (Test-Path $Python)) { throw "audited Python environment missing" }
    $Branch = (git branch --show-current).Trim()
    if ($Branch -ne "feat/ecir-mvr-progressive") { throw "branch mismatch: $Branch" }
    $Head = (git rev-parse HEAD).Trim()
    if ([string]::IsNullOrWhiteSpace($ExpectedCommit)) { $ExpectedCommit = $Head }
    if ($Head -ne $ExpectedCommit) { throw "commit mismatch: expected $ExpectedCommit, got $Head" }
    git merge-base --is-ancestor 6d3229e HEAD
    if ($LASTEXITCODE -ne 0) { throw "V2 experiment commit is not an ancestor" }
    $Dirty = @(git status --porcelain --untracked-files=all | Where-Object {
        $_ -ne "?? reports/global4d_profile_bundle_verification.json"
    })
    if ($Dirty.Count -ne 0) { throw "unexpected worktree changes: $($Dirty -join '; ')" }
    $Protected = (Get-FileHash -Algorithm SHA256 reports/global4d_profile_bundle_verification.json).Hash.ToLower()
    if ($Protected -ne "738171eb7e5f047e94cd4e1a46689613fe1f30bf33a320a2e3ec5a6944a5ec7d") { throw "protected report changed" }
    if (Test-Path $Output) {
        if (@(Get-ChildItem $Output -Force).Count -ne 0) { throw "V3 output already contains artifacts" }
    }
    New-Item -ItemType Directory -Force -Path $Output, $Diagnostics | Out-Null
    Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark pipeline_start

    $Phase = "raw_vs_clipped_audit"
    Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark identity_audit_start
    Invoke-PythonChecked scripts/audit_ecir_mvr_medium_velocity_raw_vs_clipped.py `
        --config configs/ecir_mvr_medium_5k_500_run_a_seed42_20k_rescue_v2.yaml `
        --checkpoint logs_ecir_mvr/medium/run_a_seed42_rescue_v2_20k/checkpoints/last.ckpt `
        --metrics logs_ecir_mvr/medium/run_a_seed42_rescue_v2_20k/metrics.csv `
        --output $Diagnostics/raw_vs_clipped_audit.json `
        --report docs/MCVR_MEDIUM_VELOCITY_RAW_VS_CLIPPED_AUDIT.md
    Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark identity_audit_end

    $Phase = "tests"
    Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark test_start
    Invoke-PythonChecked -m pytest -q tests/test_ecir_mvr_medium_rescue_v3.py tests/test_ecir_mvr_medium_rescue_v2.py tests/test_ecir_mvr_medium.py tests/test_ecir_mvr_stage_c.py
    Invoke-PythonChecked -m pytest -q
    Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark test_end

    $Phase = "preflight"
    Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark preflight_start
    Invoke-PythonChecked scripts/audit_ecir_mvr_medium_preflight.py --config $Config --output $Preflight --report docs/MCVR_MEDIUM_RESCUE_V3_PREFLIGHT.md
    Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark preflight_end

    $Phase = "training"
    $Monitor = Start-Process -FilePath $Python `
        -ArgumentList @("scripts/monitor_ecir_mvr_heartbeat.py", "--output-dir", $Output, "--interval", "45") `
        -WorkingDirectory $Root -WindowStyle Hidden -PassThru
    & $Python scripts/train_ecir_mvr_medium_20k.py --config $Config
    if ($LASTEXITCODE -ne 0) {
        $Heartbeat = Get-Content "$Output/heartbeat.json" -Raw | ConvertFrom-Json
        $Control = if (Test-Path "$Output/resume_control.json") { Get-Content "$Output/resume_control.json" -Raw | ConvertFrom-Json } else { [pscustomobject]@{resume_allowed=($Heartbeat.status -eq "RUNNING"); reason="external_process_exit"} }
        $Checkpoint = Get-ChildItem "$Output/checkpoints" -Filter "step*.ckpt" -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
        if (-not $Control.resume_allowed -or $null -eq $Checkpoint) { throw "training failed without authorized recovery: $($Control.reason)" }
        Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark resume_start
        & $Python scripts/train_ecir_mvr_medium_20k.py --config $Config --resume_checkpoint $Checkpoint.FullName --controller_resume
        if ($LASTEXITCODE -ne 0) { throw "single authorized V3 recovery failed" }
        Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark resume_end
    }

    $Phase = "checkpoint_selection"
    Invoke-PythonChecked scripts/select_ecir_mvr_medium_rescue_v3_checkpoint.py `
        --v2-comparison diagnostics/ecir_mvr/medium/run_a_seed42_rescue_v2_20k/checkpoint_comparison.csv `
        --v3-comparison $Diagnostics/checkpoint_comparison.csv `
        --run-metadata $Output/run_metadata.json `
        --output $Selection
    $Selected = (Get-Content $Selection -Raw | ConvertFrom-Json).selected_checkpoint

    $Phase = "final_evaluation"
    Invoke-PythonChecked scripts/evaluate_ecir_mvr_medium_seed42.py `
        --config $Config --checkpoint $Selected --preflight $Preflight `
        --run_a_checkpoint logs_ecir_mvr/stage2b/run_a_rigid_only_seed42_5k/checkpoints/best_noninferior_validity.ckpt `
        --output_dir $Evaluation --bootstrap_draws 1000 --timing_dir $Output

    $Phase = "report_generation"
    Invoke-PythonChecked scripts/report_ecir_mvr_medium_rescue_v3.py `
        --config $Config --evaluation-dir $Evaluation --selection $Selection --output-dir $Output
    Invoke-PythonChecked scripts/update_ecir_mvr_sha256_inventory.py `
        --inventory reports/ecir_mvr/SHA256SUMS.txt `
        $Config $Diagnostics docs/MCVR_MEDIUM_VELOCITY_RAW_VS_CLIPPED_AUDIT.md `
        docs/MCVR_MEDIUM_RESCUE_V3_PREFLIGHT.md docs/MCVR_MEDIUM_SEED42_RESCUE_V3_REPORT.md `
        docs/MCVR_MEDIUM_SEED42_RESCUE_V3_GATE2.md docs/MCVR_MEDIUM_SEED42_RESCUE_V3_CHECKPOINT_SELECTION.md `
        reports/ecir_mvr/progressive_state.json etflow/ecir/mvr_model.py etflow/ecir/mvr_safety.py `
        scripts/train_ecir_mvr_medium_rescue_v2.py scripts/run_ecir_mvr_medium_seed42_rescue_v3.ps1 `
        tests/test_ecir_mvr_medium_rescue_v3.py $Output
    $ProtectedAfter = (Get-FileHash -Algorithm SHA256 reports/global4d_profile_bundle_verification.json).Hash.ToLower()
    if ($ProtectedAfter -ne "738171eb7e5f047e94cd4e1a46689613fe1f30bf33a320a2e3ec5a6944a5ec7d") { throw "protected report changed during V3" }
}
catch {
    & $Python scripts/set_ecir_mvr_rescue_v3_failure.py --output-dir $Output --phase $Phase --reason $_.Exception.Message
    throw
}
