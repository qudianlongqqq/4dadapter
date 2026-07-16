param(
    [string]$ExpectedCommit = ""
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Python = "E:\miniconda\envs\etflow-5080-v2\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "audited Python environment is missing: $Python"
}
$Config = "configs/ecir_mvr_medium_5k_500_run_a_seed42_20k_rescue_v2.yaml"
$Output = "logs_ecir_mvr/medium/run_a_seed42_rescue_v2_20k"
$Diagnostics = "diagnostics/ecir_mvr/medium/run_a_seed42_rescue_v2_20k"
$Preflight = "$Diagnostics/preflight.json"
$Benchmark = "$Diagnostics/dataloader_benchmark.json"
$Evaluation = "$Diagnostics/gate2"
$Phase = "controller_start"
$TrainingStarted = $false
$ResumeCount = 0

function Invoke-PythonChecked {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "python exited with code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

try {
    $Branch = (git branch --show-current).Trim()
    if ($Branch -ne "feat/ecir-mvr-progressive") {
        throw "branch mismatch: $Branch"
    }
    $Head = (git rev-parse HEAD).Trim()
    if ([string]::IsNullOrWhiteSpace($ExpectedCommit)) {
        $ExpectedCommit = $Head
    }
    if ($Head -ne $ExpectedCommit) {
        throw "commit mismatch: expected $ExpectedCommit, got $Head"
    }
    git merge-base --is-ancestor d7837baabab356d9a9a4abb2c53ed4312386c567 HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "Medium V1 final commit is not an ancestor"
    }
    $Dirty = @(git status --porcelain --untracked-files=all | Where-Object {
        $_ -ne "?? reports/global4d_profile_bundle_verification.json"
    })
    if ($Dirty.Count -ne 0) {
        throw "worktree contains unexpected changes: $($Dirty -join '; ')"
    }
    $ProtectedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath "reports/global4d_profile_bundle_verification.json").Hash.ToLower()
    if ($ProtectedHash -ne "738171eb7e5f047e94cd4e1a46689613fe1f30bf33a320a2e3ec5a6944a5ec7d") {
        throw "protected global4d verification report changed"
    }
    if (Test-Path -LiteralPath $Output) {
        $ExistingOutput = @(Get-ChildItem -LiteralPath $Output -Force)
        if ($ExistingOutput.Count -ne 0) {
            throw "Rescue V2 output directory already contains artifacts; refusing to overwrite a non-fresh run"
        }
    }

    New-Item -ItemType Directory -Force -Path $Output, $Diagnostics | Out-Null
    Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark pipeline_start

    $Phase = "identity_audit"
    Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark identity_audit_start
    Invoke-PythonChecked scripts/audit_ecir_mvr_medium_preflight.py --config $Config --output $Preflight --report docs/MCVR_MEDIUM_SEED42_RESCUE_V2_PREFLIGHT.md
    Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark identity_audit_end

    $Phase = "tests"
    Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark test_start
    Invoke-PythonChecked -m pytest -q tests/test_ecir_mvr_medium_rescue_v2.py tests/test_ecir_mvr_medium.py tests/test_ecir_mvr_stage_c.py
    Invoke-PythonChecked -m pytest -q
    Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark test_end

    $Phase = "preflight"
    Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark preflight_start
    Invoke-PythonChecked scripts/audit_ecir_mvr_medium_preflight.py --config $Config --output $Preflight --report docs/MCVR_MEDIUM_SEED42_RESCUE_V2_PREFLIGHT.md
    Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark preflight_end

    $Phase = "dataloader_benchmark"
    Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark dataloader_benchmark_start
    Invoke-PythonChecked scripts/benchmark_ecir_mvr_medium_dataloader.py --config $Config --output $Benchmark --batches 32
    Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark dataloader_benchmark_end

    $Phase = "resolved_config"
    Invoke-PythonChecked scripts/resolve_ecir_mvr_medium_rescue_v2_config.py --config $Config --benchmark $Benchmark --output "$Output/config.resolved.yaml"

    $Phase = "training"
    $TrainingStarted = $true
    $HeartbeatMonitor = Start-Process -FilePath $Python `
        -ArgumentList @("scripts/monitor_ecir_mvr_heartbeat.py", "--output-dir", $Output, "--interval", "45") `
        -WorkingDirectory $Root -WindowStyle Hidden -PassThru
    & $Python scripts/train_ecir_mvr_medium_20k.py --config $Config
    $TrainExit = $LASTEXITCODE
    if ($TrainExit -ne 0) {
        $Heartbeat = Get-Content -LiteralPath "$Output/heartbeat.json" -Raw | ConvertFrom-Json
        $ResumeControl = if (Test-Path "$Output/resume_control.json") {
            Get-Content -LiteralPath "$Output/resume_control.json" -Raw | ConvertFrom-Json
        } else {
            [pscustomobject]@{ resume_allowed = ($Heartbeat.status -eq "RUNNING"); reason = "external_process_exit" }
        }
        $ResumeCheckpoint = Get-ChildItem -LiteralPath "$Output/checkpoints" -Filter "step*.ckpt" -ErrorAction SilentlyContinue |
            Sort-Object Name | Select-Object -Last 1
        if (-not $ResumeControl.resume_allowed -or $null -eq $ResumeCheckpoint -or $ResumeCount -ge 1) {
            throw "training failed without authorized recovery: status=$($Heartbeat.status), reason=$($ResumeControl.reason)"
        }
        $ResumeCount++
        Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark resume_start --details "{`"segment`":2,`"resume_reason`":`"$($ResumeControl.reason)`",`"checkpoint`":`"$($ResumeCheckpoint.FullName.Replace('\','/'))`"}"
        & $Python scripts/train_ecir_mvr_medium_20k.py --config $Config --resume_checkpoint $ResumeCheckpoint.FullName --controller_resume
        if ($LASTEXITCODE -ne 0) {
            throw "the single authorized automatic resume failed"
        }
        Invoke-PythonChecked -m etflow.commons.run_timing --output-dir $Output mark resume_end
    }

    $Metadata = Get-Content -LiteralPath "$Output/run_metadata.json" -Raw | ConvertFrom-Json
    $Checkpoint = if (Test-Path "$Output/checkpoints/best_noninferior_validity.ckpt") {
        "$Output/checkpoints/best_noninferior_validity.ckpt"
    } else {
        "$Output/checkpoints/last.ckpt"
    }

    $Phase = "final_evaluation"
    Invoke-PythonChecked scripts/evaluate_ecir_mvr_medium_seed42.py `
        --config $Config `
        --checkpoint $Checkpoint `
        --preflight $Preflight `
        --run_a_checkpoint logs_ecir_mvr/stage2b/run_a_rigid_only_seed42_5k/checkpoints/best_noninferior_validity.ckpt `
        --output_dir $Evaluation `
        --bootstrap_draws 1000 `
        --timing_dir $Output

    $Phase = "report_generation"
    Invoke-PythonChecked scripts/report_ecir_mvr_medium_rescue_v2.py --config $Config --evaluation-dir $Evaluation --output-dir $Output
    Invoke-PythonChecked scripts/update_ecir_mvr_sha256_inventory.py `
        --inventory reports/ecir_mvr/SHA256SUMS.txt `
        $Config $Diagnostics `
        docs/MCVR_MEDIUM_SEED42_RESCUE_V2_PREFLIGHT.md `
        docs/MCVR_MEDIUM_SEED42_RESCUE_V2_REPORT.md `
        docs/MCVR_MEDIUM_SEED42_RESCUE_V2_CHECKPOINT_SELECTION.md `
        docs/MCVR_MEDIUM_SEED42_RESCUE_V2_GATE2.md `
        reports/ecir_mvr/progressive_state.json `
        etflow/commons/run_timing.py `
        scripts/train_ecir_mvr_medium_rescue_v2.py `
        scripts/run_ecir_mvr_medium_seed42_overnight.ps1 `
        tests/test_ecir_mvr_medium_rescue_v2.py `
        $Output
    $ProtectedHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath "reports/global4d_profile_bundle_verification.json").Hash.ToLower()
    if ($ProtectedHashAfter -ne "738171eb7e5f047e94cd4e1a46689613fe1f30bf33a320a2e3ec5a6944a5ec7d") {
        throw "protected global4d verification report changed during overnight execution"
    }
}
catch {
    $Reason = $_.Exception.Message
    $PreflightFailure = -not $TrainingStarted
    $Arguments = @(
        "scripts/set_ecir_mvr_rescue_v2_failure.py", "--output-dir", $Output,
        "--phase", $Phase, "--reason", $Reason
    )
    if ($PreflightFailure) {
        $Arguments += "--preflight"
    }
    & $Python @Arguments
    throw
}
