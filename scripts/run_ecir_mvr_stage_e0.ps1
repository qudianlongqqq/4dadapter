param(
    [switch]$ConfirmFormal,
    [string]$Python = "E:\miniconda\envs\etflow-5080-v2\python.exe",
    [string]$Config = "configs/ecir_mvr_stage_e0_confidence_calibration.yaml"
)

$ErrorActionPreference = "Stop"

if (-not $ConfirmFormal) {
    throw "Stage E0 formal execution requires -ConfirmFormal. This runs training-data calibration and one validation evaluation."
}

& $Python scripts/build_ecir_mvr_stage_e0_calibration_data.py --config $Config
if ($LASTEXITCODE -ne 0) { throw "Stage E0 calibration data build failed" }

& $Python scripts/fit_ecir_mvr_stage_e0_calibrator.py --config $Config
if ($LASTEXITCODE -ne 0) { throw "Stage E0 calibrator fit failed" }

& $Python scripts/evaluate_ecir_mvr_stage_e0.py --config $Config
if ($LASTEXITCODE -ne 0) { throw "Stage E0 validation evaluation failed" }

& $Python scripts/report_ecir_mvr_stage_e0.py --config $Config
if ($LASTEXITCODE -ne 0) { throw "Stage E0 reporting failed" }

Write-Host "Stage E0 formal validation complete. No test, 20k, 100k, or additional seed was run."
