param(
    [switch]$ConfirmFormal,
    [string]$Python = "E:\miniconda\envs\etflow-5080-v2\python.exe",
    [string]$Config = "configs/ecir_mvr_stage_f_feature_confidence.yaml"
)

$ErrorActionPreference = "Stop"

if (-not $ConfirmFormal) {
    throw "Stage F formal execution requires -ConfirmFormal. This fits only the small train-only calibrator and evaluates validation once."
}

& $Python scripts/build_ecir_mvr_stage_f_calibration_data.py --config $Config
if ($LASTEXITCODE -ne 0) { throw "Stage F calibration data build failed" }

& $Python scripts/fit_ecir_mvr_stage_f_calibrator.py --config $Config
if ($LASTEXITCODE -ne 0) { throw "Stage F calibrator fit failed" }

& $Python scripts/evaluate_ecir_mvr_stage_f.py --config $Config
if ($LASTEXITCODE -ne 0) { throw "Stage F validation evaluation failed" }

& $Python scripts/report_ecir_mvr_stage_f.py --config $Config
if ($LASTEXITCODE -ne 0) { throw "Stage F reporting failed" }

Write-Host "Stage F pilot complete. No test, seed43/44, 20k, 100k, or 200k run was started."
