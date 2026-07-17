param(
    [switch]$ConfirmFormal,
    [string]$Python = "E:\miniconda\envs\etflow-5080-v2\python.exe",
    [string]$Config = "configs/ecir_mvr_stage_g_bounded_residual.yaml",
    [string]$OutputRoot = "diagnostics/ecir_mvr/stage_g",
    [string]$Device = "cuda",
    [int]$Seed = 42,
    [int]$BuilderBatchSize = 64,
    [int]$BatchSize = 65536,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$DatasetResidency = "auto",
    [int]$NumWorkers = 0,
    [switch]$ProfileCudaMemory,
    [int]$ProfileEverySteps = 100
)

$ErrorActionPreference = "Stop"

if (-not $ConfirmFormal) {
    throw "Stage G formal execution requires -ConfirmFormal. No test or long neural training is authorized."
}

& $Python scripts/build_ecir_mvr_stage_g_calibration_data.py `
    --config $Config --output-dir $OutputRoot --device $Device --seed $Seed `
    --builder-batch-size $BuilderBatchSize
if ($LASTEXITCODE -ne 0) { throw "Stage G calibration data build failed" }

$fitArgs = @(
    "scripts/fit_ecir_mvr_stage_g_calibrator.py",
    "--config", $Config,
    "--input-dir", $OutputRoot,
    "--output-dir", $OutputRoot,
    "--device", $Device,
    "--seed", $Seed,
    "--batch-size", $BatchSize,
    "--dataset-residency", $DatasetResidency,
    "--num-workers", $NumWorkers,
    "--profile-every-steps", $ProfileEverySteps,
    "--pin-memory"
)
if ($ProfileCudaMemory) { $fitArgs += "--profile-cuda-memory" }
& $Python @fitArgs
if ($LASTEXITCODE -ne 0) { throw "Stage G calibrator fit failed" }

$fitResult = Get-Content (Join-Path $OutputRoot "fit_result.json") -Raw | ConvertFrom-Json
if ($fitResult.decision -eq "STAGE_G_COLLAPSED") {
    Write-Host "Stage G stopped: every preregistered checkpoint collapsed. Validation was not run."
    exit 0
}

& $Python scripts/evaluate_ecir_mvr_stage_g.py `
    --config $Config --input-dir $OutputRoot --output-dir $OutputRoot `
    --device $Device --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "Stage G validation evaluation failed" }

Write-Host "Stage G pilot complete. Stage F is unchanged; no test, 50k, 100k, or 200k run was started."
