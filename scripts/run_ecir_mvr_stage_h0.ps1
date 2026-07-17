param(
    [switch]$ConfirmFormal,
    [string]$Python = "E:\miniconda\envs\etflow-5080-v2\python.exe",
    [string]$Config = "configs/ecir_mvr_stage_h0_conflict_fusion.yaml",
    [string]$Device = "cuda",
    [int]$RecordBatchSize = 64,
    [string]$OutputDir = "diagnostics/ecir_mvr/stage_h0",
    [switch]$ProfileCudaMemory,
    [int]$ProfileEveryRecords = 100
)
$ErrorActionPreference = "Stop"
if (-not $ConfirmFormal) { throw "Stage H0 700-record diagnostic requires -ConfirmFormal" }
$arguments = @("scripts/evaluate_ecir_mvr_stage_h0.py", "--config", $Config, "--device", $Device,
    "--record-batch-size", $RecordBatchSize, "--output-dir", $OutputDir, "--confirm-formal",
    "--profile-every-records", $ProfileEveryRecords)
if ($ProfileCudaMemory) { $arguments += "--profile-cuda-memory" }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Stage H0 diagnostic failed" }
Write-Host "Stage H0 validation-only diagnostic complete. No training or test read occurred."
