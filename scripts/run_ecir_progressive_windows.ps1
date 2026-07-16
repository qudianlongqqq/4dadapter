param(
    [string]$Python = "E:\miniconda\envs\etflow-5080-v2\python.exe",
    [string]$CacheDir = "E:\3dconformergenerationcode\dataset\flexbond_cache_formal_large",
    [string]$CartesianTrainCache = "E:\3dconformergenerationcode\serial_global4d_work\pilot_cache",
    [string]$CartesianValCache = "E:\3dconformergenerationcode\serial_global4d_work\pilot_cache\val_confirm30",
    [string]$AtlasDir = "data\ecir_error_atlas"
)

$ErrorActionPreference = "Stop"

$sourceConfig = "diagnostics\ecir\runtime_sources_windows.json"
New-Item -ItemType Directory -Force (Split-Path $sourceConfig) | Out-Null
$sourcePayload = @{
    sources = @(
        @{source_type="upstream_etflow_formal"; cache_dir=$CacheDir; coordinate_key="x_init"; checkpoint="formal_upstream"; NFE=10; solver="etflow"; seed=42},
        @{source_type="cartesian_teacher_100k"; split_paths=@{train=$CartesianTrainCache; val=$CartesianValCache}; coordinate_key="x_cart"; checkpoint="cartesian_step100000"; NFE=10; solver="cartesian_refinement"; seed=42}
    )
} | ConvertTo-Json -Depth 5
[IO.File]::WriteAllText(
    (Join-Path (Get-Location) $sourceConfig),
    $sourcePayload,
    [Text.UTF8Encoding]::new($false)
)

& $Python scripts\build_conformer_error_atlas.py `
    --sources_config $sourceConfig `
    --splits train,val,test --limits 500,100,100 `
    --output_dir $AtlasDir --report docs\ECIR_ERROR_ATLAS_REPORT.md

# Stage 1: the config selects exactly five molecules. Both CPU and CUDA must pass.
& $Python scripts\train_ecir_flow.py --config configs\ecir_flow_smoke.yaml `
    --cache_dir $CacheDir --target_cache_dir "$AtlasDir\targets" `
    --device cpu --output_dir logs_ecir\stage1_cpu
& $Python scripts\train_ecir_flow.py --config configs\ecir_flow_smoke.yaml `
    --cache_dir $CacheDir --target_cache_dir "$AtlasDir\targets" `
    --device cuda --output_dir logs_ecir\stage1_cuda

# Stage 2: fixed 500/100 cohort and 5,000 updates.
& $Python scripts\train_ecir_flow.py --config configs\ecir_flow_formal_small.yaml `
    --cache_dir $CacheDir --target_cache_dir "$AtlasDir\targets" `
    --output_dir logs_ecir\stage2_heterogeneous_500_100_5k
& $Python scripts\eval_ecir_refiner.py `
    --checkpoint logs_ecir\stage2_heterogeneous_500_100_5k\step005000.ckpt `
    --cache_dir $CacheDir --target_cache_dir "$AtlasDir\targets" `
    --atlas_path "$AtlasDir\val.parquet" --split val --max_records 100 --steps 4 `
    --output_dir diagnostics\ecir\stage2_heterogeneous_eval

$decision = Get-Content diagnostics\ecir\stage2_heterogeneous_eval\result.json -Raw | ConvertFrom-Json
if ($decision.status -ne "GO") {
    Write-Host "ECIR Stage 2 NO_GO. Stage 3 and formal-large are blocked."
    exit 3
}

Write-Host "ECIR Stage 2 GO. Review before executing:"
Write-Host "$Python scripts\train_ecir_flow.py --config configs\ecir_flow_formal_medium.yaml --cache_dir `"$CacheDir`" --target_cache_dir `"$AtlasDir\targets`""
Write-Host "After an independent Stage 3 GO:"
Write-Host "$Python scripts\train_ecir_flow.py --config configs\ecir_flow_formal_large.yaml --cache_dir `"$CacheDir`" --target_cache_dir `"$AtlasDir\targets`""
