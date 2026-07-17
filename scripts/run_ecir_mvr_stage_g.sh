#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-python}"
CONFIG="configs/ecir_mvr_stage_g_bounded_residual.yaml"
OUTPUT_ROOT="diagnostics/ecir_mvr/stage_g"
DEVICE="cuda"
SEED=42
BUILDER_BATCH_SIZE=128
BATCH_SIZE=131072
DATASET_RESIDENCY="auto"
NUM_WORKERS=0
PROFILE_CUDA_MEMORY=0
PROFILE_EVERY_STEPS=100
CONFIRM_FORMAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --builder-batch-size) BUILDER_BATCH_SIZE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --dataset-residency) DATASET_RESIDENCY="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --profile-cuda-memory) PROFILE_CUDA_MEMORY=1; shift ;;
    --profile-every-steps) PROFILE_EVERY_STEPS="$2"; shift 2 ;;
    --confirm-formal) CONFIRM_FORMAL=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "${CONFIRM_FORMAL}" -ne 1 ]]; then
  echo "Stage G formal execution requires --confirm-formal" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
COMMIT_SHA="$(git rev-parse HEAD)"
LOG_PATH="${OUTPUT_ROOT}/stage_g_${TIMESTAMP}_seed${SEED}.log"

run_pipeline() {
  echo "timestamp=${TIMESTAMP}"
  echo "seed=${SEED}"
  echo "commit_sha=${COMMIT_SHA}"
  echo "device=${DEVICE}"
  echo "builder_batch_size=${BUILDER_BATCH_SIZE}"
  echo "calibrator_batch_size=${BATCH_SIZE}"

  "${PYTHON}" -u scripts/build_ecir_mvr_stage_g_calibration_data.py \
    --config "${CONFIG}" --output-dir "${OUTPUT_ROOT}" --device "${DEVICE}" \
    --seed "${SEED}" --builder-batch-size "${BUILDER_BATCH_SIZE}"

  FIT_ARGS=(
    scripts/fit_ecir_mvr_stage_g_calibrator.py
    --config "${CONFIG}"
    --input-dir "${OUTPUT_ROOT}"
    --output-dir "${OUTPUT_ROOT}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --batch-size "${BATCH_SIZE}"
    --dataset-residency "${DATASET_RESIDENCY}"
    --num-workers "${NUM_WORKERS}"
    --profile-every-steps "${PROFILE_EVERY_STEPS}"
    --pin-memory
  )
  if [[ "${PROFILE_CUDA_MEMORY}" -eq 1 ]]; then
    FIT_ARGS+=(--profile-cuda-memory)
  fi
  "${PYTHON}" -u "${FIT_ARGS[@]}"

  FIT_DECISION="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["decision"])' "${OUTPUT_ROOT}/fit_result.json")"
  if [[ "${FIT_DECISION}" == "STAGE_G_COLLAPSED" ]]; then
    echo "Stage G stopped: every preregistered checkpoint collapsed."
    return 0
  fi

  "${PYTHON}" -u scripts/evaluate_ecir_mvr_stage_g.py \
    --config "${CONFIG}" --input-dir "${OUTPUT_ROOT}" --output-dir "${OUTPUT_ROOT}" \
    --device "${DEVICE}" --seed "${SEED}"
  echo "Stage G complete; Stage F unchanged; no test or long neural training was run."
}

run_pipeline 2>&1 | tee "${LOG_PATH}"
