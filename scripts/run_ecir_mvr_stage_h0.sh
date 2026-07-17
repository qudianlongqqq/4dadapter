#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
CONFIG="configs/ecir_mvr_stage_h0_conflict_fusion.yaml"
DEVICE="cuda"
RECORD_BATCH_SIZE=128
OUTPUT_DIR="diagnostics/ecir_mvr/stage_h0"
CONFIRM=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --confirm-formal) CONFIRM=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --record-batch-size) RECORD_BATCH_SIZE="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --profile-cuda-memory) EXTRA+=(--profile-cuda-memory); shift ;;
    --profile-every-records) EXTRA+=(--profile-every-records "$2"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "${CONFIRM}" -eq 1 ]] || { echo "Stage H0 requires --confirm-formal" >&2; exit 2; }
"${PYTHON}" -u scripts/evaluate_ecir_mvr_stage_h0.py --config "${CONFIG}" --device "${DEVICE}" \
  --record-batch-size "${RECORD_BATCH_SIZE}" --output-dir "${OUTPUT_DIR}" --confirm-formal "${EXTRA[@]}"
