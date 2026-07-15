#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
: "${GLOBAL4D_REFERENCE_CACHE:?Set GLOBAL4D_REFERENCE_CACHE to the training/reference cache}"

RUN_DIR="${GLOBAL4D_PILOT_RUN_DIR:-logs_gated_global4d_v2/pilot_seed42_2k}"
CONFIG="configs/gated_global4d_v2_pilot.yaml"
mkdir -p "${RUN_DIR}"

python scripts/train_gated_global4d_v2.py \
  --config "${CONFIG}" \
  --cache_dir "${GLOBAL4D_REFERENCE_CACHE}" \
  --output_dir "${RUN_DIR}" \
  --mode formal \
  --max_steps 2000 \
  --batch_size 8 \
  --accumulate_grad_batches 1 \
  --checkpoint_steps 500,1000,1500,2000 \
  --resume_from_checkpoint auto

echo "GATED GLOBAL4D V2 2K PILOT COMPLETED"
