#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_flexbond_optimizer_smoke.sh CACHE_DIR [OUTPUT_ROOT]"
  exit 2
fi

CACHE_DIR="$1"
OUTPUT_ROOT="${2:-logs_flexbond_optimizer}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/smoke_${STAMP}"
CONFIG="configs/flexbond_optimizer_egnn.yaml"

python scripts/check_flexbond_data_pairs.py \
  --cache_dir "${CACHE_DIR}" --split train --num_samples 5
python scripts/check_flexbond_jacobian.py \
  --cache_dir "${CACHE_DIR}" --split train

for MODE in cartesian_optimizer flexbond4d_hybrid_optimizer; do
  MODE_DIR="${RUN_DIR}/${MODE}"
  python scripts/train_flexbond_optimizer.py \
    --config "${CONFIG}" \
    --mode "${MODE}" \
    --cache_dir "${CACHE_DIR}" \
    --output_dir "${MODE_DIR}" \
    --max_molecules 100 \
    --max_steps 5000
  python scripts/check_flexbond_equivariance.py \
    --cache_dir "${CACHE_DIR}" \
    --split test \
    --checkpoint "${MODE_DIR}/checkpoints/last.ckpt"
  python scripts/sample_flexbond_optimizer.py \
    --checkpoint "${MODE_DIR}/checkpoints/last.ckpt" \
    --config "${MODE_DIR}/config.resolved.yaml" \
    --cache_dir "${CACHE_DIR}" \
    --split test \
    --max_molecules 100 \
    --refinement_steps 10 \
    --output "${MODE_DIR}/samples.pt"
  python scripts/eval_flexbond_optimizer.py \
    --samples "${MODE_DIR}/samples.pt" \
    --output_dir "${MODE_DIR}/evaluation"
done

python scripts/summarize_flexbond_optimizer.py \
  "${RUN_DIR}/cartesian_optimizer/evaluation" \
  "${RUN_DIR}/flexbond4d_hybrid_optimizer/evaluation" \
  --output_dir "${RUN_DIR}"

echo "Smoke run complete: ${RUN_DIR}"
