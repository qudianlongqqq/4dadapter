#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_flexbond_optimizer_smoke.sh TRAIN_CACHE INFERENCE_CACHE [OUTPUT_ROOT]"
  exit 2
fi

CACHE_DIR="$1"
INFERENCE_CACHE="$2"
OUTPUT_ROOT="${3:-logs_flexbond_optimizer}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/smoke_${STAMP}"
CONFIG="configs/flexbond_optimizer_egnn.yaml"
MANIFEST="${RUN_DIR}/eval_manifest.json"

python scripts/check_flexbond_data_pairs.py \
  --cache_dir "${CACHE_DIR}" --split train --num_samples 5
python scripts/check_flexbond_jacobian.py \
  --cache_dir "${CACHE_DIR}" --split train
python scripts/check_flexbond_graph_consistency.py \
  --cache_dir "${CACHE_DIR}" --split train
python scripts/check_flexbond_inference_no_labels.py \
  --cache_dir "${INFERENCE_CACHE}" --split test
python scripts/build_flexbond_eval_manifest.py \
  --cache_dir "${INFERENCE_CACHE}" --split test --max_molecules 100 --output "${MANIFEST}"

for MODE in cartesian_optimizer flexbond4d_hybrid_optimizer; do
  MODE_DIR="${RUN_DIR}/${MODE}"
  python scripts/train_flexbond_optimizer.py \
    --config "${CONFIG}" \
    --mode "${MODE}" \
    --cache_dir "${INFERENCE_CACHE}" \
    --manifest "${MANIFEST}" \
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
done

python scripts/check_flexbond_eval_cohort.py \
  --manifest "${MANIFEST}" \
  --cartesian_samples "${RUN_DIR}/cartesian_optimizer/samples.pt" \
  --flexbond_samples "${RUN_DIR}/flexbond4d_hybrid_optimizer/samples.pt"
python scripts/eval_flexbond_optimizer.py \
  --manifest "${MANIFEST}" \
  --inference_cache "${INFERENCE_CACHE}" \
  --reference_cache "${CACHE_DIR}" \
  --split test \
  --cartesian_samples "${RUN_DIR}/cartesian_optimizer/samples.pt" \
  --flexbond_samples "${RUN_DIR}/flexbond4d_hybrid_optimizer/samples.pt" \
  --output_dir "${RUN_DIR}/evaluation"

echo "Smoke run complete: ${RUN_DIR}"
