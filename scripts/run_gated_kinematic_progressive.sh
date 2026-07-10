#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CONFIG="${CONFIG:-configs/gated_kinematic_local025_small.yaml}"
CACHE="${CACHE:-data/flexbond_cache_formal_small}"
INFERENCE_CACHE="${INFERENCE_CACHE:-data/flexbond_inference_formal_small}"
MANIFEST="${MANIFEST:-eval_manifest_formal_small.json}"
SEED="${SEED:-42}"
STEPS="${STEPS:-2000}"
DEVICE="${DEVICE:-cuda}"
TRAIN_DIR="logs_gated_kinematic/gated_torsion_local025_seed${SEED}_${STEPS}step"
BASIS_DIR="diagnostics/gated_kinematic_basis/basis_local025_seed${SEED}_500samples"

python scripts/diagnose_gated_kinematic_basis.py \
  --cache_dir "${CACHE}" --split val --max_samples 500 \
  --fixed_times 0.05 0.1 0.25 --seed "${SEED}" --device "${DEVICE}" \
  --resume --output_dir "${BASIS_DIR}"

python scripts/train_gated_kinematic_flow.py \
  --config "${CONFIG}" --cache_dir "${CACHE}" --max_steps "${STEPS}" \
  --seed "${SEED}" --output_dir "${TRAIN_DIR}"

CHECKPOINT="${TRAIN_DIR}/checkpoints/last.ckpt"
for ALPHA in 0.2 0.5; do
  TAG="${ALPHA/./p}"
  EVAL_DIR="diagnostics/gated_kinematic_eval/gated_torsion_local025_seed${SEED}_step${STEPS}_alpha${TAG}_gatelearned"
  python scripts/sample_gated_kinematic_flow.py \
    --checkpoint "${CHECKPOINT}" --config "${TRAIN_DIR}/config.resolved.yaml" \
    --cache_dir "${INFERENCE_CACHE}" --manifest "${MANIFEST}" --device "${DEVICE}" \
    --refinement_steps 10 --update_scale "${ALPHA}" --max_displacement 0.1 \
    --gate_override none --save_trajectory_metrics --output "${EVAL_DIR}/samples.pt"
  python scripts/eval_gated_kinematic_flow.py \
    --manifest "${MANIFEST}" --inference_cache "${INFERENCE_CACHE}" \
    --reference_cache "${CACHE}" --gated_samples "${EVAL_DIR}/samples.pt" \
    --output_dir "${EVAL_DIR}/evaluation"
done

python scripts/report_gated_kinematic_progress.py
