#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
LOG_ROOT="logs_global_coupled_4d"
RUN_DIR="${LOG_ROOT}/global4d_local025_seed42_5000step"
OUTPUT="diagnostics/global_coupled_4d/ablation_5k"
SWEEP="diagnostics/global_coupled_4d/checkpoint_sweep_5k"
MANIFEST="${GLOBAL4D_MANIFEST:-eval_manifest_formal_small.json}"
INFERENCE="${GLOBAL4D_INFERENCE_CACHE:-data/flexbond_inference_formal_small}"
REFERENCE="${GLOBAL4D_REFERENCE_CACHE:-data/flexbond_cache_formal_small}"
mkdir -p "${OUTPUT}"
BEST_NAME="$(python -c 'import json;print(json.load(open("diagnostics/global_coupled_4d/checkpoint_sweep_5k/best_checkpoint.json"))["checkpoint_name"])')"
CHECKPOINT="${RUN_DIR}/checkpoints/${BEST_NAME}.ckpt"
[[ -s "${CHECKPOINT}" ]] || { echo "Best checkpoint missing: ${CHECKPOINT}"; exit 2; }
printf '%s\n' "ABLATION" > "${LOG_ROOT}/CURRENT_STAGE"

for mode in full_4d torsion_only bending_torsion angular_only stretch_only internal_zero; do
  for alpha_code in 02 05; do
    alpha="0.${alpha_code#0}"; group="${OUTPUT}/${mode}_${BEST_NAME}_alpha${alpha_code}"
    mkdir -p "${group}"
    if [[ ! -s "${group}/samples.pt" ]]; then
      python scripts/sample_global_coupled_4d_flow.py \
        --checkpoint "${CHECKPOINT}" --config "${RUN_DIR}/config.resolved.yaml" \
        --cache_dir "${INFERENCE}" --manifest "${MANIFEST}" --split test \
        --output "${group}/samples.pt" --update_scale "${alpha}" --joint_mode "${mode}" &
      echo $! > "${LOG_ROOT}/SAMPLE.pid"; wait "$(cat "${LOG_ROOT}/SAMPLE.pid")"; rm -f "${LOG_ROOT}/SAMPLE.pid"
    fi
    if [[ ! -s "${group}/eval/summary.csv" ]]; then
      python scripts/eval_global_coupled_4d_flow.py \
        --manifest "${MANIFEST}" --inference_cache "${INFERENCE}" \
        --reference_cache "${REFERENCE}" --split test --samples "${group}/samples.pt" \
        --output_dir "${group}/eval" --threshold 1.25 &
      echo $! > "${LOG_ROOT}/EVAL.pid"; wait "$(cat "${LOG_ROOT}/EVAL.pid")"; rm -f "${LOG_ROOT}/EVAL.pid"
    fi
  done
done
python scripts/summarize_global_coupled_4d_evaluations.py --root "${OUTPUT}" --output_dir "${OUTPUT}"
cp "${OUTPUT}/comparison.csv" "${OUTPUT}/comparison_full.csv"
touch "${OUTPUT}/COMPLETED"
