#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
LOG_ROOT="logs_formal_large"; DIAG="diagnostics/formal_large/screen10"
SOURCE_MANIFEST="manifests/formal_large_val.json"
MANIFEST="manifests/formal_large_val_screen10.json"
INFERENCE="data/flexbond_inference_formal_large"
REFERENCE="data/flexbond_cache_formal_large"
SCREEN_MAX_RECORDS="${SCREEN_MAX_RECORDS:-200}"
[[ -e "${LOG_ROOT}/FORMAL_LARGE_TRAINING_COMPLETED" ]] || { echo "Training is incomplete"; exit 2; }
mkdir -p "${DIAG}" reports
touch "${LOG_ROOT}/FORMAL_LARGE_SCREEN10_RUNNING"
if [[ ! -s "${MANIFEST}" ]]; then
  python scripts/formal_large_selection.py create-manifest \
    --kind screen10 --source "${SOURCE_MANIFEST}" --output "${MANIFEST}" \
    --max_records "${SCREEN_MAX_RECORDS}"
fi

for method in cartesian global4d; do
  if [[ "${method}" == cartesian ]]; then
    run="${LOG_ROOT}/cartesian_seed42_200k"
  else
    run="${LOG_ROOT}/global4d_seed42_200k"
  fi
  for step in 50000 100000 150000 200000; do
    checkpoint="${run}/checkpoints/step${step}.ckpt"
    [[ -s "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}"; exit 2; }
    for alpha_code in 02 05; do
      alpha="0.${alpha_code#0}"; group="${DIAG}/${method}/step${step}_alpha${alpha_code}"
      samples="${group}/samples.pt"; evaluation="${group}/eval"; mkdir -p "${group}"
      if [[ ! -s "${samples}" ]]; then
        if [[ "${method}" == cartesian ]]; then
          python scripts/sample_formal_large_cartesian.py \
            --checkpoint "${checkpoint}" --config "${run}/config.resolved.yaml" \
            --cache_dir "${INFERENCE}" --manifest "${MANIFEST}" --split val \
            --output "${samples}" --update_scale "${alpha}" --device auto
        else
          python scripts/sample_global_coupled_4d_flow.py \
            --checkpoint "${checkpoint}" --config "${run}/config.resolved.yaml" \
            --cache_dir "${INFERENCE}" --manifest "${MANIFEST}" --split val \
            --output "${samples}" --update_scale "${alpha}" --device auto
        fi
      fi
      if [[ ! -s "${evaluation}/summary.csv" ]]; then
        sample_flag="--${method}_samples"
        [[ "${method}" == global4d ]] && sample_flag="--global_coupled_4d_samples"
        [[ "${method}" == cartesian ]] && sample_flag="--cartesian_samples"
        python scripts/eval_flexbond_optimizer.py \
          --manifest "${MANIFEST}" --inference_cache "${INFERENCE}" \
          --reference_cache "${REFERENCE}" --split val "${sample_flag}" "${samples}" \
          --output_dir "${evaluation}" --threshold 1.25
      fi
    done
  done
done
python scripts/formal_large_selection.py summarize --kind screen10 --root "${DIAG}" \
  --output reports/formal_large_screen10.csv
touch "${LOG_ROOT}/FORMAL_LARGE_SCREEN10_COMPLETED"
rm -f "${LOG_ROOT}/FORMAL_LARGE_SCREEN10_RUNNING"
