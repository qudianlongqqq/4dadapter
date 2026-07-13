#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
LOG_ROOT="logs_formal_large"; DIAG="diagnostics/formal_large/confirm30"
MANIFEST="manifests/formal_large_val_confirm30.json"
INFERENCE="data/flexbond_inference_formal_large"; REFERENCE="data/flexbond_cache_formal_large"
SCREEN="reports/formal_large_screen10.json"
CONFIRM_MAX_RECORDS="${CONFIRM_MAX_RECORDS:-600}"
[[ -e "${LOG_ROOT}/FORMAL_LARGE_SCREEN10_COMPLETED" && -s "${SCREEN}" ]] || {
  echo "screen10 is incomplete"; exit 2;
}
mkdir -p "${DIAG}" reports
touch "${LOG_ROOT}/FORMAL_LARGE_CONFIRM30_RUNNING"
if [[ ! -s "${MANIFEST}" ]]; then
  python scripts/formal_large_selection.py create-manifest --kind confirm30 \
    --source manifests/formal_large_val.json --output "${MANIFEST}" \
    --max_records "${CONFIRM_MAX_RECORDS}"
fi

for method in cartesian global4d; do
  run="${LOG_ROOT}/${method}_seed42_200k"
  [[ "${method}" == global4d ]] && run="${LOG_ROOT}/global4d_seed42_200k"
  for rank in 0 1; do
    read -r step alpha <<< "$(python -c 'import json,sys; r=json.load(open(sys.argv[1],encoding="utf-8"))["top2"][sys.argv[2]][int(sys.argv[3])]; print(r["checkpoint_step"],r["alpha"])' "${SCREEN}" "${method}" "${rank}")"
    alpha_code="$(python -c 'import sys; print("02" if abs(float(sys.argv[1])-.2)<1e-9 else "05")' "${alpha}")"
    group="${DIAG}/${method}/step${step}_alpha${alpha_code}"
    checkpoint="${run}/checkpoints/step${step}.ckpt"; samples="${group}/samples.pt"
    evaluation="${group}/eval"; mkdir -p "${group}"
    if [[ ! -s "${samples}" ]]; then
      if [[ "${method}" == cartesian ]]; then
        python scripts/sample_formal_large_cartesian.py --checkpoint "${checkpoint}" \
          --config "${run}/config.resolved.yaml" --cache_dir "${INFERENCE}" \
          --manifest "${MANIFEST}" --split val --output "${samples}" \
          --update_scale "${alpha}" --device auto
      else
        python scripts/sample_global_coupled_4d_flow.py --checkpoint "${checkpoint}" \
          --config "${run}/config.resolved.yaml" --cache_dir "${INFERENCE}" \
          --manifest "${MANIFEST}" --split val --output "${samples}" \
          --update_scale "${alpha}" --device auto
      fi
    fi
    if [[ ! -s "${evaluation}/summary.csv" ]]; then
      sample_flag="--cartesian_samples"; [[ "${method}" == global4d ]] && sample_flag="--global_coupled_4d_samples"
      python scripts/eval_flexbond_optimizer.py --manifest "${MANIFEST}" \
        --inference_cache "${INFERENCE}" --reference_cache "${REFERENCE}" --split val \
        "${sample_flag}" "${samples}" --output_dir "${evaluation}" --threshold 1.25
    fi
  done
done
python scripts/formal_large_selection.py summarize --kind confirm30 --root "${DIAG}" \
  --output reports/formal_large_confirm30.csv \
  --best_configs reports/formal_large_best_configs.json
touch "${LOG_ROOT}/FORMAL_LARGE_CONFIRM30_COMPLETED"
rm -f "${LOG_ROOT}/FORMAL_LARGE_CONFIRM30_RUNNING"
