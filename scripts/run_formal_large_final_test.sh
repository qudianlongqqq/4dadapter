#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
LOG_ROOT="logs_formal_large"; DIAG="diagnostics/formal_large/final_test"
BEST="reports/formal_large_best_configs.json"
VAL_MANIFEST="manifests/formal_large_val_confirm30.json"
TEST_MANIFEST="manifests/formal_large_test.json"
INFERENCE="data/flexbond_inference_formal_large"; REFERENCE="data/flexbond_cache_formal_large"
for marker in FORMAL_LARGE_TRAINING_COMPLETED FORMAL_LARGE_SCREEN10_COMPLETED FORMAL_LARGE_CONFIRM30_COMPLETED; do
  [[ -e "${LOG_ROOT}/${marker}" ]] || { echo "Missing gate: ${marker}"; exit 2; }
done
[[ -s "${BEST}" ]] || { echo "Missing frozen best configs"; exit 2; }
mkdir -p "${DIAG}" reports
touch "${LOG_ROOT}/FORMAL_LARGE_FINAL_TEST_RUNNING"
python scripts/verify_formal_large_best_configs.py --best "${BEST}" \
  --validation_manifest "${VAL_MANIFEST}" --test_manifest "${TEST_MANIFEST}" \
  --lock "${DIAG}/final_test_lock.json"

for method in cartesian global4d; do
  read -r checkpoint config alpha steps <<< "$(python -c 'import json,sys; c=json.load(open(sys.argv[1],encoding="utf-8"))["configs"][sys.argv[2]]; print(c["checkpoint_path"],c["config_path"],c["alpha"],c["refinement_steps"])' "${BEST}" "${method}")"
  group="${DIAG}/${method}"; samples="${group}/samples.pt"; mkdir -p "${group}"
  if [[ ! -s "${samples}" ]]; then
    if [[ "${method}" == cartesian ]]; then
      python scripts/sample_formal_large_cartesian.py --checkpoint "${checkpoint}" \
        --config "${config}" --cache_dir "${INFERENCE}" --manifest "${TEST_MANIFEST}" \
        --split test --output "${samples}" --update_scale "${alpha}" \
        --refinement_steps "${steps}" --device auto
    else
      python scripts/sample_global_coupled_4d_flow.py --checkpoint "${checkpoint}" \
        --config "${config}" --cache_dir "${INFERENCE}" --manifest "${TEST_MANIFEST}" \
        --split test --output "${samples}" --update_scale "${alpha}" \
        --refinement_steps "${steps}" --device auto
    fi
  fi
done

EVAL="${DIAG}/eval"
if [[ ! -s "${EVAL}/summary.csv" ]]; then
  python scripts/eval_flexbond_optimizer.py --manifest "${TEST_MANIFEST}" \
    --inference_cache "${INFERENCE}" --reference_cache "${REFERENCE}" --split test \
    --cartesian_samples "${DIAG}/cartesian/samples.pt" \
    --global_coupled_4d_samples "${DIAG}/global4d/samples.pt" \
    --output_dir "${EVAL}" --threshold 1.25
fi
python scripts/diagnose_flexbond_diversity.py --manifest "${TEST_MANIFEST}" \
  --inference_cache "${INFERENCE}" --reference_cache "${REFERENCE}" --split test \
  --cartesian_samples "${DIAG}/cartesian/samples.pt" \
  --global_coupled_4d_samples "${DIAG}/global4d/samples.pt" \
  --output_dir "${DIAG}/diversity" --threshold 1.25
python scripts/report_formal_large_final_test.py --summary "${EVAL}/summary.csv" \
  --diversity "${DIAG}/diversity/diversity_summary.csv" \
  --cartesian_samples "${DIAG}/cartesian/samples.pt" \
  --global4d_samples "${DIAG}/global4d/samples.pt" \
  --output reports/formal_large_final_test.csv
touch "${LOG_ROOT}/FORMAL_LARGE_FINAL_TEST_COMPLETED"
rm -f "${LOG_ROOT}/FORMAL_LARGE_FINAL_TEST_RUNNING"
