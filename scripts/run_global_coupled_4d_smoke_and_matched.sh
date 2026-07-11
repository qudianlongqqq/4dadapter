#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
LOG_ROOT="logs_global_coupled_4d"
MASTER_LOG="${LOG_ROOT}/global_coupled_4d_master.log"
MANIFEST="${GLOBAL4D_MANIFEST:-eval_manifest_formal_small.json}"
INFERENCE="${GLOBAL4D_INFERENCE_CACHE:-data/flexbond_inference_formal_small}"
REFERENCE="${GLOBAL4D_REFERENCE_CACHE:-data/flexbond_cache_formal_small}"
mkdir -p "${LOG_ROOT}" reports diagnostics/global_coupled_4d
echo $$ > "${LOG_ROOT}/MASTER.pid"
printf 'pid=%s\nstarted_at=%s\n' "$$" "$(date --iso-8601=seconds)" > "${LOG_ROOT}/RUNNING"
rm -f "${LOG_ROOT}/COMPLETED" "${LOG_ROOT}/FAILED"
STAGE="CHECK"

fail() {
  code=$?; command_text="${BASH_COMMAND}"; tail_text="$(tail -100 "${MASTER_LOG}" 2>/dev/null || true)"
  python -c 'import json,sys,datetime,pathlib; pathlib.Path(sys.argv[1]).write_text(json.dumps({"stage":sys.argv[2],"time":datetime.datetime.now().astimezone().isoformat(),"command":sys.argv[3],"exit_code":int(sys.argv[4]),"log":sys.argv[5],"tail":sys.argv[6]},indent=2),encoding="utf-8")' \
    "${LOG_ROOT}/FAILED" "${STAGE}" "${command_text}" "${code}" "${MASTER_LOG}" "${tail_text}"
  rm -f "${LOG_ROOT}/RUNNING" "${LOG_ROOT}/MASTER.pid"
  exit "${code}"
}
trap fail ERR

printf '%s\n' "CHECK" > "${LOG_ROOT}/CURRENT_STAGE"
for path in "${MANIFEST}" "${INFERENCE}" "${REFERENCE}"; do
  [[ -e "${path}" ]] || { echo "Missing required new-model input: ${path}"; exit 2; }
done
python scripts/extract_reference_4d_training_budget.py \
  --reference_run logs_flexbond_formal_small/flexbond4d_hybrid_5k
python scripts/validate_global_coupled_4d_budget.py
touch "${LOG_ROOT}/BUDGET_MATCHED"

if [[ ! -e "${LOG_ROOT}/TESTS_PASSED" ]]; then
  STAGE="TEST"; printf '%s\n' "TEST" > "${LOG_ROOT}/CURRENT_STAGE"
  python -m pytest -q \
    tests/test_global_coupled_4d_topology.py \
    tests/test_global_coupled_4d_jacobian.py \
    tests/test_global_coupled_4d_projection.py \
    tests/test_global_coupled_4d_flow.py | tee "${LOG_ROOT}/unit_tests.log"
  touch "${LOG_ROOT}/TESTS_PASSED"
fi

ORACLE="diagnostics/global_coupled_4d/oracle"
if [[ ! -e "${LOG_ROOT}/ORACLE_PASSED" || ! -s "${ORACLE}/summary.csv" ]]; then
  STAGE="ORACLE"; printf '%s\n' "ORACLE" > "${LOG_ROOT}/CURRENT_STAGE"
  python scripts/diagnose_global_coupled_4d_oracle.py \
    --manifest "${MANIFEST}" --inference_cache "${INFERENCE}" \
    --reference_cache "${REFERENCE}" --split test --max_molecules 200 \
    --output_dir "${ORACLE}"
  python -c 'import csv,math; rows=list(csv.DictReader(open("diagnostics/global_coupled_4d/oracle/summary.csv",encoding="utf-8-sig"))); assert rows and all(math.isfinite(float(v)) for r in rows for k,v in r.items() if k!="subset" and v not in ("",None))'
  touch "${LOG_ROOT}/ORACLE_PASSED"
fi

if [[ ! -e "${LOG_ROOT}/SMOKE_COMPLETED" \
   || ! -s "${LOG_ROOT}/smoke200/checkpoints/step200.ckpt" \
   || ! -s "diagnostics/global_coupled_4d/smoke200/step200_alpha05_eval/summary.csv" ]]; then
  STAGE="SMOKE"; printf '%s\n' "SMOKE" > "${LOG_ROOT}/CURRENT_STAGE"
  bash scripts/run_global_coupled_4d_smoke.sh
fi

if [[ ! -e "${LOG_ROOT}/FORMAL_COMPLETED" \
   || ! -s "${LOG_ROOT}/global4d_local025_seed42_5000step/checkpoints/step5000.ckpt" \
   || ! -s "diagnostics/global_coupled_4d/checkpoint_sweep_5k/best_checkpoint.json" \
   || ! -e "diagnostics/global_coupled_4d/ablation_5k/COMPLETED" ]]; then
  STAGE="FORMAL_TRAIN"; printf '%s\n' "FORMAL_TRAIN" > "${LOG_ROOT}/CURRENT_STAGE"
  bash scripts/run_global_coupled_4d_formal_matched.sh
fi

python scripts/report_global_coupled_4d_5k_comparison.py
printf '%s\n' "COMPLETED" > "${LOG_ROOT}/CURRENT_STAGE"
rm -f "${LOG_ROOT}/RUNNING" "${LOG_ROOT}/FAILED" "${LOG_ROOT}/MASTER.pid"
touch "${LOG_ROOT}/COMPLETED"
echo "GLOBAL COUPLED 4D SMOKE + MATCHED 5K COMPLETED"
