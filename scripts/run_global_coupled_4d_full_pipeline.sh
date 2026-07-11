#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
LOG_ROOT="logs_global_coupled_4d"
MASTER_LOG="${LOG_ROOT}/global_coupled_4d_master.log"
mkdir -p "${LOG_ROOT}" reports diagnostics/global_coupled_4d
echo $$ > "${LOG_ROOT}/MASTER.pid"
STAGE="PRE_AUDIT"

fail() {
  code=$?; command_text="${BASH_COMMAND}"; tail_text="$(tail -100 "${MASTER_LOG}" 2>/dev/null || true)"
  python -c 'import json,sys,datetime,pathlib; pathlib.Path(sys.argv[1]).write_text(json.dumps({"stage":sys.argv[2],"time":datetime.datetime.now().astimezone().isoformat(),"command":sys.argv[3],"exit_code":int(sys.argv[4]),"log":sys.argv[5],"tail":sys.argv[6]},indent=2),encoding="utf-8")' \
    "${LOG_ROOT}/FAILED" "${STAGE}" "${command_text}" "${code}" "${MASTER_LOG}" "${tail_text}"
  rm -f "${LOG_ROOT}/MASTER.pid"
  exit "${code}"
}
trap fail ERR
exec >> "${MASTER_LOG}" 2>&1

if [[ ! -e "${LOG_ROOT}/PRE_AUDIT_PASSED" ]]; then
  STAGE="PRE_AUDIT"
  python scripts/extract_reference_4d_training_budget.py
  grep -Eq '^Decision: \*\*(GO|GO_WITH_REQUIRED_FIXES)\*\*$' reports/global_coupled_4d_preimplementation_audit.md
  touch "${LOG_ROOT}/PRE_AUDIT_PASSED"
fi

if [[ ! -e "${LOG_ROOT}/TESTS_PASSED" ]]; then
  STAGE="TEST"
  python -m pytest -q tests/test_global_coupled_4d_topology.py \
    tests/test_global_coupled_4d_jacobian.py \
    tests/test_global_coupled_4d_projection.py \
    tests/test_global_coupled_4d_flow.py | tee "${LOG_ROOT}/unit_tests.log"
  touch "${LOG_ROOT}/TESTS_PASSED"
fi

if [[ ! -e "${LOG_ROOT}/ORACLE_PASSED" ]]; then
  STAGE="ORACLE"
  python scripts/diagnose_global_coupled_4d_oracle.py \
    --manifest "${GLOBAL4D_MANIFEST:-eval_manifest_formal_small.json}" \
    --inference_cache "${GLOBAL4D_INFERENCE_CACHE:-data/flexbond_inference_formal_small}" \
    --reference_cache "${GLOBAL4D_REFERENCE_CACHE:-data/flexbond_cache_formal_small}" \
    --split test --max_molecules 200 --output_dir diagnostics/global_coupled_4d/oracle
  python -c 'import csv,math; rows=list(csv.DictReader(open("diagnostics/global_coupled_4d/oracle/summary.csv",encoding="utf-8-sig"))); assert rows and all(math.isfinite(float(v)) for r in rows for k,v in r.items() if k not in ("subset",) and v not in ("",None))'
  touch "${LOG_ROOT}/ORACLE_PASSED"
fi

if [[ ! -e "${LOG_ROOT}/SMOKE_PASSED" ]]; then
  STAGE="SMOKE"
  bash scripts/run_global_coupled_4d_smoke.sh
fi

if [[ ! -e "${LOG_ROOT}/POST_REVIEW_PASSED" ]]; then
  STAGE="POST_REVIEW"
  grep -Eq '^Conclusion: PASS$|^Conclusion: PASS_WITH_WARNINGS$' reports/global_coupled_4d_postimplementation_review.md
  touch "${LOG_ROOT}/POST_REVIEW_PASSED"
fi

STAGE="FORMAL"
bash scripts/run_global_coupled_4d_formal_matched.sh
python scripts/report_global_coupled_4d_comparison.py
touch "${LOG_ROOT}/COMPLETED"
rm -f "${LOG_ROOT}/FAILED" "${LOG_ROOT}/MASTER.pid"
echo "GLOBAL COUPLED 4D FULL PIPELINE COMPLETED"
