#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
LOG_ROOT="logs_global_coupled_4d"
RUN_DIR="${LOG_ROOT}/formal_matched"
SWEEP="diagnostics/global_coupled_4d/checkpoint_sweep"
CONFIG="configs/global_coupled_4d_local025_matched.yaml"
MANIFEST="${GLOBAL4D_MANIFEST:-eval_manifest_formal_small.json}"
INFERENCE="${GLOBAL4D_INFERENCE_CACHE:-data/flexbond_inference_formal_small}"
REFERENCE="${GLOBAL4D_REFERENCE_CACHE:-data/flexbond_cache_formal_small}"
mkdir -p "${RUN_DIR}" "${SWEEP}" "${LOG_ROOT}"
STAGE="FORMAL"

fail() {
  code=$?; command_text="${BASH_COMMAND}"
  tail_text="$(tail -100 "${LOG_ROOT}/global_coupled_4d_master.log" 2>/dev/null || tail -100 "${RUN_DIR}/formal.log" 2>/dev/null || true)"
  python -c 'import json,sys,datetime,pathlib; pathlib.Path(sys.argv[1]).write_text(json.dumps({"stage":sys.argv[2],"time":datetime.datetime.now().astimezone().isoformat(),"command":sys.argv[3],"exit_code":int(sys.argv[4]),"log":sys.argv[5],"tail":sys.argv[6]},indent=2),encoding="utf-8")' \
    "${LOG_ROOT}/FAILED" "${STAGE}" "${command_text}" "${code}" "${RUN_DIR}/formal.log" "${tail_text}"
  exit "${code}"
}
trap fail ERR
exec > >(tee -a "${RUN_DIR}/formal.log") 2>&1

for marker in PRE_AUDIT_PASSED TESTS_PASSED ORACLE_PASSED SMOKE_PASSED POST_REVIEW_PASSED; do
  [[ -e "${LOG_ROOT}/${marker}" ]] || { echo "Missing formal gate: ${marker}"; exit 2; }
done
python scripts/validate_global_coupled_4d_budget.py --config "${CONFIG}"
touch "${LOG_ROOT}/BUDGET_MATCHED" "${LOG_ROOT}/FORMAL_RUNNING"
MAX_STEPS="$(python -c 'import json;print(json.load(open("reports/reference_4d_training_budget.json"))["max_steps"])')"

FINAL_CHECKPOINT="${RUN_DIR}/checkpoints/step${MAX_STEPS}.ckpt"
if [[ ! -s "${FINAL_CHECKPOINT}" ]]; then
  python scripts/train_global_coupled_4d_flow.py \
    --config "${CONFIG}" --cache_dir "${REFERENCE}" --output_dir "${RUN_DIR}" \
    --mode formal --max_steps "${MAX_STEPS}" --resume_from_checkpoint auto &
  echo $! > "${LOG_ROOT}/TRAIN.pid"
  wait "$(cat "${LOG_ROOT}/TRAIN.pid")"
  rm -f "${LOG_ROOT}/TRAIN.pid"
fi

for checkpoint in "${RUN_DIR}"/checkpoints/step*.ckpt; do
  [[ -s "${checkpoint}" ]] || continue
  step="$(basename "${checkpoint}" .ckpt)"
  for alpha_code in 02 05; do
    alpha="0.${alpha_code#0}"
    group="${SWEEP}/${step}_alpha${alpha_code}"
    samples="${group}/samples.pt"
    evaluation="${group}/eval"
    mkdir -p "${group}"
    if [[ ! -s "${samples}" ]]; then
      python scripts/sample_global_coupled_4d_flow.py \
        --checkpoint "${checkpoint}" --config "${RUN_DIR}/config.resolved.yaml" \
        --cache_dir "${INFERENCE}" --manifest "${MANIFEST}" --split test \
        --output "${samples}" --max_molecules 100 --update_scale "${alpha}"
    fi
    if [[ ! -s "${evaluation}/summary.csv" ]]; then
      python scripts/eval_global_coupled_4d_flow.py \
        --manifest "${MANIFEST}" --inference_cache "${INFERENCE}" \
        --reference_cache "${REFERENCE}" --split test --samples "${samples}" \
        --output_dir "${evaluation}"
    fi
  done
done
python scripts/summarize_global_coupled_4d_evaluations.py --root "${SWEEP}" --output_dir "${SWEEP}"
BEST_STEP="$(python -c 'import json;print(json.load(open("diagnostics/global_coupled_4d/checkpoint_sweep/best_checkpoint.json"))["checkpoint_step"])')"
BEST_ALPHA="$(python -c 'import json;print(json.load(open("diagnostics/global_coupled_4d/checkpoint_sweep/best_checkpoint.json"))["alpha"])')"
FINAL_EVAL="diagnostics/global_coupled_4d/final_evaluation"
for candidate_step in "${BEST_STEP}" "${MAX_STEPS}"; do
  checkpoint="${RUN_DIR}/checkpoints/step${candidate_step}.ckpt"
  [[ -s "${checkpoint}" ]] || continue
  group="${FINAL_EVAL}/step${candidate_step}_alpha${BEST_ALPHA/./}"
  mkdir -p "${group}"
  if [[ ! -s "${group}/samples.pt" ]]; then
    python scripts/sample_global_coupled_4d_flow.py \
      --checkpoint "${checkpoint}" --config "${RUN_DIR}/config.resolved.yaml" \
      --cache_dir "${INFERENCE}" --manifest "${MANIFEST}" --split test \
      --output "${group}/samples.pt" --update_scale "${BEST_ALPHA}"
  fi
  if [[ ! -s "${group}/eval/summary.csv" ]]; then
    python scripts/eval_global_coupled_4d_flow.py \
      --manifest "${MANIFEST}" --inference_cache "${INFERENCE}" \
      --reference_cache "${REFERENCE}" --split test --samples "${group}/samples.pt" \
      --output_dir "${group}/eval"
  fi
done
python scripts/summarize_global_coupled_4d_evaluations.py --root "${FINAL_EVAL}" --output_dir "${FINAL_EVAL}"
bash scripts/run_global_coupled_4d_ablation_all.sh
python scripts/report_global_coupled_4d_comparison.py
rm -f "${LOG_ROOT}/FORMAL_RUNNING" "${LOG_ROOT}/FAILED"
touch "${LOG_ROOT}/FORMAL_COMPLETED"
echo "GLOBAL COUPLED 4D FORMAL MATCHED COMPLETED"
