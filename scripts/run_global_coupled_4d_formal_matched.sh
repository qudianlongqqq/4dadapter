#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
LOG_ROOT="logs_global_coupled_4d"
RUN_DIR="${LOG_ROOT}/global4d_local025_seed42_5000step"
SWEEP="diagnostics/global_coupled_4d/checkpoint_sweep_5k"
CONFIG="configs/global_coupled_4d_local025_matched.yaml"
MANIFEST="${GLOBAL4D_MANIFEST:-eval_manifest_formal_small.json}"
INFERENCE="${GLOBAL4D_INFERENCE_CACHE:-data/flexbond_inference_formal_small}"
REFERENCE="${GLOBAL4D_REFERENCE_CACHE:-data/flexbond_cache_formal_small}"
mkdir -p "${RUN_DIR}" "${SWEEP}" "${LOG_ROOT}"
STAGE="FORMAL_TRAIN"

fail() {
  code=$?; command_text="${BASH_COMMAND}"
  tail_text="$(tail -100 "${LOG_ROOT}/global_coupled_4d_master.log" 2>/dev/null || tail -100 "${RUN_DIR}/formal.log" 2>/dev/null || true)"
  python -c 'import json,sys,datetime,pathlib; pathlib.Path(sys.argv[1]).write_text(json.dumps({"stage":sys.argv[2],"time":datetime.datetime.now().astimezone().isoformat(),"command":sys.argv[3],"exit_code":int(sys.argv[4]),"log":sys.argv[5],"tail":sys.argv[6]},indent=2),encoding="utf-8")' \
    "${LOG_ROOT}/FAILED" "${STAGE}" "${command_text}" "${code}" "${RUN_DIR}/formal.log" "${tail_text}"
  exit "${code}"
}
trap fail ERR
exec > >(tee -a "${RUN_DIR}/formal.log") 2>&1

[[ -e "${LOG_ROOT}/SMOKE_COMPLETED" ]] || { echo "Smoke has not completed"; exit 2; }
python scripts/validate_global_coupled_4d_budget.py --config "${CONFIG}"
touch "${LOG_ROOT}/FORMAL_RUNNING"
printf '%s\n' "FORMAL_TRAIN" > "${LOG_ROOT}/CURRENT_STAGE"

FINAL_CHECKPOINT="${RUN_DIR}/checkpoints/step5000.ckpt"
if [[ ! -s "${FINAL_CHECKPOINT}" ]]; then
  python scripts/train_global_coupled_4d_flow.py \
    --config "${CONFIG}" --cache_dir "${REFERENCE}" --output_dir "${RUN_DIR}" \
    --mode formal --max_steps 5000 --checkpoint_steps 1000,2000,3000,4000,5000 \
    --resume_from_checkpoint auto &
  echo $! > "${LOG_ROOT}/TRAIN.pid"
  wait "$(cat "${LOG_ROOT}/TRAIN.pid")"
  rm -f "${LOG_ROOT}/TRAIN.pid"
fi

checkpoints=(step1000 step2000 step3000 step4000 step5000 last)
for checkpoint_name in "${checkpoints[@]}"; do
  checkpoint="${RUN_DIR}/checkpoints/${checkpoint_name}.ckpt"
  [[ -s "${checkpoint}" ]] || { echo "Required checkpoint missing: ${checkpoint}"; exit 2; }
  for alpha_code in 02 05; do
    alpha="0.${alpha_code#0}"
    group="${SWEEP}/${checkpoint_name}_alpha${alpha_code}"
    samples="${group}/samples.pt"; evaluation="${group}/eval"
    mkdir -p "${group}"
    if [[ ! -s "${samples}" ]]; then
      STAGE="SAMPLE"; printf '%s\n' "SAMPLE" > "${LOG_ROOT}/CURRENT_STAGE"
      python scripts/sample_global_coupled_4d_flow.py \
        --checkpoint "${checkpoint}" --config "${RUN_DIR}/config.resolved.yaml" \
        --cache_dir "${INFERENCE}" --manifest "${MANIFEST}" --split test \
        --output "${samples}" --max_molecules 100 --update_scale "${alpha}" &
      echo $! > "${LOG_ROOT}/SAMPLE.pid"; wait "$(cat "${LOG_ROOT}/SAMPLE.pid")"; rm -f "${LOG_ROOT}/SAMPLE.pid"
    fi
    if [[ ! -s "${evaluation}/summary.csv" ]]; then
      STAGE="EVAL"; printf '%s\n' "EVAL" > "${LOG_ROOT}/CURRENT_STAGE"
      python scripts/eval_global_coupled_4d_flow.py \
        --manifest "${MANIFEST}" --inference_cache "${INFERENCE}" \
        --reference_cache "${REFERENCE}" --split test --samples "${samples}" \
        --output_dir "${evaluation}" --threshold 1.25 &
      echo $! > "${LOG_ROOT}/EVAL.pid"; wait "$(cat "${LOG_ROOT}/EVAL.pid")"; rm -f "${LOG_ROOT}/EVAL.pid"
    fi
  done
done
python scripts/summarize_global_coupled_4d_evaluations.py \
  --root "${SWEEP}" --output_dir "${SWEEP}" --checkpoint_dir "${RUN_DIR}/checkpoints"

STAGE="ABLATION"; printf '%s\n' "ABLATION" > "${LOG_ROOT}/CURRENT_STAGE"
bash scripts/run_global_coupled_4d_ablation_all.sh
python scripts/report_global_coupled_4d_5k_comparison.py
rm -f "${LOG_ROOT}/FORMAL_RUNNING" "${LOG_ROOT}/FAILED"
touch "${LOG_ROOT}/FORMAL_COMPLETED"
echo "GLOBAL COUPLED 4D FORMAL 5K COMPLETED"
