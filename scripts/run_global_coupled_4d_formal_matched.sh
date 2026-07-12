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
DEVICE="${GLOBAL4D_DEVICE:-auto}"
CPU_THREADS="${GLOBAL4D_CPU_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CPU_THREADS}}"
mkdir -p "${RUN_DIR}" "${SWEEP}" "${LOG_ROOT}"
if [[ -e "${LOG_ROOT}/SMALL_SWEEP_STOPPED_AFTER_FIRST_RESULT" ]]; then
  echo "Legacy 5k sweep was intentionally stopped after its first valid result"
  exit 0
fi
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

[[ -e "${LOG_ROOT}/SMOKE_EVAL_COMPLETED" ]] || { echo "Smoke evaluation has not completed"; exit 2; }
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
python -c 'import torch,sys; p=torch.load(sys.argv[1],map_location="cpu",weights_only=False); assert int(p.get("global_step",0))>=5000' "${FINAL_CHECKPOINT}"
rm -f "${LOG_ROOT}/FORMAL_RUNNING"
touch "${LOG_ROOT}/FORMAL_COMPLETED"

checkpoints=(step1000 step2000 step3000 step4000 step5000 last)
IDENTITIES="${SWEEP}/checkpoint_identities.json"
python scripts/hash_global_coupled_4d_checkpoints.py \
  --checkpoint_dir "${RUN_DIR}/checkpoints" --names "${checkpoints[@]}" \
  --output "${IDENTITIES}"
rm -f "${LOG_ROOT}/CHECKPOINT_SWEEP_COMPLETED"
touch "${LOG_ROOT}/CHECKPOINT_SWEEP_RUNNING"
combination=0
for checkpoint_name in "${checkpoints[@]}"; do
  checkpoint="${RUN_DIR}/checkpoints/${checkpoint_name}.ckpt"
  [[ -s "${checkpoint}" ]] || { echo "Required checkpoint missing: ${checkpoint}"; exit 2; }
  canonical="$(python -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["canonical_checkpoint"][sys.argv[2]])' "${IDENTITIES}" "${checkpoint_name}")"
  for alpha_code in 02 05; do
    combination=$((combination + 1))
    alpha="0.${alpha_code#0}"
    group="${SWEEP}/${checkpoint_name}_alpha${alpha_code}"
    samples="${group}/samples.pt"; evaluation="${group}/eval"
    mkdir -p "${group}"
    python -c 'import json,sys,datetime,pathlib; pathlib.Path(sys.argv[1]).write_text(json.dumps({"updated_at":datetime.datetime.now().astimezone().isoformat(),"combination_index":int(sys.argv[2]),"combination_total":12,"checkpoint":sys.argv[3],"alpha":float(sys.argv[4]),"group":sys.argv[5]},indent=2),encoding="utf-8")' \
      "${SWEEP}/checkpoint_sweep_state.json" "${combination}" "${checkpoint_name}" "${alpha}" "${group}"
    if [[ "${canonical}" != "${checkpoint_name}" ]]; then
      source_group="${SWEEP}/${canonical}_alpha${alpha_code}"
      [[ -s "${source_group}/samples.pt" && -s "${source_group}/eval/summary.csv" ]] || {
        echo "Canonical checkpoint result is incomplete: ${source_group}"; exit 2;
      }
      if [[ ! -s "${evaluation}/summary.csv" ]]; then
        mkdir -p "${evaluation}"
        cp -f "${source_group}/eval/"*.csv "${evaluation}/"
        cp -f "${source_group}/eval/"*.json "${evaluation}/" 2>/dev/null || true
        cp -f "${source_group}/eval/"*.md "${evaluation}/" 2>/dev/null || true
      fi
      python -c 'import json,sys,pathlib; pathlib.Path(sys.argv[1]).write_text(json.dumps({"reused_from":sys.argv[2],"checkpoint":sys.argv[3],"alpha":float(sys.argv[4]),"reason":"identical inference state_dict, hyperparameters, and global_step"},indent=2),encoding="utf-8")' \
        "${group}/reused_from.json" "${canonical}" "${checkpoint_name}" "${alpha}"
      echo "REUSED ${checkpoint_name} alpha=${alpha} from ${canonical}"
      continue
    fi
    if [[ ! -s "${samples}" ]]; then
      STAGE="CHECKPOINT_SWEEP"; printf '%s\n' "CHECKPOINT_SWEEP" > "${LOG_ROOT}/CURRENT_STAGE"
      touch "${LOG_ROOT}/CHECKPOINT_SWEEP_PARTIAL"
      profile_args=()
      if [[ ! -e "${LOG_ROOT}/SAMPLING_PROFILE_COMPLETED" ]]; then
        profile_args=(--profile --profile_molecules 5)
      fi
      python scripts/sample_global_coupled_4d_flow.py \
        --checkpoint "${checkpoint}" --config "${RUN_DIR}/config.resolved.yaml" \
        --cache_dir "${INFERENCE}" --manifest "${MANIFEST}" --split test \
        --output "${samples}" --max_molecules 100 --update_scale "${alpha}" \
        --device "${DEVICE}" --cpu_threads "${CPU_THREADS}" "${profile_args[@]}" &
      echo $! > "${LOG_ROOT}/SAMPLE.pid"; wait "$(cat "${LOG_ROOT}/SAMPLE.pid")"; rm -f "${LOG_ROOT}/SAMPLE.pid"
      if [[ "${#profile_args[@]}" -gt 0 ]]; then
        touch "${LOG_ROOT}/SAMPLING_PROFILE_COMPLETED"
      fi
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
rm -f "${LOG_ROOT}/CHECKPOINT_SWEEP_RUNNING" "${LOG_ROOT}/CHECKPOINT_SWEEP_PARTIAL"
touch "${LOG_ROOT}/CHECKPOINT_SWEEP_COMPLETED"
python scripts/summarize_global_coupled_4d_evaluations.py \
  --root "${SWEEP}" --output_dir "${SWEEP}" --checkpoint_dir "${RUN_DIR}/checkpoints"

if [[ -e "diagnostics/global_coupled_4d/ablation_5k/COMPLETED" ]]; then
  touch "${LOG_ROOT}/ABLATION_COMPLETED"
fi
if [[ ! -e "${LOG_ROOT}/ABLATION_COMPLETED" ]]; then
  STAGE="ABLATION"; printf '%s\n' "ABLATION" > "${LOG_ROOT}/CURRENT_STAGE"
  touch "${LOG_ROOT}/ABLATION_RUNNING"
  bash scripts/run_global_coupled_4d_ablation_all.sh
  rm -f "${LOG_ROOT}/ABLATION_RUNNING"
  touch "${LOG_ROOT}/ABLATION_COMPLETED"
fi
python scripts/report_global_coupled_4d_5k_comparison.py
rm -f "${LOG_ROOT}/FAILED"
echo "GLOBAL COUPLED 4D FORMAL 5K COMPLETED"
