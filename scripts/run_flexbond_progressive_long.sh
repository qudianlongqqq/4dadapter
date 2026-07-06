#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-logs_long/${STAMP}}"
DEVICE="${DEVICE:-cuda}"
BASE_CONFIG="${BASE_CONFIG:-configs/flexbond_optimizer_egnn.yaml}"
SMALL_CACHE="${SMALL_CACHE:-data/flexbond_cache_formal_small}"
SMALL_INFERENCE="${SMALL_INFERENCE:-data/flexbond_inference_formal_small}"
SMALL_MANIFEST="${SMALL_MANIFEST:-eval_manifest_formal_small.json}"
MEDIUM_CACHE="${MEDIUM_CACHE:-data/flexbond_cache_formal_medium}"
MEDIUM_INFERENCE="${MEDIUM_INFERENCE:-data/flexbond_inference_formal_medium}"
MEDIUM_MANIFEST="${MEDIUM_MANIFEST:-eval_manifest_formal_medium.json}"
TRAIN_MOLS="${TRAIN_MOLS:-500}"
VAL_MOLS="${VAL_MOLS:-100}"
TEST_MOLS="${TEST_MOLS:-200}"
SMALL_STEPS="${SMALL_STEPS:-2000}"
MEDIUM_STEPS="${MEDIUM_STEPS:-3000}"
ALPHAS="${ALPHAS:-0.1 0.2 0.5 1.0}"
MAX_DISPLACEMENT="${MAX_DISPLACEMENT:-}"
SKIP_MEDIUM="${SKIP_MEDIUM:-0}"
STATE_KEY="small${SMALL_STEPS}_medium${MEDIUM_STEPS}_mols${TRAIN_MOLS}-${VAL_MOLS}-${TEST_MOLS}"
STATE_KEY+="_alpha${ALPHAS// /-}_clip${MAX_DISPLACEMENT:-none}"
STATE_ROOT="${STATE_ROOT:-logs_long/progressive_state/${STATE_KEY}}"
mkdir -p "${LOG_ROOT}" "${STATE_ROOT}"
exec > >(tee -a "${LOG_ROOT}/progressive.log") 2>&1

for integer in "${TRAIN_MOLS}" "${VAL_MOLS}" "${TEST_MOLS}" \
  "${SMALL_STEPS}" "${MEDIUM_STEPS}"; do
  [[ "${integer}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Expected a positive integer, got: ${integer}" >&2
    exit 2
  }
done
read -r -a ALPHA_VALUES <<<"${ALPHAS}"
[[ "${#ALPHA_VALUES[@]}" -gt 0 ]] || { echo "ALPHAS cannot be empty" >&2; exit 2; }

step_tag() {
  local steps="$1"
  if (( steps % 1000 == 0 )); then
    echo "$((steps / 1000))k"
  else
    echo "${steps}step"
  fi
}

SMALL_TAG="$(step_tag "${SMALL_STEPS}")"
MEDIUM_TAG="$(step_tag "${MEDIUM_STEPS}")"
SMALL_CART025="cartesian_local025_${SMALL_TAG}"
SMALL_FLEX025="flexbond4d_local025_${SMALL_TAG}"
SMALL_CART050="cartesian_local050_${SMALL_TAG}"
SMALL_FLEX050="flexbond4d_local050_${SMALL_TAG}"
MEDIUM_CART025="cartesian_local025_${MEDIUM_TAG}"
MEDIUM_FLEX025="flexbond4d_local025_${MEDIUM_TAG}"

trap 'echo "Interrupted safely. Completed .done markers are under ${STATE_ROOT}."; exit 130' INT TERM

run_stage() {
  local stage="$1"
  shift
  local marker="${STATE_ROOT}/${stage}.done"
  local log="${LOG_ROOT}/${stage}.log"
  if [[ -f "${marker}" ]]; then
    echo "[skip] ${stage} already completed"
    return 0
  fi
  echo "[start] ${stage}"
  set +e
  "$@" 2>&1 | tee "${log}"
  local status="${PIPESTATUS[0]}"
  set -e
  if [[ "${status}" -eq 75 ]]; then
    echo "[deferred] ${stage}; prerequisites are not available" | tee -a "${log}"
    return 0
  fi
  if [[ "${status}" -ne 0 ]]; then
    echo "[failed] ${stage} status=${status}" | tee -a "${log}"
    return "${status}"
  fi
  touch "${marker}"
  echo "[done] ${stage}" | tee -a "${log}"
}

require_files_or_defer() {
  local missing=0
  for path in "$@"; do
    if [[ -z "${path}" || ! -f "${path}" ]]; then
      echo "Missing prerequisite file: ${path:-<unset>}"
      missing=1
    fi
  done
  [[ "${missing}" -eq 0 ]] || return 75
}

medium_is_disabled() {
  case "${SKIP_MEDIUM,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

train_variant() {
  local name="$1" mode="$2" t_max="$3" steps="$4" cache="$5" root="$6"
  local output="${root}/${name}"
  if [[ -f "${output}/.done" ]]; then
    echo "[skip run] ${name}"
    return 0
  fi
  mkdir -p "${output}"
  local command=(python scripts/train_flexbond_optimizer.py
    --config "${BASE_CONFIG}" --mode "${mode}" --cache_dir "${cache}"
    --output_dir "${output}" --max_steps "${steps}" --t_min 0 --t_max "${t_max}")
  if [[ -f "${output}/checkpoints/last.ckpt" ]]; then
    command+=(--resume_from_checkpoint "${output}/checkpoints/last.ckpt")
  fi
  "${command[@]}"
  touch "${output}/.done"
}

stage0() {
  local configs=(logs_flexbond_formal_small/*/config.resolved.yaml)
  [[ -e "${configs[0]}" ]] || { echo "No existing formal-small runs."; return 75; }
  local runs=()
  for config in "${configs[@]}"; do runs+=("$(dirname "${config}")"); done
  python scripts/summarize_flexbond_runs.py \
    --run_dirs "${runs[@]}" --output_dir diagnostics/formal_small_run_summary
}

stage1() {
  [[ -d "${SMALL_CACHE}/train" ]] || { echo "Missing ${SMALL_CACHE}/train"; return 75; }
  python scripts/diagnose_flexbond_path_geometry.py \
    --cache_dir "${SMALL_CACHE}" --split train --num_records 500 \
    --t_values 0 0.25 0.5 0.75 1 \
    --output_dir diagnostics/path_geometry_formal_small
}

stage2() {
  local c1="${CARTESIAN_1K_CKPT:-}" c5="${CARTESIAN_5K_CKPT:-}"
  local f1="${FLEXBOND_1K_CKPT:-}" f5="${FLEXBOND_5K_CKPT:-}"
  require_files_or_defer "${c1}" "${c5}" "${f1}" "${f5}" || return $?
  require_files_or_defer "${SMALL_MANIFEST}" || return $?
  local clipping_args=()
  [[ -n "${MAX_DISPLACEMENT}" ]] && \
    clipping_args=(--max_displacement "${MAX_DISPLACEMENT}")
  python scripts/sweep_flexbond_update_scale.py \
    --manifest "${SMALL_MANIFEST}" --inference_cache "${SMALL_INFERENCE}" \
    --reference_cache "${SMALL_CACHE}" --split test \
    --cartesian_checkpoints "${c1}" "${c5}" \
    --flexbond_checkpoints "${f1}" "${f5}" \
    --cartesian_config "$(dirname "$(dirname "${c1}")")/config.resolved.yaml" \
    --flexbond_config "$(dirname "$(dirname "${f1}")")/config.resolved.yaml" \
    --update_scales "${ALPHA_VALUES[@]}" "${clipping_args[@]}" \
    --device "${DEVICE}" --skip_existing \
    --output_dir diagnostics/update_scale_sweep_core
}

stage3() {
  [[ -d "${SMALL_CACHE}/train" ]] || return 75
  local root="logs_flexbond_local_time_small"
  train_variant "${SMALL_CART025}" cartesian_optimizer 0.25 "${SMALL_STEPS}" "${SMALL_CACHE}" "${root}"
  train_variant "${SMALL_FLEX025}" flexbond4d_hybrid_optimizer 0.25 "${SMALL_STEPS}" "${SMALL_CACHE}" "${root}"
  train_variant "${SMALL_CART050}" cartesian_optimizer 0.50 "${SMALL_STEPS}" "${SMALL_CACHE}" "${root}"
  train_variant "${SMALL_FLEX050}" flexbond4d_hybrid_optimizer 0.50 "${SMALL_STEPS}" "${SMALL_CACHE}" "${root}"
}

stage4() {
  local root="logs_flexbond_local_time_small"
  local c25="${root}/${SMALL_CART025}/checkpoints/last.ckpt"
  local c50="${root}/${SMALL_CART050}/checkpoints/last.ckpt"
  local f25="${root}/${SMALL_FLEX025}/checkpoints/last.ckpt"
  local f50="${root}/${SMALL_FLEX050}/checkpoints/last.ckpt"
  require_files_or_defer "${c25}" "${c50}" "${f25}" "${f50}" "${SMALL_MANIFEST}" || return $?
  local clipping_args=()
  [[ -n "${MAX_DISPLACEMENT}" ]] && \
    clipping_args=(--max_displacement "${MAX_DISPLACEMENT}")
  python scripts/sweep_flexbond_update_scale.py \
    --manifest "${SMALL_MANIFEST}" --inference_cache "${SMALL_INFERENCE}" \
    --reference_cache "${SMALL_CACHE}" --split test \
    --cartesian_checkpoints "${c25}" "${c50}" \
    --flexbond_checkpoints "${f25}" "${f50}" \
    --cartesian_config "${root}/${SMALL_CART025}/config.resolved.yaml" \
    --flexbond_config "${root}/${SMALL_FLEX025}/config.resolved.yaml" \
    --update_scales "${ALPHA_VALUES[@]}" "${clipping_args[@]}" \
    --device "${DEVICE}" --skip_existing \
    --output_dir diagnostics/local_time_small_eval
  python scripts/summarize_flexbond_runs.py \
    --run_dirs \
      "${root}/${SMALL_CART025}" "${root}/${SMALL_CART050}" \
      "${root}/${SMALL_FLEX025}" "${root}/${SMALL_FLEX050}" \
    --rollout_summaries diagnostics/local_time_small_eval/sweep_summary.csv \
    --output_dir diagnostics/local_time_small_run_summary
}

stage5() {
  local summary="diagnostics/local_time_small_eval/sweep_summary.csv"
  [[ -f "${summary}" ]] || return 75
  local selection
  selection="$(python - "${summary}" <<'PY'
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1], encoding='utf-8')) if r['subset']=='all']
best=min(rows, key=lambda r:(float(r['rmsd_mean']),float(r['failure_rate'])))
print(best['method'], best['checkpoint_path'])
PY
)"
  local method checkpoint
  read -r method checkpoint <<<"${selection}"
  require_files_or_defer "${checkpoint}" || return $?
  local run_dir="$(dirname "$(dirname "${checkpoint}")")"
  local arguments=()
  if [[ "${method}" == "cartesian_adapter" ]]; then
    arguments=(--cartesian_checkpoints "${checkpoint}" --cartesian_config "${run_dir}/config.resolved.yaml")
  else
    arguments=(--flexbond_checkpoints "${checkpoint}" --flexbond_config "${run_dir}/config.resolved.yaml")
  fi
  python scripts/sweep_flexbond_update_scale.py \
    --manifest "${SMALL_MANIFEST}" --inference_cache "${SMALL_INFERENCE}" \
    --reference_cache "${SMALL_CACHE}" --split test "${arguments[@]}" \
    --update_scales "${ALPHA_VALUES[@]}" \
    --max_displacement "${MAX_DISPLACEMENT:-0.1}" \
    --device "${DEVICE}" --skip_existing \
    --output_dir diagnostics/local_time_clipping_eval
}

stage6() {
  if medium_is_disabled; then
    echo "SKIP_MEDIUM=${SKIP_MEDIUM}; TODO: medium preparation intentionally skipped."
    return 75
  fi
  if [[ -d "${MEDIUM_CACHE}/train" && -f "${MEDIUM_MANIFEST}" ]]; then
    echo "Formal-medium cache already exists."
    return 0
  fi
  local config="${UPSTREAM_CONFIG:-}" checkpoint="${UPSTREAM_CHECKPOINT:-}"
  local data_dir="${PROCESSED_DATA_DIR:-}"
  if [[ -z "${config}" || ! -f "${config}" || \
        -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
    echo "TODO: set valid UPSTREAM_CONFIG and UPSTREAM_CHECKPOINT; skipping medium safely."
    return 75
  fi
  [[ -d "${data_dir}" ]] || {
    echo "TODO: set PROCESSED_DATA_DIR to the processed drugs dataset; skipping medium safely."
    return 75
  }
  TRAIN_MOLS="${TRAIN_MOLS}" VAL_MOLS="${VAL_MOLS}" TEST_MOLS="${TEST_MOLS}" \
  DEVICE="${DEVICE}" \
    bash scripts/prepare_flexbond_formal_medium.sh \
      "${config}" "${checkpoint}" "${data_dir}"
}

stage7() {
  if medium_is_disabled; then
    echo "SKIP_MEDIUM=${SKIP_MEDIUM}; medium training skipped."
    return 75
  fi
  [[ -d "${MEDIUM_CACHE}/train" ]] || return 75
  local root="logs_flexbond_local_time_medium"
  train_variant "${MEDIUM_CART025}" cartesian_optimizer 0.25 "${MEDIUM_STEPS}" "${MEDIUM_CACHE}" "${root}"
  train_variant "${MEDIUM_FLEX025}" flexbond4d_hybrid_optimizer 0.25 "${MEDIUM_STEPS}" "${MEDIUM_CACHE}" "${root}"
}

stage8() {
  if medium_is_disabled; then
    echo "SKIP_MEDIUM=${SKIP_MEDIUM}; medium evaluation skipped."
    return 75
  fi
  local root="logs_flexbond_local_time_medium"
  local cart="${root}/${MEDIUM_CART025}/checkpoints/last.ckpt"
  local flex="${root}/${MEDIUM_FLEX025}/checkpoints/last.ckpt"
  require_files_or_defer "${cart}" "${flex}" "${MEDIUM_MANIFEST}" || return $?
  python scripts/sweep_flexbond_update_scale.py \
    --manifest "${MEDIUM_MANIFEST}" --inference_cache "${MEDIUM_INFERENCE}" \
    --reference_cache "${MEDIUM_CACHE}" --split test \
    --cartesian_checkpoints "${cart}" --flexbond_checkpoints "${flex}" \
    --cartesian_config "${root}/${MEDIUM_CART025}/config.resolved.yaml" \
    --flexbond_config "${root}/${MEDIUM_FLEX025}/config.resolved.yaml" \
    --update_scales "${ALPHA_VALUES[@]}" \
    --max_displacement "${MAX_DISPLACEMENT:-0.1}" \
    --device "${DEVICE}" --skip_existing \
    --output_dir diagnostics/formal_medium_eval
  python scripts/summarize_flexbond_runs.py \
    --run_dirs "${root}/${MEDIUM_CART025}" "${root}/${MEDIUM_FLEX025}" \
    --rollout_summaries diagnostics/formal_medium_eval/sweep_summary.csv \
    --output_dir diagnostics/formal_medium_run_summary
}

run_stage stage0_summarize_existing stage0
run_stage stage1_path_geometry stage1
run_stage stage2_core_scale_sweep stage2
run_stage stage3_train_local_small stage3
run_stage stage4_eval_local_small stage4
run_stage stage5_clipping_best_small stage5
run_stage stage6_prepare_medium stage6
run_stage stage7_train_medium stage7
run_stage stage8_eval_medium stage8

echo "Progressive experiment finished or deferred safely. Logs: ${LOG_ROOT}"
