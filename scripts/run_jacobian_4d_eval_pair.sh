#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export DATA_DIR=/home/aidd5090/Experiment/qdl/data/etflow_geom
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_CONFIG="logs_replicates/jacobian_4d_seq_42_43_44_5000steps_so3_20260701_140501/seed42/base/config.resolved.yaml"
BASE_CHECKPOINT="logs_replicates/jacobian_4d_seq_42_43_44_5000steps_so3_20260701_140501/seed42/base/checkpoints/jacobian-4d-4750.ckpt"
JAC_CONFIG="logs_replicates/jacobian_4d_seq_42_43_44_5000steps_so3_20260701_140501/seed42/scale001_q0001/config.resolved.yaml"
JAC_CHECKPOINT="logs_replicates/jacobian_4d_seq_42_43_44_5000steps_so3_20260701_140501/seed42/scale001_q0001/checkpoints/jacobian-4d-3750.ckpt"
OUTPUT_DIR="logs_eval_pair/seed42_base_vs_scale001_$(date +%Y%m%d_%H%M%S)"
NUM_MOLECULES=20
START_IDX=0
DEVICE=cuda
MODE=subset
DEBUG_SUBSET=0
RUN_COV_MAT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base_config) BASE_CONFIG="$2"; shift 2 ;;
    --base_checkpoint) BASE_CHECKPOINT="$2"; shift 2 ;;
    --jacobian_config) JAC_CONFIG="$2"; shift 2 ;;
    --jacobian_checkpoint) JAC_CHECKPOINT="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --num_molecules) NUM_MOLECULES="$2"; shift 2 ;;
    --start_idx) START_IDX="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --debug_subset) MODE=subset; DEBUG_SUBSET=1; shift ;;
    --full) MODE=full; shift ;;
    --run_cov_mat) RUN_COV_MAT=1; shift ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/run_jacobian_4d_eval_pair.sh [options]
  --base_config PATH --base_checkpoint PATH
  --jacobian_config PATH --jacobian_checkpoint PATH
  --output_dir DIR --num_molecules N --start_idx N --device DEVICE
  --debug_subset    subset mode with extra diagnostics (default mode is subset)
  --full            run scripts/eval.py over the full test split
  --run_cov_mat      invoke scripts/eval_cov_mat.py after generation
EOF
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

BASE_DIR="${OUTPUT_DIR}/base"
JAC_DIR="${OUTPUT_DIR}/scale001_q0001"
mkdir -p "${BASE_DIR}" "${JAC_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/master.log") 2>&1

git rev-parse HEAD > "${OUTPUT_DIR}/git_commit.txt"
git status --short > "${OUTPUT_DIR}/git_status.txt"
cat > "${OUTPUT_DIR}/run_manifest.txt" <<EOF
mode=${MODE}
num_molecules=${NUM_MOLECULES}
start_idx=${START_IDX}
device=${DEVICE}
base_config=${BASE_CONFIG}
base_checkpoint=${BASE_CHECKPOINT}
jacobian_config=${JAC_CONFIG}
jacobian_checkpoint=${JAC_CHECKPOINT}
git_commit=$(cat "${OUTPUT_DIR}/git_commit.txt")
EOF

status=0
base_generated=""
jac_generated=""

if [[ "${MODE}" == "subset" ]]; then
  debug_args=()
  if [[ "${DEBUG_SUBSET}" -eq 1 ]]; then
    debug_args+=(--debug)
  fi
  if ! python scripts/eval_jacobian_4d_subset.py \
    --config "${BASE_CONFIG}" \
    --checkpoint "${BASE_CHECKPOINT}" \
    --output_dir "${BASE_DIR}" \
    --num_molecules "${NUM_MOLECULES}" \
    --start_idx "${START_IDX}" \
    --device "${DEVICE}" \
    --allow_non_jacobian \
    "${debug_args[@]}" 2>&1 | tee "${BASE_DIR}/eval.log"; then
    status=1
  fi
  if ! python scripts/eval_jacobian_4d_subset.py \
    --config "${JAC_CONFIG}" \
    --checkpoint "${JAC_CHECKPOINT}" \
    --output_dir "${JAC_DIR}" \
    --num_molecules "${NUM_MOLECULES}" \
    --start_idx "${START_IDX}" \
    --device "${DEVICE}" \
    "${debug_args[@]}" 2>&1 | tee "${JAC_DIR}/eval.log"; then
    status=1
  fi
  base_generated="${BASE_DIR}/generated_files.pkl"
  jac_generated="${JAC_DIR}/generated_files.pkl"
  if [[ -f "${BASE_DIR}/subset_output.pt" && -f "${JAC_DIR}/subset_output.pt" ]]; then
    if ! python scripts/summarize_jacobian_4d_subset.py \
      --base_output "${BASE_DIR}/subset_output.pt" \
      --jacobian_output "${JAC_DIR}/subset_output.pt" \
      --output_dir "${OUTPUT_DIR}"; then
      status=1
    fi
  else
    echo "Subset diagnostic output missing; summary could not be generated"
    status=1
  fi
else
  if ! python scripts/eval.py \
    --config "${BASE_CONFIG}" \
    --checkpoint "${BASE_CHECKPOINT}" \
    --output_dir "${BASE_DIR}" 2>&1 | tee "${BASE_DIR}/eval.log"; then
    status=1
  fi
  if ! python scripts/eval.py \
    --config "${JAC_CONFIG}" \
    --checkpoint "${JAC_CHECKPOINT}" \
    --output_dir "${JAC_DIR}" 2>&1 | tee "${JAC_DIR}/eval.log"; then
    status=1
  fi
  base_generated="$(find "${BASE_DIR}" -name generated_files.pkl -type f | sort | tail -n 1 || true)"
  jac_generated="$(find "${JAC_DIR}" -name generated_files.pkl -type f | sort | tail -n 1 || true)"
  cat > "${OUTPUT_DIR}/summary.md" <<EOF
# Full base versus 4D evaluation run

- base generated file: ${base_generated:-missing}
- 4D generated file: ${jac_generated:-missing}
- base config: ${BASE_CONFIG}
- base checkpoint: ${BASE_CHECKPOINT}
- 4D config: ${JAC_CONFIG}
- 4D checkpoint: ${JAC_CHECKPOINT}
EOF
fi

cat >> "${OUTPUT_DIR}/summary.md" <<EOF

## Run provenance

- git commit: $(cat "${OUTPUT_DIR}/git_commit.txt")
- git status: see git_status.txt
- base config: ${BASE_CONFIG}
- base checkpoint: ${BASE_CHECKPOINT}
- 4D config: ${JAC_CONFIG}
- 4D checkpoint: ${JAC_CHECKPOINT}

## COV/MAT next commands

    WANDB_MODE=offline python scripts/eval_cov_mat.py --path "${base_generated}" --num_workers 1
    WANDB_MODE=offline python scripts/eval_cov_mat.py --path "${jac_generated}" --num_workers 1
EOF

if [[ "${RUN_COV_MAT}" -eq 1 ]]; then
  if [[ -f "${base_generated}" ]]; then
    if ! python scripts/eval_cov_mat.py --path "${base_generated}" --num_workers 1 \
      2>&1 | tee "${BASE_DIR}/eval_cov_mat.log"; then
      status=1
    fi
  else
    echo "Base generated_files.pkl missing"
    status=1
  fi
  if [[ -f "${jac_generated}" ]]; then
    if ! python scripts/eval_cov_mat.py --path "${jac_generated}" --num_workers 1 \
      2>&1 | tee "${JAC_DIR}/eval_cov_mat.log"; then
      status=1
    fi
  else
    echo "4D generated_files.pkl missing"
    status=1
  fi

  base_metrics="${BASE_DIR}/eval_cov_mat_metrics.csv"
  base_metrics_json="${BASE_DIR}/eval_cov_mat_metrics.json"
  jac_metrics="${JAC_DIR}/eval_cov_mat_metrics.csv"
  jac_metrics_json="${JAC_DIR}/eval_cov_mat_metrics.json"
  if [[ -f "${base_metrics}" && -f "${base_metrics_json}" \
    && -f "${jac_metrics}" && -f "${jac_metrics_json}" ]]; then
    if ! python scripts/summarize_eval_cov_mat_pair.py \
      --base_dir "${BASE_DIR}" \
      --jacobian_dir "${JAC_DIR}" \
      --output_dir "${OUTPUT_DIR}" \
      --model_name scale001_q0001; then
      status=1
    fi
  else
    echo "Coverage metrics files missing; pair summary could not be generated"
    status=1
  fi

  if [[ -f "${OUTPUT_DIR}/cov_mat_pair_summary.csv" \
    && -f "${OUTPUT_DIR}/cov_mat_pair_summary.md" ]]; then
    echo "COV/MAT pair summaries:"
    echo "- ${OUTPUT_DIR}/cov_mat_pair_summary.csv"
    echo "- ${OUTPUT_DIR}/cov_mat_pair_summary.md"
  else
    echo "COV/MAT pair summary files missing"
    status=1
  fi
fi

if [[ "${status}" -eq 0 ]]; then
  echo "JACOBIAN 4D EVAL PAIR PASSED"
else
  echo "JACOBIAN 4D EVAL PAIR FAILED"
fi
echo "output_dir: ${OUTPUT_DIR}"
exit "${status}"
