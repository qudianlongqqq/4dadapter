#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export DATA_DIR=/home/aidd5090/Experiment/qdl/data/etflow_geom
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="configs/drugs-so3-jacobian-4d-bs4.yaml"
OUTPUT_DIR="logs_midtrain/jacobian_4d_midtrain_seed42_$(date +%Y%m%d_%H%M%S)"
MAX_STEPS=25000
BATCH_SIZE=4
ACCUMULATE=2
VAL_CHECK_INTERVAL=1000
LIMIT_VAL_BATCHES=10
LOG_EVERY_N_STEPS=10
SEED=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --accumulate_grad_batches) ACCUMULATE="$2"; shift 2 ;;
    --val_check_interval) VAL_CHECK_INTERVAL="$2"; shift 2 ;;
    --limit_val_batches) LIMIT_VAL_BATCHES="$2"; shift 2 ;;
    --log_every_n_steps) LOG_EVERY_N_STEPS="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/run_jacobian_4d_midtrain_seed42.sh [options]
  --config PATH --output_dir DIR --max_steps N --batch_size N
  --accumulate_grad_batches N --val_check_interval N
  --limit_val_batches N --log_every_n_steps N
EOF
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/master.log") 2>&1

git rev-parse HEAD > "${OUTPUT_DIR}/git_commit.txt"
git status --short > "${OUTPUT_DIR}/git_status.txt"
cat > "${OUTPUT_DIR}/run_manifest.txt" <<EOF
config=${CONFIG}
max_steps=${MAX_STEPS}
batch_size=${BATCH_SIZE}
accumulate_grad_batches=${ACCUMULATE}
val_check_interval=${VAL_CHECK_INTERVAL}
limit_val_batches=${LIMIT_VAL_BATCHES}
log_every_n_steps=${LOG_EVERY_N_STEPS}
seed=${SEED}
checkpoint_monitor=val/flow_matching_loss
git_commit=$(cat "${OUTPUT_DIR}/git_commit.txt")
EOF

names=(base scale001_q0001 scale003_q0003)
enabled=(false true true)
scales=(0.0 0.01 0.03)
q_weights=(0.0 0.001 0.003)
status=0

for index in "${!names[@]}"; do
  name="${names[$index]}"
  experiment_dir="${OUTPUT_DIR}/${name}"
  mkdir -p "${experiment_dir}"
  echo
  echo "================================================================================"
  echo "experiment: ${name}"
  echo "use_jacobian_4d_correction: ${enabled[$index]}"
  echo "correction_scale: ${scales[$index]}"
  echo "q_loss_weight: ${q_weights[$index]}"

  if ! python scripts/train_jacobian_4d.py \
    --config "${CONFIG}" \
    --output_dir "${experiment_dir}" \
    --max_steps "${MAX_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --accumulate_grad_batches "${ACCUMULATE}" \
    --val_check_interval "${VAL_CHECK_INTERVAL}" \
    --limit_val_batches "${LIMIT_VAL_BATCHES}" \
    --log_every_n_steps "${LOG_EVERY_N_STEPS}" \
    --seed "${SEED}" \
    --use_jacobian_4d_correction "${enabled[$index]}" \
    --jacobian_4d_correction_scale "${scales[$index]}" \
    --jacobian_4d_q_loss_weight "${q_weights[$index]}" \
    --jacobian_4d_corr_reg_weight 0.0001 \
    2>&1 | tee "${experiment_dir}/run.log"; then
    echo "experiment failed: ${name}"
    status=1
  fi
done

if ! python scripts/summarize_jacobian_4d_midtrain.py \
  --base_output_dir "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}"; then
  status=1
fi

if [[ "${status}" -eq 0 ]]; then
  echo "JACOBIAN 4D MIDTRAIN SEED42 PASSED"
else
  echo "JACOBIAN 4D MIDTRAIN SEED42 FAILED"
fi
echo "output_dir: ${OUTPUT_DIR}"
exit "${status}"
