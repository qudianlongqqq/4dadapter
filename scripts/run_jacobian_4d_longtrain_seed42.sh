#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export DATA_DIR="${DATA_DIR:-/home/aidd5090/Experiment/qdl/data/etflow_geom}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

CONFIG="configs/drugs-so3-jacobian-4d-bs4.yaml"
OUTPUT_DIR="logs_longtrain/jacobian_4d_seed42_base_vs_scale001_100k_$(date +%Y%m%d_%H%M%S)"
MAX_STEPS=100000
SEED=42
BATCH_SIZE=4
ACCUMULATE=2
VAL_CHECK_INTERVAL=2500
LIMIT_VAL_BATCHES=10
LOG_EVERY_N_STEPS=10
TIME_LIMIT_HOURS=24

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --accumulate_grad_batches) ACCUMULATE="$2"; shift 2 ;;
    --val_check_interval) VAL_CHECK_INTERVAL="$2"; shift 2 ;;
    --limit_val_batches) LIMIT_VAL_BATCHES="$2"; shift 2 ;;
    --log_every_n_steps) LOG_EVERY_N_STEPS="$2"; shift 2 ;;
    --time_limit_hours) TIME_LIMIT_HOURS="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/run_jacobian_4d_longtrain_seed42.sh [options]
  --config PATH --output_dir DIR --max_steps N --seed N --batch_size N
  --accumulate_grad_batches N --val_check_interval N
  --limit_val_batches N --log_every_n_steps N --time_limit_hours HOURS
EOF
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

TIME_LIMIT_SECONDS="$(awk -v hours="${TIME_LIMIT_HOURS}" 'BEGIN { printf "%d", hours * 3600 }')"
if [[ "${TIME_LIMIT_SECONDS}" -le 0 ]]; then
  echo "--time_limit_hours must be positive" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/master.log") 2>&1

START_TIME="$(date +%s)"
DEADLINE=$((START_TIME + TIME_LIMIT_SECONDS))
git rev-parse HEAD > "${OUTPUT_DIR}/git_commit.txt"
git status --short > "${OUTPUT_DIR}/git_status.txt"
cat > "${OUTPUT_DIR}/run_manifest.txt" <<EOF
config=${CONFIG}
max_steps=${MAX_STEPS}
seed=${SEED}
batch_size=${BATCH_SIZE}
accumulate_grad_batches=${ACCUMULATE}
val_check_interval=${VAL_CHECK_INTERVAL}
limit_val_batches=${LIMIT_VAL_BATCHES}
log_every_n_steps=${LOG_EVERY_N_STEPS}
time_limit_hours=${TIME_LIMIT_HOURS}
checkpoint_monitor=val/flow_matching_loss
git_commit=$(cat "${OUTPUT_DIR}/git_commit.txt")
EOF

names=(base scale001_q0001)
enabled=(false true)
scales=(0.0 0.01)
q_weights=(0.0 0.001)
corr_reg_weights=(0.0 0.0001)
status=0

for index in "${!names[@]}"; do
  now="$(date +%s)"
  remaining=$((DEADLINE - now))
  if [[ "${remaining}" -le 0 ]]; then
    echo "Global time limit reached; ${names[$index]} was not started"
    status=1
    break
  fi

  name="${names[$index]}"
  experiment_dir="${OUTPUT_DIR}/${name}"
  mkdir -p "${experiment_dir}"
  echo
  echo "================================================================================"
  echo "experiment: ${name}"
  echo "remaining time limit seconds: ${remaining}"
  echo "use_jacobian_4d_correction: ${enabled[$index]}"
  echo "correction_scale: ${scales[$index]}"
  echo "q_loss_weight: ${q_weights[$index]}"
  echo "corr_reg_weight: ${corr_reg_weights[$index]}"

  set +e
  timeout --signal=TERM --kill-after=60s "${remaining}s" \
    python scripts/train_jacobian_4d.py \
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
      --jacobian_4d_corr_reg_weight "${corr_reg_weights[$index]}" \
      2>&1 | tee "${experiment_dir}/run.log"
  return_code=${PIPESTATUS[0]}
  set -e

  metrics_path="$(find "${experiment_dir}/csv_logs" -name metrics.csv -type f \
    2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${metrics_path}" ]]; then
    cp "${metrics_path}" "${experiment_dir}/metrics.csv"
  else
    echo "metrics.csv was not found for ${name}"
    status=1
  fi

  if [[ "${return_code}" -eq 124 ]]; then
    echo "experiment timed out at the global ${TIME_LIMIT_HOURS}-hour limit: ${name}"
    status=1
    break
  elif [[ "${return_code}" -ne 0 ]]; then
    echo "experiment failed with return code ${return_code}: ${name}"
    status=1
  fi
done

if ! python scripts/summarize_jacobian_4d_midtrain.py \
  --base_output_dir "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --experiments base,scale001_q0001 \
  --title "Jacobian 4D seed42 base vs scale001 100k summary"; then
  status=1
fi

if [[ -f "${OUTPUT_DIR}/summary.csv" && -f "${OUTPUT_DIR}/summary.md" ]]; then
  echo "Longtrain summaries:"
  echo "- ${OUTPUT_DIR}/summary.csv"
  echo "- ${OUTPUT_DIR}/summary.md"
else
  echo "Longtrain summary files are missing"
  status=1
fi

if [[ "${status}" -eq 0 ]]; then
  echo "JACOBIAN 4D LONGTRAIN SEED42 PASSED"
else
  echo "JACOBIAN 4D LONGTRAIN SEED42 INCOMPLETE OR FAILED"
fi
echo "output_dir: ${OUTPUT_DIR}"
exit "${status}"
