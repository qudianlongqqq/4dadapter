#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export DATA_DIR="${DATA_DIR:-/home/aidd5090/Experiment/qdl/data/etflow_geom}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

CONFIG="configs/drugs-so3-jacobian-4d-bs4.yaml"
OUTPUT_DIR="logs_formal/jacobian_4d_multiseed_100k_$(date +%Y%m%d_%H%M%S)"
MAX_STEPS=100000
BATCH_SIZE=4
ACCUMULATE=2
VAL_CHECK_INTERVAL=2500
LIMIT_VAL_BATCHES=10
LOG_EVERY_N_STEPS=10
TIME_LIMIT_HOURS=48

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
    --time_limit_hours) TIME_LIMIT_HOURS="$2"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/run_jacobian_4d_formal_multiseed_100k.sh [options]
  --config PATH --output_dir DIR --max_steps N --batch_size N
  --accumulate_grad_batches N --val_check_interval N
  --limit_val_batches N --log_every_n_steps N --time_limit_hours HOURS

This runner always trains seeds 43 and 44. It never runs seed 42.
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

seeds=(43 44)
names=(base scale001_q0001)
enabled=(false true)
scales=(0.0 0.01)
q_weights=(0.0 0.001)
corr_reg_weights=(0.0 0.0001)

mkdir -p "${OUTPUT_DIR}"
for seed in "${seeds[@]}"; do
  for name in "${names[@]}"; do
    experiment_dir="${OUTPUT_DIR}/seed${seed}/${name}"
    mkdir -p "${experiment_dir}/checkpoints"
    printf '%s\n' "not_started" > "${experiment_dir}/.run_status"
  done
done

exec > >(tee -a "${OUTPUT_DIR}/master.log") 2>&1

START_TIME="$(date +%s)"
DEADLINE=$((START_TIME + TIME_LIMIT_SECONDS))
git rev-parse HEAD > "${OUTPUT_DIR}/git_commit.txt"
git status --short > "${OUTPUT_DIR}/git_status.txt"
cat > "${OUTPUT_DIR}/run_manifest.txt" <<EOF
config=${CONFIG}
seeds=43,44
experiments=base,scale001_q0001
max_steps=${MAX_STEPS}
batch_size=${BATCH_SIZE}
accumulate_grad_batches=${ACCUMULATE}
val_check_interval=${VAL_CHECK_INTERVAL}
limit_val_batches=${LIMIT_VAL_BATCHES}
log_every_n_steps=${LOG_EVERY_N_STEPS}
time_limit_hours=${TIME_LIMIT_HOURS}
checkpoint_monitor=val/flow_matching_loss
git_commit=$(cat "${OUTPUT_DIR}/git_commit.txt")
EOF

status=0
stop_for_deadline=false
for seed in "${seeds[@]}"; do
  for index in "${!names[@]}"; do
    name="${names[$index]}"
    experiment_dir="${OUTPUT_DIR}/seed${seed}/${name}"
    now="$(date +%s)"
    remaining=$((DEADLINE - now))
    if [[ "${remaining}" -le 0 ]]; then
      echo "Global time limit reached; seed${seed}/${name} was not started"
      status=1
      stop_for_deadline=true
      break
    fi

    printf '%s\n' "running" > "${experiment_dir}/.run_status"
    echo
    echo "================================================================================"
    echo "seed: ${seed}"
    echo "experiment: ${name}"
    echo "remaining global time limit seconds: ${remaining}"
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
        --seed "${seed}" \
        --use_jacobian_4d_correction "${enabled[$index]}" \
        --jacobian_4d_correction_scale "${scales[$index]}" \
        --jacobian_4d_q_loss_weight "${q_weights[$index]}" \
        --jacobian_4d_corr_reg_weight "${corr_reg_weights[$index]}" \
        2>&1 | tee "${experiment_dir}/run.log"
    return_code=${PIPESTATUS[0]}
    set -e

    metrics_found=true
    metrics_path="$(find "${experiment_dir}/csv_logs" -name metrics.csv -type f \
      2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "${metrics_path}" ]]; then
      cp "${metrics_path}" "${experiment_dir}/metrics.csv"
    else
      echo "metrics.csv was not found for seed${seed}/${name}"
      metrics_found=false
      status=1
    fi

    if [[ "${return_code}" -eq 0 && "${metrics_found}" == true ]]; then
      printf '%s\n' "completed" > "${experiment_dir}/.run_status"
    elif [[ "${return_code}" -eq 124 ]]; then
      printf '%s\n' "timed_out" > "${experiment_dir}/.run_status"
      echo "Global ${TIME_LIMIT_HOURS}-hour limit reached during seed${seed}/${name}"
      status=1
      stop_for_deadline=true
      break
    elif [[ "${return_code}" -ne 0 ]]; then
      printf '%s\n' "failed" > "${experiment_dir}/.run_status"
      echo "Experiment failed with return code ${return_code}: seed${seed}/${name}"
      status=1
    else
      printf '%s\n' "failed" > "${experiment_dir}/.run_status"
      echo "Experiment finished without metrics.csv: seed${seed}/${name}"
      status=1
    fi
  done
  if [[ "${stop_for_deadline}" == true ]]; then
    break
  fi
done

if ! python scripts/summarize_jacobian_4d_multiseed.py \
  --base_output_dir "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --seeds 43,44 \
  --experiments base,scale001_q0001 \
  --title "Jacobian 4D formal multiseed 100k summary"; then
  status=1
fi

if [[ -f "${OUTPUT_DIR}/summary.csv" && -f "${OUTPUT_DIR}/summary.md" ]]; then
  echo "Formal multiseed summaries:"
  echo "- ${OUTPUT_DIR}/summary.csv"
  echo "- ${OUTPUT_DIR}/summary.md"
else
  echo "Formal multiseed summary files are missing"
  status=1
fi

if [[ "${status}" -eq 0 ]]; then
  echo "JACOBIAN 4D FORMAL MULTISEED 100K PASSED"
else
  echo "JACOBIAN 4D FORMAL MULTISEED 100K INCOMPLETE OR FAILED"
fi
echo "output_dir: ${OUTPUT_DIR}"
exit "${status}"
