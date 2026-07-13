#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ETFLOW_ROOT="${FORMAL_LARGE_ETFLOW_ROOT:-/home/aidd5090/Experiment/qdl/ETFlow}"
CHECKPOINT="${FORMAL_LARGE_ETFLOW_CHECKPOINT:-/home/aidd5090/.cache/etflow/drugs-o3.ckpt}"
CONFIG="${FORMAL_LARGE_ETFLOW_CONFIG:-/home/aidd5090/Experiment/qdl/ETFlow/configs/drugs-o3.yaml}"
PROCESSED_DATA="${FORMAL_LARGE_PROCESSED_DATA:-/home/aidd5090/Experiment/qdl/data/etflow_geom/processed}"
TEST_OUTPUT="${FORMAL_LARGE_ETFLOW_TEST_OUTPUT:-${ROOT_DIR}/data/upstream_formal_small/test/generated_files.pkl}"
OUTPUT_ROOT="${FORMAL_LARGE_ETFLOW_OUTPUT_ROOT:-${ROOT_DIR}/data/upstream_formal_large}"
DEVICE="${DEVICE:-cuda}"
RESUME="${RESUME:-1}"
SAVE_EVERY_MOLECULES="${SAVE_EVERY_MOLECULES:-100}"
SEED="${FORMAL_LARGE_SEED:-42}"

LOG_DIR="${ROOT_DIR}/logs_formal_large"
REPORT_PATH="${ROOT_DIR}/reports/formal_large_upstream_integrity.json"
MASTER_LOG="${LOG_DIR}/upstream_generation_master.log"
TRAIN_LOG="${LOG_DIR}/upstream_train.log"
VAL_LOG="${LOG_DIR}/upstream_val.log"
mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}/train" "${OUTPUT_ROOT}/val" "$(dirname "${REPORT_PATH}")"
exec > >(tee -a "${MASTER_LOG}") 2>&1

case "${RESUME,,}" in
  1|true|yes|on) RESUME_ARGS=(--resume) ;;
  0|false|no|off) RESUME_ARGS=() ;;
  *) echo "RESUME must be one of 1/0, true/false, yes/no, or on/off"; exit 2 ;;
esac

run_split() {
  local split="$1"
  local max_molecules="$2"
  local samples_per_molecule="$3"
  local split_log="$4"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/generate_etflow_formal_large_upstream.py" \
    --etflow_root "${ETFLOW_ROOT}" \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --processed_data "${PROCESSED_DATA}" \
    --split "${split}" \
    --max_molecules "${max_molecules}" \
    --samples_per_molecule "${samples_per_molecule}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_ROOT}/${split}" \
    --device "${DEVICE}" \
    --save_every_molecules "${SAVE_EVERY_MOLECULES}" \
    "${RESUME_ARGS[@]}" |& tee -a "${split_log}"
}

echo "Starting deterministic ETFlow formal-large train generation (50000 x 3)."
run_split train 50000 3 "${TRAIN_LOG}"

echo "Starting deterministic ETFlow formal-large validation generation (5000 x 2)."
run_split val 5000 2 "${VAL_LOG}"

echo "Checking exact counts, hashes, coordinate validity, and train/val/test identity overlap."
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_etflow_formal_large_upstream.py" \
  --train_dir "${OUTPUT_ROOT}/train" \
  --val_dir "${OUTPUT_ROOT}/val" \
  --test_output "${TEST_OUTPUT}" \
  --report "${REPORT_PATH}"

echo "Formal-large upstream generation is ready. Data build and training were not started."
