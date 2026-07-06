#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/prepare_flexbond_formal_medium.sh CONFIG CHECKPOINT PROCESSED_DATA_DIR"
  exit 2
fi

CONFIG="$1"
CHECKPOINT="$2"
PROCESSED_DATA_DIR="$3"
TRAIN_MOLECULES="${TRAIN_MOLS:-${TRAIN_MOLECULES:-500}}"
VAL_MOLECULES="${VAL_MOLS:-${VAL_MOLECULES:-100}}"
TEST_MOLECULES="${TEST_MOLS:-${TEST_MOLECULES:-200}}"
DEVICE="${DEVICE:-cuda}"
SAMPLE_SEED="${SAMPLE_SEED:-12}"
UPSTREAM_ROOT="data/upstream_formal_medium"
CACHE_ROOT="data/flexbond_cache_formal_medium"
INFERENCE_ROOT="data/flexbond_inference_formal_medium"
MANIFEST="eval_manifest_formal_medium.json"

for path in "${CONFIG}" "${CHECKPOINT}"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done
for split in train val test; do
  [[ -d "${PROCESSED_DATA_DIR}/drugs/${split}" ]] || {
    echo "Missing processed split: ${PROCESSED_DATA_DIR}/drugs/${split}" >&2
    exit 1
  }
done

for specification in \
  "train:${TRAIN_MOLECULES}" \
  "val:${VAL_MOLECULES}" \
  "test:${TEST_MOLECULES}"; do
  split="${specification%%:*}"
  count="${specification##*:}"
  output_dir="${UPSTREAM_ROOT}/${split}"
  split_marker="${UPSTREAM_ROOT}/.${split}.done"
  if [[ -f "${split_marker}" ]]; then
    echo "[skip] formal-medium ${split} already prepared"
    continue
  fi
  python scripts/eval_jacobian_4d_subset.py \
    --config "${CONFIG}" --checkpoint "${CHECKPOINT}" \
    --output_dir "${output_dir}" --split "${split}" \
    --data_dir "${PROCESSED_DATA_DIR}" --num_molecules "${count}" \
    --start_idx 0 --seed "${SAMPLE_SEED}" --device "${DEVICE}" \
    --allow_non_jacobian
  python scripts/build_flexbond_init_cache.py \
    --init_path "${output_dir}/generated_files.pkl" \
    --output_dir "${CACHE_ROOT}" --split "${split}" \
    --generator_name "ETFlow-drugs-o3" \
    --generator_checkpoint "${CHECKPOINT}" \
    --sample_seed "${SAMPLE_SEED}" --data_dir "${PROCESSED_DATA_DIR}"
  python scripts/check_flexbond_graph_consistency.py \
    --cache_dir "${CACHE_ROOT}" --split "${split}"
  mkdir -p "${UPSTREAM_ROOT}"
  touch "${split_marker}"
done

python scripts/export_flexbond_inference_cache.py \
  --cache_dir "${CACHE_ROOT}" --split test --output_dir "${INFERENCE_ROOT}"
python scripts/check_flexbond_inference_no_labels.py \
  --cache_dir "${INFERENCE_ROOT}" --split test
python scripts/build_flexbond_eval_manifest.py \
  --cache_dir "${INFERENCE_ROOT}" --split test \
  --max_molecules "${TEST_MOLECULES}" --output "${MANIFEST}"
python scripts/summarize_flexbond_cache.py \
  --cache_dir "${CACHE_ROOT}" --output data/formal_medium_cache_summary.json

echo "Formal-medium preparation complete: ${MANIFEST}"
