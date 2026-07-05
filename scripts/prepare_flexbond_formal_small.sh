#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/prepare_flexbond_formal_small.sh CONFIG CHECKPOINT PROCESSED_DATA_DIR"
  exit 2
fi

CONFIG="$1"
CHECKPOINT="$2"
PROCESSED_DATA_DIR="$3"
UPSTREAM_ROOT="data/upstream_formal_small"
CACHE_ROOT="data/flexbond_cache_formal_small"
INFERENCE_ROOT="data/flexbond_inference_formal_small"
MANIFEST="eval_manifest_formal_small.json"
SAMPLE_SEED=12

for path in "${CONFIG}" "${CHECKPOINT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required file does not exist: ${path}" >&2
    exit 1
  fi
done
if [[ ! -d "${PROCESSED_DATA_DIR}/drugs/train" || \
      ! -d "${PROCESSED_DATA_DIR}/drugs/val" || \
      ! -d "${PROCESSED_DATA_DIR}/drugs/test" ]]; then
  echo "Expected processed split directories under ${PROCESSED_DATA_DIR}/drugs" >&2
  exit 1
fi

for specification in train:100 val:20 test:100; do
  split="${specification%%:*}"
  count="${specification##*:}"
  output_dir="${UPSTREAM_ROOT}/${split}"
  python scripts/eval_jacobian_4d_subset.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${output_dir}" \
    --split "${split}" \
    --data_dir "${PROCESSED_DATA_DIR}" \
    --num_molecules "${count}" \
    --start_idx 0 \
    --seed "${SAMPLE_SEED}" \
    --device cuda \
    --allow_non_jacobian

  python scripts/build_flexbond_init_cache.py \
    --init_path "${output_dir}/generated_files.pkl" \
    --output_dir "${CACHE_ROOT}" \
    --split "${split}" \
    --generator_name "ETFlow-drugs-o3" \
    --generator_checkpoint "${CHECKPOINT}" \
    --sample_seed "${SAMPLE_SEED}" \
    --data_dir "${PROCESSED_DATA_DIR}"

  python scripts/check_flexbond_graph_consistency.py \
    --cache_dir "${CACHE_ROOT}" --split "${split}"
done

python scripts/export_flexbond_inference_cache.py \
  --cache_dir "${CACHE_ROOT}" --split test \
  --output_dir "${INFERENCE_ROOT}"
python scripts/check_flexbond_inference_no_labels.py \
  --cache_dir "${INFERENCE_ROOT}" --split test
python scripts/build_flexbond_eval_manifest.py \
  --cache_dir "${INFERENCE_ROOT}" --split test \
  --max_molecules 100 --output "${MANIFEST}"
python scripts/summarize_flexbond_cache.py \
  --cache_dir "${CACHE_ROOT}" \
  --output data/formal_small_cache_summary.json

echo "Formal-small data preparation complete. Manifest: ${MANIFEST}"
