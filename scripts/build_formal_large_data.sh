#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
CANDIDATES="data/flexbond_cache_formal_large_candidates"
CACHE="data/flexbond_cache_formal_large"
INFERENCE="data/flexbond_inference_formal_large"
: "${FORMAL_LARGE_ETFLOW_CHECKPOINT:?Set FORMAL_LARGE_ETFLOW_CHECKPOINT}"
: "${FORMAL_LARGE_PROCESSED_DATA:?Set FORMAL_LARGE_PROCESSED_DATA}"
: "${FORMAL_LARGE_ETFLOW_TRAIN_OUTPUT:?Set FORMAL_LARGE_ETFLOW_TRAIN_OUTPUT}"
: "${FORMAL_LARGE_ETFLOW_VAL_OUTPUT:?Set FORMAL_LARGE_ETFLOW_VAL_OUTPUT}"
: "${FORMAL_LARGE_ETFLOW_TEST_OUTPUT:?Set FORMAL_LARGE_ETFLOW_TEST_OUTPUT}"

declare -A INPUTS=(
  [train]="${FORMAL_LARGE_ETFLOW_TRAIN_OUTPUT}"
  [val]="${FORMAL_LARGE_ETFLOW_VAL_OUTPUT}"
  [test]="${FORMAL_LARGE_ETFLOW_TEST_OUTPUT}"
)
declare -A REFERENCES=(
  [train]="${FORMAL_LARGE_REFERENCE_TRAIN:-}"
  [val]="${FORMAL_LARGE_REFERENCE_VAL:-}"
  [test]="${FORMAL_LARGE_REFERENCE_TEST:-}"
)

for split in train val test; do
  [[ -e "${INPUTS[$split]}" ]] || { echo "Missing ETFlow output: ${INPUTS[$split]}" >&2; exit 2; }
  reference_args=()
  if [[ -n "${REFERENCES[$split]}" ]]; then
    [[ -e "${REFERENCES[$split]}" ]] || { echo "Missing reference source: ${REFERENCES[$split]}" >&2; exit 2; }
    reference_args=(--reference_path "${REFERENCES[$split]}")
  fi
  python scripts/build_flexbond_init_cache.py \
    --init_path "${INPUTS[$split]}" "${reference_args[@]}" \
    --output_dir "${CANDIDATES}" --split "${split}" \
    --generator_name ETFlow --generator_checkpoint "${FORMAL_LARGE_ETFLOW_CHECKPOINT}" \
    --sample_seed 42 --data_dir "${FORMAL_LARGE_PROCESSED_DATA}"
done

python scripts/materialize_formal_large_dataset.py \
  --candidate_cache "${CANDIDATES}" --output_cache "${CACHE}" \
  --manifest_dir manifests
for split in train val test; do
  python scripts/export_flexbond_inference_cache.py \
    --cache_dir "${CACHE}" --split "${split}" --output_dir "${INFERENCE}"
  python scripts/check_flexbond_inference_no_labels.py \
    --cache_dir "${INFERENCE}" --split "${split}"
done
python scripts/validate_formal_large.py \
  --cache "${CACHE}" --inference "${INFERENCE}" --manifest_dir manifests
echo "FORMAL LARGE DATA READY"
