#!/usr/bin/env bash
# Manual small run. Invoke only after the 100-molecule smoke run succeeds.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_flexbond_optimizer_20k.sh CACHE_DIR [OUTPUT_ROOT] [NUM_MOLECULES]"
  exit 2
fi

CACHE_DIR="$1"
OUTPUT_ROOT="${2:-logs_flexbond_optimizer}"
NUM_MOLECULES="${3:-1000}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/small20k_${STAMP}"

for MODE in cartesian_optimizer flexbond4d_hybrid_optimizer; do
  python scripts/train_flexbond_optimizer.py \
    --config configs/flexbond_optimizer_egnn.yaml \
    --mode "${MODE}" \
    --cache_dir "${CACHE_DIR}" \
    --output_dir "${RUN_DIR}/${MODE}" \
    --max_molecules "${NUM_MOLECULES}" \
    --max_steps 20000
done

echo "20k training complete. Run sampling/evaluation explicitly from ${RUN_DIR}."
