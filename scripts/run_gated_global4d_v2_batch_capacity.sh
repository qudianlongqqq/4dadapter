#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
: "${GLOBAL4D_REFERENCE_CACHE:?Set GLOBAL4D_REFERENCE_CACHE to the training/reference cache}"

python scripts/benchmark_gated_global4d_batch_capacity.py \
  --config configs/gated_global4d_v2_pilot.yaml \
  --cache_dir "${GLOBAL4D_REFERENCE_CACHE}" \
  --batch_sizes 4,8,16,32,48,64,96,128 \
  --compositions low_complexity,mixed,high_complexity \
  --warmup_optimizer_steps 3 \
  --fixed_records_seen 768
