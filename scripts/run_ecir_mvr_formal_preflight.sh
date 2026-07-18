#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_INDEX="${GPU_INDEX:?Set GPU_INDEX to the physical GPU index to benchmark}"
EXTRA_ARGS=()
if [[ "${ALLOW_SHARED_GPU:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--allow-shared-gpu)
fi
if [[ "${CAPACITY_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--capacity-only)
fi
if [[ -n "${CANDIDATE_MICRO_BATCHES:-}" ]]; then
  EXTRA_ARGS+=(--candidate-micro-batches "$CANDIDATE_MICRO_BATCHES")
fi

exec python scripts/preflight_ecir_mvr_formal_large.py \
  --config configs/ecir_mvr_formal_large_d1b_base.yaml \
  --gpu-index "$GPU_INDEX" \
  --target-effective-batch "${TARGET_EFFECTIVE_BATCH:-64}" \
  --preflight-steps 100 \
  --warmup-steps 20 \
  "${EXTRA_ARGS[@]}"
