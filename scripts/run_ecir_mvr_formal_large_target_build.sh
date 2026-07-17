#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-configs/ecir_mvr_formal_large_minimal_targets.yaml}"
INPUT_CACHE="${INPUT_CACHE:-data/flexbond_cache_formal_large}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/aidd4090v2/Experiment/qdl/data/4dadapter/ecir_mvr/formal_large}"
GPU_ID="${GPU_ID:-0}"

args=(
  python scripts/build_ecir_mvr_formal_large_targets.py
  --config "${CONFIG}"
  --input-cache "${INPUT_CACHE}"
  --output-root "${OUTPUT_ROOT}"
  --gpu-index "${GPU_ID}"
)

case "${1:-}" in
  "") ;;
  --pilot-only) args+=(--pilot-only) ;;
  --no-auto-continue) args+=(--no-auto-continue) ;;
  *) echo "usage: $0 [--pilot-only|--no-auto-continue]" >&2; exit 2 ;;
esac

mkdir -p "${OUTPUT_ROOT}/logs"
"${args[@]}" 2>&1 | tee -a "${OUTPUT_ROOT}/logs/build.log"
