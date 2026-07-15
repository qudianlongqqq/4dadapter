#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

: "${GLOBAL4D_REFERENCE_CACHE:?Set GLOBAL4D_REFERENCE_CACHE to the training/reference cache}"
: "${GLOBAL4D_INFERENCE_CACHE:?Set GLOBAL4D_INFERENCE_CACHE to the label-free inference cache}"
: "${GLOBAL4D_MANIFEST:?Set GLOBAL4D_MANIFEST to the smoke evaluation manifest}"

RUN_DIR="${GLOBAL4D_SMOKE_RUN_DIR:-logs_gated_global4d_v2/linux_smoke}"
SAMPLE_DIR="${RUN_DIR}/sampling"
EVAL_DIR="${RUN_DIR}/evaluation"
CONFIG="configs/gated_global4d_v2_pilot.yaml"
mkdir -p "${RUN_DIR}" "${SAMPLE_DIR}"

python - <<'PY'
import platform
import torch
print({"python": platform.python_version(), "torch": torch.__version__,
       "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
if not torch.cuda.is_available():
    raise SystemExit("Linux CUDA smoke requires an available CUDA device")
PY

python -m pytest \
  tests/test_global_coupled_4d_flow.py \
  tests/test_gated_global4d_v2.py \
  tests/test_global_coupled_4d_sampling_resume.py \
  -q -p no:cacheprovider

python scripts/train_gated_global4d_v2.py \
  --config "${CONFIG}" \
  --cache_dir "${GLOBAL4D_REFERENCE_CACHE}" \
  --output_dir "${RUN_DIR}" \
  --mode smoke \
  --max_steps 10 \
  --max_molecules 5 \
  --batch_size 2 \
  --accumulate_grad_batches 1 \
  --num_workers 0 \
  --no-pin_memory \
  --checkpoint_steps 10 \
  --val_check_interval 5 \
  --resume_from_checkpoint none

python scripts/train_gated_global4d_v2.py \
  --config "${CONFIG}" \
  --cache_dir "${GLOBAL4D_REFERENCE_CACHE}" \
  --output_dir "${RUN_DIR}" \
  --mode smoke \
  --max_steps 20 \
  --max_molecules 5 \
  --batch_size 2 \
  --accumulate_grad_batches 1 \
  --num_workers 0 \
  --no-pin_memory \
  --checkpoint_steps 10,20 \
  --val_check_interval 5 \
  --resume_from_checkpoint auto

python scripts/sample_global_coupled_4d_flow.py \
  --checkpoint "${RUN_DIR}/checkpoints/step20.ckpt" \
  --config "${RUN_DIR}/config.resolved.yaml" \
  --cache_dir "${GLOBAL4D_INFERENCE_CACHE}" \
  --manifest "${GLOBAL4D_MANIFEST}" \
  --split test \
  --output "${SAMPLE_DIR}/samples.pt" \
  --max_molecules 5 \
  --refinement_steps 2 \
  --update_scale 0.5 \
  --device cuda

python scripts/eval_global_coupled_4d_flow.py \
  --manifest "${GLOBAL4D_MANIFEST}" \
  --inference_cache "${GLOBAL4D_INFERENCE_CACHE}" \
  --reference_cache "${GLOBAL4D_REFERENCE_CACHE}" \
  --split test \
  --samples "${SAMPLE_DIR}/samples.pt" \
  --output_dir "${EVAL_DIR}"

echo "GATED GLOBAL4D V2 LINUX SMOKE COMPLETED"
