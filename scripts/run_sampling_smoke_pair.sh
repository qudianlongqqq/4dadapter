#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export DATA_DIR=/home/aidd5090/Experiment/qdl/data/etflow_geom
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ROOT="${1:-logs_eval_smoke/pair_seed42_base_vs_scale001_$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${RUN_ROOT}/base"
JAC_DIR="${RUN_ROOT}/scale001_q0001"
mkdir -p "${BASE_DIR}" "${JAC_DIR}"
exec > >(tee -a "${RUN_ROOT}/master.log") 2>&1

BASE_CONFIG="logs_replicates/jacobian_4d_seq_42_43_44_5000steps_so3_20260701_140501/seed42/base/config.resolved.yaml"
BASE_CHECKPOINT="logs_replicates/jacobian_4d_seq_42_43_44_5000steps_so3_20260701_140501/seed42/base/checkpoints/jacobian-4d-4750.ckpt"
JAC_CONFIG="logs_replicates/jacobian_4d_seq_42_43_44_5000steps_so3_20260701_140501/seed42/scale001_q0001/config.resolved.yaml"
JAC_CHECKPOINT="logs_replicates/jacobian_4d_seq_42_43_44_5000steps_so3_20260701_140501/seed42/scale001_q0001/checkpoints/jacobian-4d-3750.ckpt"

status=0
echo "run_root: ${RUN_ROOT}"
echo "Starting baseline smoke"
if ! python scripts/eval_jacobian_4d_smoke.py \
  --config "${BASE_CONFIG}" \
  --checkpoint "${BASE_CHECKPOINT}" \
  --output_dir "${BASE_DIR}" \
  --num_molecules 3 \
  --start_idx 0 \
  --device cuda \
  --debug \
  --allow_non_jacobian; then
  status=1
fi

echo "Starting 4D smoke"
if ! python scripts/eval_jacobian_4d_smoke.py \
  --config "${JAC_CONFIG}" \
  --checkpoint "${JAC_CHECKPOINT}" \
  --output_dir "${JAC_DIR}" \
  --num_molecules 3 \
  --start_idx 0 \
  --device cuda \
  --debug; then
  status=1
fi

if ! python - "${BASE_DIR}/smoke_output.pt" "${JAC_DIR}/smoke_output.pt" <<'PY'
import sys
from pathlib import Path

import torch

base_path, jac_path = map(Path, sys.argv[1:])
base = torch.load(base_path, map_location="cpu", weights_only=False)
jac = torch.load(jac_path, map_location="cpu", weights_only=False)

def calls(payload):
    return [row.get("jacobian_4d_head_calls") for row in payload.get("molecules", [])]

base_calls = calls(base)
jac_calls = calls(jac)
print(f"base num_successes={base.get('num_successes')} num_failures={base.get('num_failures')}")
print(f"4D num_successes={jac.get('num_successes')} num_failures={jac.get('num_failures')}")
print(f"base jacobian_4d_head_calls={base_calls}")
print(f"4D jacobian_4d_head_calls={jac_calls}")

ok = (
    int(base.get("num_successes", 0)) > 0
    and int(jac.get("num_successes", 0)) > 0
    and all(value in (0, None) for value in base_calls)
    and bool(jac_calls)
    and all(value is not None and int(value) > 0 for value in jac_calls)
)
raise SystemExit(0 if ok else 1)
PY
then
  status=1
fi

if [[ "${status}" -eq 0 ]]; then
  echo "SMOKE PAIR PASSED"
else
  echo "SMOKE PAIR FAILED"
fi
exit "${status}"
