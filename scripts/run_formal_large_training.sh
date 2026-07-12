#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
LOG_ROOT="logs_formal_large"
METHODS="${METHODS:-cartesian,global4d}"
mkdir -p "${LOG_ROOT}" reports
rm -f "${LOG_ROOT}/FAILED"
STAGE="VALIDATE"

fail() {
  code=$?; command_text="${BASH_COMMAND}"
  python -c 'import json,sys,datetime,pathlib; pathlib.Path(sys.argv[1]).write_text(json.dumps({"stage":sys.argv[2],"time":datetime.datetime.now().astimezone().isoformat(),"command":sys.argv[3],"exit_code":int(sys.argv[4])},indent=2),encoding="utf-8")' \
    "${LOG_ROOT}/FAILED" "${STAGE}" "${command_text}" "${code}"
  exit "${code}"
}
trap fail ERR

python scripts/validate_formal_large.py
touch "${LOG_ROOT}/DATA_READY"

IFS=',' read -ra requested <<< "${METHODS}"
for method in "${requested[@]}"; do
  case "${method}" in
    cartesian|global4d) ;;
    *) echo "Unsupported METHODS entry: ${method}" >&2; exit 2 ;;
  esac
done

contains_method() {
  [[ ",${METHODS}," == *",$1,"* ]]
}

checkpoint_complete() {
  [[ -s "$1" ]] && python -c 'import torch,sys; p=torch.load(sys.argv[1],map_location="cpu",weights_only=False); raise SystemExit(0 if int(p.get("global_step",0))>=200000 else 1)' "$1"
}

if contains_method cartesian; then
  CART_RUN="${LOG_ROOT}/cartesian_seed42_200k"
  CART_FINAL="${CART_RUN}/checkpoints/step200000.ckpt"
  if ! checkpoint_complete "${CART_FINAL}"; then
    STAGE="CARTESIAN_TRAINING"
    touch "${LOG_ROOT}/CARTESIAN_TRAINING"
    python scripts/train_flexbond_optimizer.py \
      --config configs/formal_large_cartesian_seed42_200k.yaml \
      --mode cartesian_optimizer \
      --cache_dir data/flexbond_cache_formal_large \
      --output_dir "${CART_RUN}" --max_steps 200000 \
      --checkpoint_steps 50000,100000,150000,200000 \
      --resume_from_checkpoint auto
  fi
  checkpoint_complete "${CART_FINAL}"
  rm -f "${LOG_ROOT}/CARTESIAN_TRAINING"
  touch "${LOG_ROOT}/CARTESIAN_COMPLETED"
fi

if contains_method global4d; then
  GLOBAL_RUN="${LOG_ROOT}/global4d_seed42_200k"
  GLOBAL_FINAL="${GLOBAL_RUN}/checkpoints/step200000.ckpt"
  if ! checkpoint_complete "${GLOBAL_FINAL}"; then
    STAGE="GLOBAL4D_TRAINING"
    touch "${LOG_ROOT}/GLOBAL4D_TRAINING"
    python scripts/train_global_coupled_4d_flow.py \
      --config configs/formal_large_global4d_seed42_200k.yaml \
      --cache_dir data/flexbond_cache_formal_large \
      --output_dir "${GLOBAL_RUN}" --mode formal --max_steps 200000 \
      --checkpoint_steps 50000,100000,150000,200000 \
      --resume_from_checkpoint auto
  fi
  checkpoint_complete "${GLOBAL_FINAL}"
  rm -f "${LOG_ROOT}/GLOBAL4D_TRAINING"
  touch "${LOG_ROOT}/GLOBAL4D_COMPLETED"
fi

if [[ -e "${LOG_ROOT}/CARTESIAN_COMPLETED" && -e "${LOG_ROOT}/GLOBAL4D_COMPLETED" ]]; then
  python scripts/summarize_formal_large_training.py
  touch "${LOG_ROOT}/FORMAL_LARGE_TRAINING_COMPLETED"
fi
rm -f "${LOG_ROOT}/FAILED"
echo "FORMAL-LARGE REQUESTED TRAINING METHODS COMPLETED"
