#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
V7_CONFIG="${ROOT}/configs/ecir_mvr_v7_formal_large.yaml"

seed=""
device=""
config=""
data_audit=""
resume_checkpoint=""
expected_checkpoint_sha=""
output_dir=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_v7_formal_large_seed.sh \
    --seed 42|43 --device cuda:0 --config PRIOR_CONFIG \
    [--data-audit AUDIT.json] [--resume COMPLETED_D1B.ckpt] \
    [--expected-checkpoint-sha256 SHA256] [--output-dir DIR]

Without --resume, the existing frozen D1-B formal trainer is run first.
With --resume, prior training is skipped and the completed D1-B checkpoint is
strict-loaded and bound to the parameter-free V7 wrapper.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) seed="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --data-audit) data_audit="$2"; shift 2 ;;
    --resume) resume_checkpoint="$2"; shift 2 ;;
    --expected-checkpoint-sha256) expected_checkpoint_sha="$2"; shift 2 ;;
    --output-dir) output_dir="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${seed}" != "42" && "${seed}" != "43" ]]; then
  echo "--seed must be 42 or 43" >&2
  exit 2
fi
if [[ "${seed}" == "42" ]]; then
  frozen_checkpoint_sha="721b4384f3a64eef48ead2fc2b4ea35bf83802b84952e8e3f3aa6c5172e33a2f"
else
  frozen_checkpoint_sha="c7f2e5e36a400600951d846b7d11d1d9aa57a0da78d2e540340fe44b470868ca"
fi
if [[ -n "${expected_checkpoint_sha}" && "${expected_checkpoint_sha}" != "${frozen_checkpoint_sha}" ]]; then
  echo "Requested checkpoint SHA differs from the frozen dual-seed plan" >&2
  exit 2
fi
expected_checkpoint_sha="${frozen_checkpoint_sha}"
if [[ -z "${device}" || -z "${config}" ]]; then
  echo "--device and --config are required" >&2
  exit 2
fi

config="$(cd "$(dirname "${config}")" && pwd)/$(basename "${config}")"
if [[ ! -f "${config}" ]]; then
  echo "Training config does not exist: ${config}" >&2
  exit 2
fi
if [[ -z "${output_dir}" ]]; then
  output_dir="${ROOT}/artifacts/ecir_mvr/formal_large/v7_seed${seed}"
fi

configured_seed="$(${PYTHON_BIN} -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["seed"])' "${config}")"
if [[ "${configured_seed}" != "${seed}" ]]; then
  echo "Seed differs between --seed and --config" >&2
  exit 2
fi

prior_mode="resumed"
if [[ -z "${resume_checkpoint}" ]]; then
  if [[ -z "${data_audit}" ]]; then
    echo "--data-audit is required when prior training is not skipped" >&2
    exit 2
  fi
  prior_mode="trained"
  "${PYTHON_BIN}" "${ROOT}/scripts/train_ecir_mvr_medium_rescue_v2.py" \
    --config "${config}" \
    --data_audit "${data_audit}" \
    --device "${device}"
  prior_output="$(${PYTHON_BIN} -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "${config}")"
  resume_checkpoint="${prior_output}/checkpoints/best_noninferior_validity.ckpt"
else
  resume_checkpoint="$(cd "$(dirname "${resume_checkpoint}")" && pwd)/$(basename "${resume_checkpoint}")"
fi

if [[ ! -f "${resume_checkpoint}" ]]; then
  echo "Completed D1-B checkpoint does not exist: ${resume_checkpoint}" >&2
  exit 1
fi

binding=(
  "${PYTHON_BIN}" "${ROOT}/scripts/prepare_ecir_mvr_v7_formal_seed.py"
  --seed "${seed}"
  --training-config "${config}"
  --v7-config "${V7_CONFIG}"
  --checkpoint "${resume_checkpoint}"
  --output-dir "${output_dir}"
  --prior-mode "${prior_mode}"
)
binding+=(--expected-checkpoint-sha256 "${expected_checkpoint_sha}")
"${binding[@]}"

echo "V7 formal seed ${seed} binding complete: ${output_dir}"
