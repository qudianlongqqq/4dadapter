#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
LOG_ROOT="logs_global_coupled_4d"
RUN_DIR="${LOG_ROOT}/smoke"
DIAG_DIR="diagnostics/global_coupled_4d/smoke"
CONFIG="configs/global_coupled_4d_local025_matched.yaml"
MANIFEST="${GLOBAL4D_MANIFEST:-eval_manifest_formal_small.json}"
INFERENCE="${GLOBAL4D_INFERENCE_CACHE:-data/flexbond_inference_formal_small}"
REFERENCE="${GLOBAL4D_REFERENCE_CACHE:-data/flexbond_cache_formal_small}"
mkdir -p "${RUN_DIR}" "${DIAG_DIR}" "${LOG_ROOT}"
STAGE="SMOKE"

fail() {
  code=$?
  command_text="${BASH_COMMAND}"
  tail_text="$(tail -100 "${RUN_DIR}/smoke.log" 2>/dev/null || true)"
  python -c 'import json,sys,datetime,pathlib; pathlib.Path(sys.argv[1]).write_text(json.dumps({"stage":sys.argv[2],"time":datetime.datetime.now().astimezone().isoformat(),"command":sys.argv[3],"exit_code":int(sys.argv[4]),"log":sys.argv[5],"tail":sys.argv[6]},indent=2),encoding="utf-8")' \
    "${LOG_ROOT}/FAILED" "${STAGE}" "${command_text}" "${code}" "${RUN_DIR}/smoke.log" "${tail_text}"
  exit "${code}"
}
trap fail ERR
exec > >(tee -a "${RUN_DIR}/smoke.log") 2>&1

for path in "${MANIFEST}" "${INFERENCE}" "${REFERENCE}"; do
  [[ -e "${path}" ]] || { echo "Missing smoke input: ${path}"; exit 2; }
done

CHECKPOINT="${RUN_DIR}/checkpoints/step200.ckpt"
if [[ ! -s "${CHECKPOINT}" ]]; then
  python scripts/train_global_coupled_4d_flow.py \
    --config "${CONFIG}" --cache_dir "${REFERENCE}" --output_dir "${RUN_DIR}" \
    --mode smoke --max_steps 200 --max_molecules 32 --checkpoint_steps 100,200 \
    --resume_from_checkpoint auto &
  echo $! > "${LOG_ROOT}/TRAIN.pid"
  wait "$(cat "${LOG_ROOT}/TRAIN.pid")"
  rm -f "${LOG_ROOT}/TRAIN.pid"
fi
python -c 'import torch,sys; p=torch.load(sys.argv[1],map_location="cpu",weights_only=False); assert int(p.get("global_step",0))>=200' "${CHECKPOINT}"

SAMPLES="${DIAG_DIR}/step200_alpha05_samples.pt"
if [[ ! -s "${SAMPLES}" ]]; then
  python scripts/sample_global_coupled_4d_flow.py \
    --checkpoint "${CHECKPOINT}" --config "${RUN_DIR}/config.resolved.yaml" \
    --cache_dir "${INFERENCE}" --manifest "${MANIFEST}" --split test \
    --output "${SAMPLES}" --max_molecules 20 --update_scale 0.5 \
    --save_trajectory_metrics &
  echo $! > "${LOG_ROOT}/SAMPLE.pid"
  wait "$(cat "${LOG_ROOT}/SAMPLE.pid")"
  rm -f "${LOG_ROOT}/SAMPLE.pid"
fi

EVAL_DIR="${DIAG_DIR}/step200_alpha05_eval"
if [[ ! -s "${EVAL_DIR}/summary.csv" ]]; then
  python scripts/eval_global_coupled_4d_flow.py \
    --manifest "${MANIFEST}" --inference_cache "${INFERENCE}" \
    --reference_cache "${REFERENCE}" --split test --samples "${SAMPLES}" \
    --output_dir "${EVAL_DIR}" &
  echo $! > "${LOG_ROOT}/EVAL.pid"
  wait "$(cat "${LOG_ROOT}/EVAL.pid")"
  rm -f "${LOG_ROOT}/EVAL.pid"
fi

python -c 'import csv,math,sys,torch; rows=list(csv.DictReader(open(sys.argv[1],encoding="utf-8-sig"))); assert rows; numeric=[]
for row in rows:
  for value in row.values():
    try: numeric.append(float(value))
    except (TypeError,ValueError): pass
assert numeric and all(math.isfinite(v) for v in numeric)
payload=torch.load(sys.argv[2],map_location="cpu",weights_only=False); assert payload["records"] and payload["failure_rate"] <= .05
assert all(math.isfinite(float(record.get("solver_fallback_rate",0.0))) for record in payload["records"])' "${EVAL_DIR}/summary.csv" "${SAMPLES}"
rm -f "${LOG_ROOT}/FAILED"
touch "${LOG_ROOT}/SMOKE_PASSED"
echo "GLOBAL COUPLED 4D SMOKE PASSED"
