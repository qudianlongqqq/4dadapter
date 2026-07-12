#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
LOG_ROOT="logs_global_coupled_4d"
RUN_DIR="${LOG_ROOT}/smoke200"
DIAG_DIR="diagnostics/global_coupled_4d/smoke200"
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

checkpoint_is_complete() {
  [[ -s "$1" ]] && python -c 'import torch,sys; p=torch.load(sys.argv[1],map_location="cpu",weights_only=False); raise SystemExit(0 if int(p.get("global_step",0))>=200 else 1)' "$1"
}

CHECKPOINT="${RUN_DIR}/checkpoints/step200.ckpt"
if ! checkpoint_is_complete "${CHECKPOINT}"; then
  LAST_CHECKPOINT="${RUN_DIR}/checkpoints/last.ckpt"
  if checkpoint_is_complete "${LAST_CHECKPOINT}"; then
    CHECKPOINT="${LAST_CHECKPOINT}"
  else
    printf '%s\n' "SMOKE_TRAIN" > "${LOG_ROOT}/CURRENT_STAGE"
    python scripts/train_global_coupled_4d_flow.py \
      --config "${CONFIG}" --cache_dir "${REFERENCE}" --output_dir "${RUN_DIR}" \
      --mode smoke --max_steps 200 --max_molecules 32 --checkpoint_steps 100,200 \
      --val_check_interval 100 --resume_from_checkpoint auto &
    echo $! > "${LOG_ROOT}/TRAIN.pid"
    wait "$(cat "${LOG_ROOT}/TRAIN.pid")"
    rm -f "${LOG_ROOT}/TRAIN.pid"
    CHECKPOINT="${RUN_DIR}/checkpoints/step200.ckpt"
  fi
fi
checkpoint_is_complete "${CHECKPOINT}"
touch "${LOG_ROOT}/SMOKE_TRAIN_COMPLETED"

SAMPLES="${DIAG_DIR}/step200_alpha05_samples.pt"
if [[ -s "${SAMPLES}" ]]; then
  if repair_output="$(python scripts/repair_sample_payload_provenance.py \
      --payload "${SAMPLES}" --manifest "${MANIFEST}" \
      --inference_cache "${INFERENCE}" --split test --max_molecules 20 \
      --expected_method global_coupled_4d_adapter)"; then
    echo "${repair_output}"
    if [[ "${repair_output}" == REPAIRED* ]]; then
      rm -f "${LOG_ROOT}/SMOKE_EVAL_COMPLETED"
    fi
  else
    echo "Existing Smoke sample has mismatched ordered IDs, molecule IDs, or x_init_hash; resampling."
    rm -f "${SAMPLES}" "${DIAG_DIR}/step200_alpha05_samples_trajectory.csv"
    rm -f "${LOG_ROOT}/SMOKE_SAMPLE_COMPLETED" "${LOG_ROOT}/SMOKE_EVAL_COMPLETED"
  fi
fi
if [[ ! -s "${SAMPLES}" ]]; then
  printf '%s\n' "SMOKE_SAMPLE" > "${LOG_ROOT}/CURRENT_STAGE"
  python scripts/sample_global_coupled_4d_flow.py \
    --checkpoint "${CHECKPOINT}" --config "${RUN_DIR}/config.resolved.yaml" \
    --cache_dir "${INFERENCE}" --manifest "${MANIFEST}" --split test \
    --output "${SAMPLES}" --max_molecules 20 --update_scale 0.5 \
    --save_trajectory_metrics &
  echo $! > "${LOG_ROOT}/SAMPLE.pid"
  wait "$(cat "${LOG_ROOT}/SAMPLE.pid")"
  rm -f "${LOG_ROOT}/SAMPLE.pid"
fi
python scripts/repair_sample_payload_provenance.py \
  --payload "${SAMPLES}" --manifest "${MANIFEST}" \
  --inference_cache "${INFERENCE}" --split test --max_molecules 20 \
  --expected_method global_coupled_4d_adapter
touch "${LOG_ROOT}/SMOKE_SAMPLE_COMPLETED"

EVAL_DIR="${DIAG_DIR}/step200_alpha05_eval"
if [[ ! -e "${LOG_ROOT}/SMOKE_EVAL_COMPLETED" || ! -s "${EVAL_DIR}/summary.csv" ]]; then
  printf '%s\n' "SMOKE_EVAL" > "${LOG_ROOT}/CURRENT_STAGE"
  rm -f "${EVAL_DIR}/summary.csv" "${EVAL_DIR}/summary.json" \
    "${EVAL_DIR}/summary.md" "${EVAL_DIR}/sample_diagnostics.csv" \
    "${EVAL_DIR}/update_diagnostics.csv" "${EVAL_DIR}/COMPLETED" \
    "${EVAL_DIR}/FAILED" "${EVAL_DIR}/STARTED"
  python scripts/eval_global_coupled_4d_flow.py \
    --manifest "${MANIFEST}" --inference_cache "${INFERENCE}" \
    --reference_cache "${REFERENCE}" --split test --samples "${SAMPLES}" \
    --output_dir "${EVAL_DIR}" &
  echo $! > "${LOG_ROOT}/EVAL.pid"
  wait "$(cat "${LOG_ROOT}/EVAL.pid")"
  rm -f "${LOG_ROOT}/EVAL.pid"
fi

python -c 'import csv,math,sys,torch,pathlib; rows=list(csv.DictReader(open(sys.argv[1],encoding="utf-8-sig"))); assert rows; numeric=[]
for row in rows:
  for value in row.values():
    try: numeric.append(float(value))
    except (TypeError,ValueError): pass
assert numeric and all(math.isfinite(v) for v in numeric)
payload=torch.load(sys.argv[2],map_location="cpu",weights_only=False); assert payload["records"] and payload["failure_rate"] <= .05
fallback=[float(record.get("solver_fallback_rate",0.0)) for record in payload["records"]]; assert all(0.0 <= v <= 1.0 and math.isfinite(v) for v in fallback)
trajectory=list(csv.DictReader(open(sys.argv[3],encoding="utf-8-sig"))); assert trajectory
internal=[float(row["internal_norm"]) for row in trajectory]; orth=[float(row["orthogonality_error"]) for row in trajectory]
assert all(math.isfinite(v) for v in internal+orth); assert max(internal)>1e-10; assert max(orth)<1e-3
metrics={}
for row in csv.DictReader(open(sys.argv[4],encoding="utf-8-sig")):
  for key,value in row.items():
    if value not in (None,""): metrics[key]=value
def metric(name):
  for key in (name,name+"_step",name+"_epoch"):
    if key in metrics:return float(metrics[key])
  raise AssertionError("missing metric "+name)
assert math.isfinite(metric("val/final_loss")); assert metric("val/jacobian_effective_rank")>0
assert metric("val/projection_orthogonality_error")<1e-3
assert metric("val/projection_reconstruction_error")<1e-5' "${EVAL_DIR}/summary.csv" "${SAMPLES}" "${DIAG_DIR}/step200_alpha05_samples_trajectory.csv" "${RUN_DIR}/metrics.csv"
touch "${LOG_ROOT}/SMOKE_EVAL_COMPLETED"
rm -f "${LOG_ROOT}/FAILED"
touch "${LOG_ROOT}/SMOKE_COMPLETED" "${LOG_ROOT}/SMOKE_PASSED"
echo "GLOBAL COUPLED 4D SMOKE PASSED"
