#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
CACHE_DIR="${ECIR_CACHE_DIR:-data/flexbond_cache_formal_large}"
ATLAS_DIR="${ECIR_ATLAS_DIR:-data/ecir_error_atlas}"
CARTESIAN_TRAIN_CACHE="${ECIR_CARTESIAN_TRAIN_CACHE:-}"
CARTESIAN_VAL_CACHE="${ECIR_CARTESIAN_VAL_CACHE:-}"
SOURCE_ARGS=(--cache_dir "${CACHE_DIR}" --source_type upstream_etflow_formal)
if [[ -n "${CARTESIAN_TRAIN_CACHE}" && -n "${CARTESIAN_VAL_CACHE}" ]]; then
  SOURCE_CONFIG="diagnostics/ecir/runtime_sources_linux.json"
  mkdir -p "$(dirname "${SOURCE_CONFIG}")"
  "${PYTHON}" -c 'import json,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps({"sources":[{"source_type":"upstream_etflow_formal","cache_dir":sys.argv[2],"coordinate_key":"x_init","checkpoint":"formal_upstream","NFE":10,"solver":"etflow","seed":42},{"source_type":"cartesian_teacher_100k","split_paths":{"train":sys.argv[3],"val":sys.argv[4]},"coordinate_key":"x_cart","checkpoint":"cartesian_step100000","NFE":10,"solver":"cartesian_refinement","seed":42}]},indent=2),encoding="utf-8")' "${SOURCE_CONFIG}" "${CACHE_DIR}" "${CARTESIAN_TRAIN_CACHE}" "${CARTESIAN_VAL_CACHE}"
  SOURCE_ARGS=(--sources_config "${SOURCE_CONFIG}")
fi

"${PYTHON}" scripts/build_conformer_error_atlas.py \
  "${SOURCE_ARGS[@]}" \
  --splits train,val,test --limits 500,100,100 \
  --output_dir "${ATLAS_DIR}" --report docs/ECIR_ERROR_ATLAS_REPORT.md

"${PYTHON}" scripts/train_ecir_flow.py --config configs/ecir_flow_smoke.yaml \
  --cache_dir "${CACHE_DIR}" --target_cache_dir "${ATLAS_DIR}/targets" \
  --device cpu --output_dir logs_ecir/stage1_cpu
"${PYTHON}" scripts/train_ecir_flow.py --config configs/ecir_flow_smoke.yaml \
  --cache_dir "${CACHE_DIR}" --target_cache_dir "${ATLAS_DIR}/targets" \
  --device cuda --output_dir logs_ecir/stage1_cuda

"${PYTHON}" scripts/train_ecir_flow.py --config configs/ecir_flow_formal_small.yaml \
  --cache_dir "${CACHE_DIR}" --target_cache_dir "${ATLAS_DIR}/targets" \
  --output_dir logs_ecir/stage2_heterogeneous_500_100_5k
"${PYTHON}" scripts/eval_ecir_refiner.py \
  --checkpoint logs_ecir/stage2_heterogeneous_500_100_5k/step005000.ckpt \
  --cache_dir "${CACHE_DIR}" --target_cache_dir "${ATLAS_DIR}/targets" \
  --atlas_path "${ATLAS_DIR}/val.parquet" --split val --max_records 100 --steps 4 \
  --output_dir diagnostics/ecir/stage2_heterogeneous_eval

decision=$("${PYTHON}" -c 'import json; print(json.load(open("diagnostics/ecir/stage2_heterogeneous_eval/result.json"))["status"])')
if [[ "${decision}" != "GO" ]]; then
  echo "ECIR Stage 2 NO_GO. Stage 3 and formal-large are blocked." >&2
  exit 3
fi

echo "ECIR Stage 2 GO. Review before executing:"
echo "${PYTHON} scripts/train_ecir_flow.py --config configs/ecir_flow_formal_medium.yaml --cache_dir ${CACHE_DIR} --target_cache_dir ${ATLAS_DIR}/targets"
echo "After an independent Stage 3 GO:"
echo "${PYTHON} scripts/train_ecir_flow.py --config configs/ecir_flow_formal_large.yaml --cache_dir ${CACHE_DIR} --target_cache_dir ${ATLAS_DIR}/targets"
