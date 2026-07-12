#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
LOG_ROOT="logs_global_coupled_4d"
python scripts/report_global_coupled_4d_first_result.py --check-only
python scripts/report_global_coupled_4d_first_result.py
touch "${LOG_ROOT}/STOP_REQUESTED_AFTER_FIRST_RESULT"
had_failed=0; [[ -e "${LOG_ROOT}/FAILED" ]] && had_failed=1

pids=()
for file in MASTER.pid SAMPLE.pid EVAL.pid TRAIN.pid; do
  if [[ -f "${LOG_ROOT}/${file}" ]]; then
    pid="$(tr -dc '0-9' < "${LOG_ROOT}/${file}")"
    [[ -n "${pid}" ]] && pids+=("${pid}")
  fi
done
for pid in "${pids[@]}"; do
  kill -TERM "${pid}" 2>/dev/null || true
done
sleep 2
if [[ "${had_failed}" -eq 0 ]]; then
  rm -f "${LOG_ROOT}/FAILED"
fi
rm -f "${LOG_ROOT}/RUNNING" "${LOG_ROOT}/FORMAL_RUNNING" \
  "${LOG_ROOT}/CHECKPOINT_SWEEP_RUNNING"
touch "${LOG_ROOT}/SMALL_SWEEP_STOPPED_AFTER_FIRST_RESULT"
echo "Stopped old 5k sweep after preserving its first valid result"
