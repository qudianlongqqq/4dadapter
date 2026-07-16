#!/usr/bin/env python
"""Fail-close the Rescue V2 unattended pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from etflow.commons.global_coupled_4d_sampling import atomic_json_save
from etflow.commons.run_timing import RunTiming, iso_now, write_heartbeat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    status = "PREFLIGHT_FAILED" if args.preflight else "FAILED"
    write_heartbeat(
        args.output_dir, status=status, current_step=0, target_step=20000,
        latest_error=args.reason,
    )
    state_path = Path("reports/ecir_mvr/progressive_state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({
        "current_stage": "MEDIUM_SEED42_RESCUE_V2_FAILED",
        "current_decision": status,
        "medium_rescue_v2_permitted": False,
        "medium_rescue_v2_failure_phase": args.phase,
        "medium_rescue_v2_failure_reason": args.reason,
        "20k_permitted": False, "100k_permitted": False, "100k_started": False,
        "seed43_started": False, "seed44_started": False,
        "test_records_read": 0, "next_command": None, "next_commands": [],
        "updated_at": iso_now(),
    })
    atomic_json_save(state, state_path)
    timing = RunTiming(args.output_dir)
    timing.mark("pipeline_end", status=status, phase=args.phase, reason=args.reason)
    print(json.dumps({"status": status, "phase": args.phase, "reason": args.reason}, indent=2))


if __name__ == "__main__":
    main()
