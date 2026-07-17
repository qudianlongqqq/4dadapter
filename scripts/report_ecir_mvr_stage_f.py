#!/usr/bin/env python
"""Render formal Stage F reports and update state without changing E0/E1 history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import pandas as pd
import yaml

from etflow.commons.global_coupled_4d_sampling import atomic_json_save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ecir_mvr_stage_f_feature_confidence.yaml"))
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = args.input_dir or Path(config["output_dir"])
    result = json.loads((source / "validation_result.json").read_text(encoding="utf-8"))
    if result.get("smoke"):
        raise RuntimeError("Stage F smoke cannot update formal reports or state")
    history = pd.read_csv(source / "training_history.csv")
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MCVR Stage F Feature-Conditioned Sign-Safe Confidence", "",
        "D1-B remained strict-loaded and frozen. Only the small bounded confidence calibrator was trained on train molecules; validation was used once after internal-check selection and test records remained zero.", "",
        f"Decision: **{result['decision']}**", "",
        f"Selected internal-check step: `{result['selected_step']}`.", "",
        f"Model-to-target recovery: `{result['metrics']['model_to_target_recovery']:.12f}`.", "",
        f"Bond relative improvement: `{result['metrics']['bond_relative_improvement']:.12f}`.", "",
        "| Gate | Result |", "|---|---|",
        *[f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in result["criteria"].items()], "",
        f"Internal-check observations recorded: `{len(history)}`.", "",
        "`confidence_all_one` remains `DIAGNOSTIC_ORACLE_ONLY`.",
    ]
    (args.docs_dir / "MCVR_STAGE_F_FEATURE_CONFIDENCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    state_path = Path("reports/ecir_mvr/progressive_state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    frozen = {
        "medium_seed42_schedule_v4_decision": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
        "stage_d_pilot_decision": "STAGE_D_NO_ADDED_VALUE",
        "stage_e0_decision": "STAGE_E0_NO_ADDED_VALUE",
        "stage_e1_primary_cause": "CALIBRATION_OBJECTIVE_MISMATCH",
    }
    for name, expected in frozen.items():
        if state.get(name) != expected:
            raise RuntimeError(f"historical conclusion changed before Stage F reporting: {name}")
    state.update({
        "current_stage": "MCVR_STAGE_F_PILOT_COMPLETE",
        "stage_f_decision": result["decision"], "stage_f_selected_step": result["selected_step"],
        "stage_f_seed43_44_permitted": result["decision"] == "STAGE_F_FEATURE_CONFIDENCE_PASS",
        "formal_training_permitted": False, "stage_f_100k_permitted": False,
        "test_records_read": 0, "next_command": None, "next_commands": [],
    })
    atomic_json_save(state, state_path)
    print(json.dumps({"decision": result["decision"], "reports_written": 1, "state_updated": True, "test_records_read": 0}, indent=2))


if __name__ == "__main__":
    main()
