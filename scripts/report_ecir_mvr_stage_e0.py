#!/usr/bin/env python
"""Render formal Stage E0 reports and update progressive state."""

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
    parser.add_argument("--config", type=Path, default=Path("configs/ecir_mvr_stage_e0_confidence_calibration.yaml"))
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = args.input_dir or Path(config["output_dir"])
    result = json.loads((source / "validation_result.json").read_text(encoding="utf-8"))
    if result.get("smoke"):
        raise RuntimeError("smoke results cannot update formal Stage E0 reports or state")
    calibrator = json.loads((source / "calibrator.json").read_text(encoding="utf-8"))
    curve = pd.read_csv(source / "calibration_curve.csv")
    transition = pd.read_csv(source / "bond_transition.csv")
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    check = curve[(curve.split == "internal_check") & (curve.group == "all")].iloc[0]
    calibration_lines = [
        "# MCVR Stage E0 Confidence Calibration", "",
        "Stage E0 fits only a monotonic two-parameter confidence map on training molecules. The D1-B checkpoint, residual predictor, Cartesian branch, solver, safety, trust clipping, acceptance, and torsion-disabled architecture remain frozen.", "",
        "## Calibrator", "", "| Field | Value |", "|---|---:|",
        f"| raw_a | {calibrator['raw_a']:.12f} |",
        f"| a = softplus(raw_a) + epsilon | {calibrator['a']:.12f} |",
        f"| b | {calibrator['b']:.12f} |",
        f"| Internal-check original MAE | {check.original_mae:.12f} |",
        f"| Internal-check calibrated MAE | {check.calibrated_mae:.12f} |", "",
        "`confidence_all_one` remains `DIAGNOSTIC_ORACLE_ONLY` and is not deployable.", "",
        "Calibration fitting read training data only. Validation was evaluated once after fitting; test records read remained zero.",
    ]
    (args.docs_dir / "MCVR_STAGE_E0_CONFIDENCE_CALIBRATION.md").write_text(
        "\n".join(calibration_lines) + "\n", encoding="utf-8"
    )
    decision_lines = [
        "# MCVR Stage E0 Decision", "",
        f"Decision: **{result['decision']}**", "",
        f"Model-to-target recovery: `{result['metrics']['model_to_target_recovery']:.12f}`.", "",
        f"Bond relative improvement: `{result['metrics']['bond_relative_improvement']:.12f}`.", "",
        "| Gate | Result |", "|---|---|",
        *[f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in result["criteria"].items()], "",
        "The formal Stage D conclusion remains **STAGE_D_NO_ADDED_VALUE**. Stage E0 does not authorize 20k, 100k, test evaluation, or additional seeds.",
    ]
    (args.docs_dir / "MCVR_STAGE_E0_DECISION.md").write_text(
        "\n".join(decision_lines) + "\n", encoding="utf-8"
    )
    state_path = Path("reports/ecir_mvr/progressive_state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    frozen = {
        "medium_seed42_schedule_v4_decision": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
        "stage_d_pilot_decision": "STAGE_D_NO_ADDED_VALUE",
        "stage_d2_primary_cause": "CONFIDENCE_CALIBRATION_WEAK",
    }
    for name, expected in frozen.items():
        if state.get(name) != expected:
            raise RuntimeError(f"historical conclusion changed before Stage E0 reporting: {name}")
    state.update({
        "current_stage": "MCVR_STAGE_E0_COMPLETE",
        "stage_e0_status": "COMPLETE", "stage_e0_decision": result["decision"],
        "stage_e0_20k_permitted": False, "stage_e0_100k_permitted": False,
        "test_records_read": 0, "next_command": None, "next_commands": [],
    })
    atomic_json_save(state, state_path)
    print(json.dumps({
        "decision": result["decision"], "reports_written": 2,
        "state_updated": True, "test_records_read": 0,
        "transition_rows": len(transition),
    }, indent=2))


if __name__ == "__main__":
    main()
