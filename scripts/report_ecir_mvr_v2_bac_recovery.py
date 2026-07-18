#!/usr/bin/env python3
"""Freeze the preregistered Phase-1 recovery comparison and decision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("diagnostics/ecir_mvr/v2_bac_recovery")
RUNS = ROOT / "runs"
RUN_NAMES = {
    "A0": "a0_pilot_1000step_seed43018",
    "D0": "d0_pilot_1000step_seed43018",
    "D1": "d1_pilot_1000step_seed43018",
}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    comparison = []
    for candidate, run_name in RUN_NAMES.items():
        run = RUNS / run_name
        metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
        if (
            metadata["status"] != "COMPLETED"
            or metadata["completed_steps"] != 1000
            or metadata["test_records_read"] != 0
            or metadata["test_assets_opened"]
            or metadata["frozen_holdout_records_opened"] != 0
        ):
            raise RuntimeError(f"candidate is incomplete or violates isolation: {candidate}")
        training = pd.read_csv(run / "training_metrics.csv")
        all_subset = metadata["active_subsets"]["all"]
        angle_subset = metadata["active_subsets"]["angle_active"]
        clash_subset = metadata["active_subsets"]["clash_active"]
        metrics = metadata["metrics"]
        comparison.append(
            {
                "candidate": candidate,
                "steps": metadata["completed_steps"],
                "records": all_subset["records"],
                "molecules": all_subset["molecules"],
                "bond_delta": metrics["bond_delta"],
                "angle_delta": metrics["angle_delta"],
                "clash_delta": metrics["clash_delta"],
                "weighted_bac_delta": all_subset["metrics"]["weighted_bac_delta"]["mean"],
                "acceptance_fraction": metrics["accepted_fraction"],
                "rollback_fraction": metrics["rollback_fraction"],
                "ring_delta": metrics["ring_delta"],
                "rmsd_delta": metrics["rmsd_delta"],
                "mat_p_delta": metrics["mat_p_delta"],
                "mat_r_delta": metrics["mat_r_delta"],
                "cov_p_delta": metrics["cov_p_delta"],
                "cov_r_delta": metrics["cov_r_delta"],
                "mean_displacement": metrics["mean_displacement"],
                "angle_active_records": angle_subset["records"],
                "angle_active_molecules": angle_subset["molecules"],
                "angle_active_delta": angle_subset["metrics"]["angle_delta"]["mean"],
                "angle_active_ci95_low": angle_subset["metrics"]["angle_delta"]["ci95_low"],
                "angle_active_ci95_high": angle_subset["metrics"]["angle_delta"]["ci95_high"],
                "clash_active_records": clash_subset["records"],
                "gradient_norm_logged_max": float(training.gradient_norm.max()),
                "gradient_norm_logged_median": float(training.gradient_norm.median()),
                "runtime_seconds": metadata["elapsed_seconds"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "failure_rate": metrics["failure_rate"],
            }
        )
    table = pd.DataFrame(comparison)
    table.to_csv(ROOT / "candidate_comparison.csv", index=False)
    d1 = table.set_index("candidate").loc["D1"]
    phase1_fixed = bool(
        d1.angle_active_delta < 0
        and d1.angle_active_ci95_high < 0
        and d1.bond_delta < 0
        and d1.acceptance_fraction > 0.5
        and d1.ring_delta <= 0
        and d1.failure_rate == 0
    )
    decision = {
        "schema_version": "mcvr-v2-bac-recovery-decision-v1",
        "decision": "PHASE1_FIXED" if phase1_fixed else "PHASE1_NOT_FIXED",
        "fixed_scope": "Bond and active-Angle Cartesian recovery; Clash unsupported",
        "best_candidate": "D1" if phase1_fixed else None,
        "best_checkpoint_sha256": d1.checkpoint_sha256 if phase1_fixed else None,
        "enter_jacobian_comparison": phase1_fixed,
        "start_10k": False,
        "start_formal_large": False,
        "expand_model": False,
        "clash_conclusion": "INCONCLUSIVE_DATA_SUPPORT",
        "reasons": [
            "D1 active-Angle bootstrap CI is strictly below zero",
            "A0 and D0 active-Angle delta are exactly zero",
            "D1 improves Bond and Ring with zero chirality delta",
            "D1 acceptance is 97.27% versus 5.18% A0 and 2.44% D0",
            "only one development record has an active clash",
            "D1 movement is materially larger and logged gradients have large clipped spikes",
        ],
        "remaining_risks": [
            "effect attribution combines three linked repair flags",
            "D1 mean displacement is about 0.0067 Angstrom versus micro-Angstrom baselines",
            "logged pre-clip gradient norm reaches about 141.7",
            "no new holdout confirmation is allowed",
            "Clash efficacy is statistically unsupported",
        ],
        "gpu_run_budget": {
            "maximum": 5,
            "used": 5,
            "runs": [
                "two-batch read-only diagnostic",
                "D1 200-step smoke",
                "A0 1000-step pilot",
                "D0 1000-step pilot",
                "D1 1000-step pilot",
            ],
        },
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "validation_only": True,
    }
    _write_json(ROOT / "recommended_next_step.json", decision)
    audit = json.loads((ROOT / "audit_summary.json").read_text(encoding="utf-8"))
    audit.update(
        {
            "status": "PHASE1_AUDIT_AND_RECOVERY_COMPLETE",
            "root_cause_classification": {
                "DATA_SUPPORT_FAILURE": "SUPPORTED_FOR_CLASH",
                "TARGET_SCALE_FAILURE": "PARTIAL",
                "TARGET_CONFLICT_FAILURE": "NOT_PRIMARY",
                "LEARNING_SIGNAL_FAILURE": "REJECTED",
                "PROPOSAL_ATTENUATION_FAILURE": "SUPPORTED",
                "SAFETY_BOTTLENECK": "SUPPORTED",
                "CAPACITY_LIMIT": "REJECTED",
                "METRIC_POWER_FAILURE": "SUPPORTED_FOR_CLASH",
            },
            "selected_repair": [
                "active-only constraint scatter normalization",
                "full first-step inference-field BAC proposal loss",
                "finite hard-safe backtracking",
            ],
            "phase1_decision": decision["decision"],
            "candidate_comparison": comparison,
            "test_records_read": 0,
            "test_assets_opened": False,
            "frozen_holdout_records_opened": 0,
        }
    )
    _write_json(ROOT / "audit_summary.json", audit)
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
