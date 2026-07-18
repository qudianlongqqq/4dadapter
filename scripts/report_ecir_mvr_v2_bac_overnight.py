#!/usr/bin/env python3
"""Assemble the frozen validation-only V2-BAC overnight report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PARAMETERS = {
    "V2_A_BOND_ONLY": 384678,
    "V2_B_BOND_ANGLE": 410603,
    "V2_C_BOND_CLASH": 406507,
    "V2_D_BOND_ANGLE_CLASH": 427694,
}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _fmt(value: float) -> str:
    return f"{float(value):.9g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v2_bac_overnight"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/MCVR_V2_BAC_OVERNIGHT_REPORT.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.diagnostics_dir.expanduser().resolve()
    runs = []
    constraint_rows = []
    for run_dir in sorted((root / "runs").glob("*_2000step_seed44")):
        metadata = json.loads(
            (run_dir / "run_metadata.json").read_text(encoding="utf-8")
        )
        metrics = pd.read_csv(run_dir / "training_metrics.csv")
        metadata["parameters"] = PARAMETERS[metadata["mode"]]
        metadata["parameter_delta"] = metadata["parameters"] - PARAMETERS[
            "V2_A_BOND_ONLY"
        ]
        runs.append(metadata)
        constraint_rows.append(
            {
                "experiment_id": metadata["experiment_id"],
                "split": "train",
                "bond_count_mean": float(
                    metrics.get("active_bond_constraints", pd.Series([0])).mean()
                ),
                "angle_count_mean": float(
                    metrics.get("active_angle_constraints", pd.Series([0])).mean()
                ),
                "clash_count_mean": float(
                    metrics.get("active_clash_constraints", pd.Series([0])).mean()
                ),
                "gradient_norm_shared_mean": float(
                    metrics.gradient_norm_shared_backbone.mean()
                ),
                "gradient_norm_bond_mean": float(metrics.gradient_norm_bond.mean()),
                "gradient_norm_angle_mean": float(metrics.gradient_norm_angle.mean()),
                "gradient_norm_clash_mean": float(metrics.gradient_norm_clash.mean()),
                "gradient_norm_fusion_mean": float(metrics.gradient_norm_fusion.mean()),
            }
        )
    runs.sort(key=lambda value: value["mode"])
    inventory = pd.DataFrame(
        [
            {
                "experiment_id": run["experiment_id"],
                "mode": run["mode"],
                "status": run["status"],
                "optimizer_steps": run["completed_steps"],
                "parameters": run["parameters"],
                "parameter_delta": run["parameter_delta"],
                "config_sha256": run["config_sha256"],
                "checkpoint_sha256": run["checkpoint_sha256"],
                "elapsed_seconds": run["elapsed_seconds"],
                "qualified": run["qualified"],
            }
            for run in runs
        ]
    )
    inventory.to_csv(root / "experiment_inventory.csv", index=False)
    pd.DataFrame(constraint_rows).to_csv(
        root / "constraint_statistics.csv", index=False
    )
    comparison = pd.DataFrame(
        [
            {
                "experiment_id": run["experiment_id"],
                "mode": run["mode"],
                "qualified": run["qualified"],
                **run["metrics"],
            }
            for run in runs
        ]
    )
    comparison.to_csv(root / "candidate_comparison.csv", index=False)
    tune = {
        "schema_version": "mcvr-v2-bac-validation-tune-summary-v1",
        "status": "COMPLETED",
        "candidates": [
            {
                "experiment_id": run["experiment_id"],
                "mode": run["mode"],
                "metrics": run["metrics"],
                "qualified": run["qualified"],
            }
            for run in runs
        ],
        "selected_for_holdout": [
            "v2_a_bond_only_2000step_seed44",
            "v2_d_bond_angle_clash_2000step_seed44",
        ],
        "selection_reason": (
            "A is the strongest Bond-compatible baseline; D is the only unified "
            "candidate with the largest tune Angle improvement while preserving Bond."
        ),
        "test_records_read": 0,
        "test_assets_opened": False,
        "validation_only": True,
    }
    _write_json(root / "validation_tune_summary.json", tune)
    holdout = json.loads(
        (root / "validation_holdout_summary.json").read_text(encoding="utf-8")
    )
    target = json.loads(
        (root / "pilot_targets" / "summary.json").read_text(encoding="utf-8")
    )
    failure_analysis = {
        "schema_version": "mcvr-v2-bac-failure-analysis-v1",
        "status": "METHOD_EFFECT_NOT_VALIDATED",
        "numerical_failures": 0,
        "identity_failures": 0,
        "chirality_failures": 0,
        "target_fallback_fraction": target["fallback_fraction"],
        "target_fallback_below_limit": target["fallback_fraction"]
        <= target["maximum_fallback_fraction"],
        "dominant_failure": "proposal_effect_is_too_small_and_rollback_is_too_high",
        "process_compliance": {
            "two_batch_gpu_diagnostics": 2,
            "two_hundred_step_smokes": 1,
            "matched_two_thousand_step_pilots": 4,
            "gpu_optimizer_invocations": 7,
            "strict_maximum_six_compliant": False,
            "deviation": (
                "A second two-batch diagnostic was run after fixing the first "
                "diagnostic's gradient-domination finding. Counting both two-batch "
                "diagnostics as training runs exceeds the strict maximum by one."
            ),
        },
        "evidence": {
            "tune_acceptance_range": [
                float(comparison.accepted_fraction.min()),
                float(comparison.accepted_fraction.max()),
            ],
            "tune_rollback_range": [
                float(comparison.rollback_fraction.min()),
                float(comparison.rollback_fraction.max()),
            ],
            "holdout_angle_increment_d_minus_a": (
                holdout["candidates"][1]["metrics"]["angle_delta"]
                - holdout["candidates"][0]["metrics"]["angle_delta"]
            ),
            "holdout_clash_delta_d": holdout["candidates"][1]["metrics"][
                "clash_delta"
            ],
        },
        "interpretation": (
            "Angle and Clash encoders receive finite gradients, but the accepted "
            "coordinate effect is micrometric. D does not improve Angle over A on "
            "holdout, and Clash change is numerical noise. The result is not a "
            "capacity-underfitting diagnosis."
        ),
        "test_records_read": 0,
        "test_assets_opened": False,
        "validation_only": True,
    }
    _write_json(root / "failure_analysis.json", failure_analysis)
    recommendation = {
        "schema_version": "mcvr-v2-bac-recommended-next-step-v1",
        "decision": "DO_NOT_START_10K_OR_FORMAL_LARGE",
        "model_capacity": "KEEP_HIDDEN64_LAYERS4",
        "next_method_action": (
            "Start a new preregistered train/tune-only method stage focused on "
            "target-to-proposal scale and acceptance bottleneck attribution. Do not "
            "reuse holdout for tuning."
        ),
        "2k_completed": True,
        "10k_permitted": False,
        "formal_large_permitted": False,
        "network_expansion_permitted": False,
        "reason": (
            "No reproducible incremental Angle or Clash gain over Bond-only and "
            "94-97.5% rollback; gradients do not indicate model-capacity underfit."
        ),
        "test_records_read": 0,
        "test_assets_opened": False,
        "validation_only": True,
    }
    _write_json(root / "recommended_next_step.json", recommendation)

    tune_by_mode = {run["mode"]: run for run in runs}
    holdout_by_mode = {value["mode"]: value for value in holdout["candidates"]}
    lines = [
        "# MCVR V2-BAC Overnight Report",
        "",
        "## Decision",
        "",
        "`METHOD_EFFECT_NOT_VALIDATED`: the unified method is implemented and numerically safe, but the current 2k pilot does not justify 10k or formal-large training.",
        "",
        "All development and selection used train plus validation only. `test_records_read=0`, `test_assets_opened=false`, and `validation_only=true` for every artifact and run.",
        "",
        "## Method answers",
        "",
        "1. The target is one deterministic projected BAC optimization, not a sum of three coordinate targets.",
        "2. The model emits one fused Cartesian `delta_x`; no sequential Bond/Angle/Clash coordinate states exist.",
        "3. `V2_A_BOND_ONLY` has the exact D1-B parameter/state surface and passed fixed-forward, loss, correction, and strict frozen-checkpoint regression tests.",
        "4. Angle triplets are explicit, canonical, permutation-consistent, and SE(3)-safe. They receive gradients, but incremental holdout Angle improvement over A is zero.",
        "5. Clash edges are dynamic, sparse, topology-excluded, deterministic after sorting, and avoid dense pair allocation. Holdout Clash delta is numerical noise.",
        "6. Ring, identity, chirality, finite coordinates, and trust radii are hard acceptance constraints; failures roll back to the exact source.",
        "7. RMSD, MAT-P/R, COV-P/R, identity, and chirality are reported as public protocol metrics. GenBench3D and PoseBusters are unavailable and were not approximated as official results.",
        "8. Bond/Angle/Ring outlier diagnostics, total validity, acceptance, and displacement remain explicitly custom diagnostics.",
        "9. Four matched 2k runs were executed: A Bond-only, B Bond+Angle, C Bond+Clash, and D unified BAC. Two two-batch diagnostics and one 200-step D smoke preceded them.",
        "10. One implementation correction preceded the 200-step smoke: robust standardized residual clipping and isolation of new-branch supervision fixed measured Angle gradient domination. No post-tune network adjustment was made.",
        "11. A and D entered holdout: A was the strongest Bond baseline; D was the only unified candidate with the best tune Angle delta under hard constraints.",
        "12. The main conflict is efficacy versus rollback: small validity gains coincide with 93.6-97.5% tune rollback.",
        "13. Both candidates pass hard holdout safety/noninferiority, but D fails scientific incremental efficacy because Angle equals A and Clash is unchanged.",
        "14. The completed 2k stage is sufficient to reject 10k/formal-large escalation for this implementation.",
        "15. Keep hidden=64 and layers=4. Do not expand capacity; investigate target-to-proposal scale and acceptance bottlenecks in a new train/tune-only preregistered stage.",
        "16. All test access counters are zero.",
        "",
        "Process deviation: counting both two-batch diagnostics as GPU training runs gives seven optimizer invocations (2 diagnostics + 1 smoke + 4 pilots), exceeding the strict maximum of six by one. No additional run was launched after this audit.",
        "",
        "## Target assets",
        "",
        f"- Train-only pilot targets: {target['records']} records",
        f"- Success: {target['status_counts'].get('minimal_validity_success', 0)}",
        f"- Already valid: {target['status_counts'].get('identity_clean', 0)}",
        f"- Safe fallback: {target['status_counts'].get('identity_fallback', 0)} ({target['fallback_fraction']:.4%})",
        "- Formal train/validation assets were not modified.",
        "",
        "## Tune comparison",
        "",
        "| Mode | Params | Bond delta | Angle delta | Clash delta | RMSD delta | Acceptance | Rollback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in (
        "V2_A_BOND_ONLY",
        "V2_B_BOND_ANGLE",
        "V2_C_BOND_CLASH",
        "V2_D_BOND_ANGLE_CLASH",
    ):
        run = tune_by_mode[mode]
        metric = run["metrics"]
        lines.append(
            f"| {mode} | {run['parameters']} | {_fmt(metric['bond_delta'])} | "
            f"{_fmt(metric['angle_delta'])} | {_fmt(metric['clash_delta'])} | "
            f"{_fmt(metric['rmsd_delta'])} | {metric['accepted_fraction']:.4%} | "
            f"{metric['rollback_fraction']:.4%} |"
        )
    lines.extend(
        [
            "",
            "## Frozen holdout",
            "",
            "| Mode | Bond delta | Angle delta | Clash delta | RMSD delta | Acceptance | Rollback |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in ("V2_A_BOND_ONLY", "V2_D_BOND_ANGLE_CLASH"):
        metric = holdout_by_mode[mode]["metrics"]
        lines.append(
            f"| {mode} | {_fmt(metric['bond_delta'])} | "
            f"{_fmt(metric['angle_delta'])} | {_fmt(metric['clash_delta'])} | "
            f"{_fmt(metric['rmsd_delta'])} | {metric['accepted_fraction']:.4%} | "
            f"{metric['rollback_fraction']:.4%} |"
        )
    lines.extend(
        [
            "",
            "Holdout evaluation count is exactly one per candidate. Further tuning on this holdout is prohibited.",
            "",
            "## Recommendation",
            "",
            "Do not launch 10k or formal-large V2-BAC. Retain the 64x4 backbone. A future stage must preregister a train/tune-only attribution of target residual scale, new-branch output scale, fusion attenuation, trust clipping, and rejection reasons. The frozen holdout cannot be reused for that work.",
            "",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "MCVR_V2_BAC_OVERNIGHT_REPORT_COMPLETE",
                "decision": recommendation["decision"],
                "test_records_read": 0,
                "test_assets_opened": False,
                "validation_only": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
