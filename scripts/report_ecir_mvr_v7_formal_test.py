#!/usr/bin/env python3
"""Report the frozen Source/D1/V5-B/V7 one-time formal test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


METHODS = ("Source", "D1", "V5-B", "V7")
CANDIDATES = ("D1", "V5-B", "V7")
METRICS = {
    "bond": "bond_outlier_rate",
    "angle": "angle_outlier_rate",
    "clash": "clash_penetration",
    "ring": "ring_bond_outlier_rate",
    "chirality": "chirality_error",
    "weighted_bac": "total_thresholded_validity_score",
    "rmsd": "aligned_RMSD",
    "mat_p": "MAT_P",
    "mat_r": "MAT_R",
    "cov_p": "COV_P",
    "cov_r": "COV_R",
    "acceptance": "accepted",
    "displacement": "molecule_rms_displacement",
}
RECORD_METRICS = {
    name: column
    for name, column in METRICS.items()
    if name not in {"mat_p", "mat_r", "cov_p", "cov_r"}
}


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _bootstrap(values: pd.DataFrame, *, seed: int, draws: int) -> dict[str, Any]:
    matrix = values.to_numpy(dtype=np.float64)
    if not len(matrix):
        raise RuntimeError("formal-test bootstrap subset is empty")
    rng = np.random.default_rng(seed)
    means = np.empty((draws, matrix.shape[1]), dtype=np.float64)
    for start in range(0, draws, 100):
        count = min(100, draws - start)
        indices = rng.integers(0, len(matrix), size=(count, len(matrix)))
        means[start : start + count] = matrix[indices].mean(axis=1)
    return {
        metric: {
            "mean": float(matrix[:, index].mean()),
            "ci95_low": float(np.quantile(means[:, index], 0.025)),
            "ci95_high": float(np.quantile(means[:, index], 0.975)),
        }
        for index, metric in enumerate(values.columns)
    }


def _records(frame: pd.DataFrame, method: str, expected: int) -> pd.DataFrame:
    selected = frame.loc[frame.method == method].set_index("sample_id").sort_index()
    if len(selected) != expected or selected.index.nunique() != expected:
        raise RuntimeError(f"formal-test sample set changed: {method}")
    return selected


def _molecules(frame: pd.DataFrame, method: str, expected: int) -> pd.DataFrame:
    selected = frame.loc[
        (frame.method == method) & (frame.group == "all")
    ].set_index("molecule_id").sort_index()
    if len(selected) != expected or selected.index.nunique() != expected:
        raise RuntimeError(f"formal-test molecule set changed: {method}")
    return selected


def _active_angle(
    source_records: pd.DataFrame, method_records: pd.DataFrame
) -> float:
    active_ids = source_records.index[source_records.angle_outlier_rate > 0]
    active = method_records.loc[
        active_ids, ["molecule_id", "angle_outlier_rate"]
    ].groupby("molecule_id").angle_outlier_rate.mean()
    return float(active.mean())


def _absolute_summary(
    method: str,
    source_records: pd.DataFrame,
    method_records: pd.DataFrame,
    method_molecules: pd.DataFrame,
) -> dict[str, Any]:
    means = method_molecules[list(METRICS.values())].mean()
    return {
        "method": method,
        "bond": float(means.bond_outlier_rate),
        "angle": float(means.angle_outlier_rate),
        "active_angle": _active_angle(source_records, method_records),
        "clash": float(means.clash_penetration),
        "weighted_bac": float(means.total_thresholded_validity_score),
        "acceptance": float(means.accepted),
        "rollback": 1.0 - float(means.accepted),
        "mean_displacement": float(means.molecule_rms_displacement),
        "ring": float(means.ring_bond_outlier_rate),
        "chirality": float(means.chirality_error),
        "rmsd": float(means.aligned_RMSD),
        "mat_p": float(means.MAT_P),
        "mat_r": float(means.MAT_R),
        "cov_p": float(means.COV_P),
        "cov_r": float(means.COV_R),
    }


def _delta_summary(
    method: str,
    source_records: pd.DataFrame,
    method_records: pd.DataFrame,
    source_molecules: pd.DataFrame,
    method_molecules: pd.DataFrame,
) -> dict[str, Any]:
    delta = method_molecules[list(METRICS.values())] - source_molecules[
        list(METRICS.values())
    ]
    means = delta.mean()
    active_ids = source_records.index[source_records.angle_outlier_rate > 0]
    active = (
        method_records.loc[active_ids, ["molecule_id", "angle_outlier_rate"]]
        .assign(
            angle_delta=lambda frame: frame.angle_outlier_rate
            - source_records.loc[active_ids, "angle_outlier_rate"].to_numpy()
        )
        .groupby("molecule_id")
        .angle_delta.mean()
    )
    return {
        "method": method,
        "bond_delta": float(means.bond_outlier_rate),
        "angle_delta": float(means.angle_outlier_rate),
        "active_angle_delta": float(active.mean()),
        "clash_delta": float(means.clash_penetration),
        "weighted_bac_delta": float(means.total_thresholded_validity_score),
        "acceptance": float(method_molecules.accepted.mean()),
        "rollback": 1.0 - float(method_molecules.accepted.mean()),
        "mean_displacement": float(method_molecules.molecule_rms_displacement.mean()),
        "ring_delta": float(means.ring_bond_outlier_rate),
        "chirality_delta": float(means.chirality_error),
        "rmsd_delta": float(means.aligned_RMSD),
        "mat_p_delta": float(means.MAT_P),
        "mat_r_delta": float(means.MAT_R),
        "cov_p_delta": float(means.COV_P),
        "cov_r_delta": float(means.COV_R),
    }


def _paired(
    source_records: pd.DataFrame,
    baseline_records: pd.DataFrame,
    candidate_records: pd.DataFrame,
    baseline_molecules: pd.DataFrame,
    candidate_molecules: pd.DataFrame,
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    all_values = pd.DataFrame(
        {
            metric: candidate_molecules[column].to_numpy(dtype=np.float64)
            - baseline_molecules[column].to_numpy(dtype=np.float64)
            for metric, column in METRICS.items()
        },
        index=baseline_molecules.index,
    )
    active_ids = source_records.index[source_records.angle_outlier_rate > 0]
    active_values = pd.DataFrame(
        {
            metric: candidate_records.loc[active_ids, column].to_numpy(
                dtype=np.float64
            )
            - baseline_records.loc[active_ids, column].to_numpy(dtype=np.float64)
            for metric, column in RECORD_METRICS.items()
        },
        index=active_ids,
    )
    active_values["molecule_id"] = source_records.loc[
        active_ids, "molecule_id"
    ].to_numpy()
    active_values = active_values.groupby("molecule_id").mean(numeric_only=True)
    return {
        "all": {
            "molecules": len(all_values),
            "records": len(source_records),
            "metrics": _bootstrap(all_values, seed=seed, draws=draws),
        },
        "angle_active": {
            "molecules": len(active_values),
            "records": len(active_ids),
            "metrics": _bootstrap(active_values, seed=seed + 10_000, draws=draws),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    absolute_rows = []
    for row in report["absolute"]:
        absolute_rows.append(
            "| {method} | {bond:.6f} | {angle:.6f} | {active_angle:.6f} | "
            "{weighted_bac:.6f} | {acceptance:.4%} | {mean_displacement:.6f} | "
            "{rmsd:.6f} | {mat_p:.6f} | {mat_r:.6f} | {cov_p:.6f} | "
            "{cov_r:.6f} |".format(**row)
        )
    delta_rows = []
    for row in report["comparison"]:
        delta_rows.append(
            "| {method} | {bond_delta:.6f} | {angle_delta:.6f} | "
            "{active_angle_delta:.6f} | {weighted_bac_delta:.6f} | "
            "{acceptance:.4%} | {rollback:.4%} | {mean_displacement:.6f} | "
            "{rmsd_delta:.6f} |".format(**row)
        )
    active = report["paired"]["V7-minus-D1"]["angle_active"]["metrics"]["angle"]
    solver = report["angle_solver"]
    return "\n".join(
        [
            "# MCVR V7 Seed43 One-Time Formal Test",
            "",
            f"Status: `{report['status']}`",
            f"Seed: `{report['seed']}`",
            f"Molecules: `{report['molecules']}`",
            f"Records: `{report['records']}`",
            "",
            "## Absolute metrics",
            "",
            "| Method | Bond | Angle | Active Angle | Weighted BAC | Acceptance | "
            "Displacement | RMSD | MAT-P | MAT-R | COV-P | COV-R |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *absolute_rows,
            "",
            "## Delta from Source",
            "",
            "| Method | Bond | Angle | Active Angle | Weighted BAC | Acceptance | "
            "Rollback | Displacement | RMSD |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            *delta_rows,
            "",
            "## Paired V7 minus D1",
            "",
            f"Active-Angle mean: `{active['mean']:.8f}`",
            f"95% CI: `[{active['ci95_low']:.8f}, {active['ci95_high']:.8f}]`",
            "",
            "## V7 solver",
            "",
            f"- Calls: `{solver['calls']}`",
            f"- Failures: `{solver['solver_failure_count']}`",
            f"- Condition mean/max: `{solver['condition_number_mean']:.6f}` / "
            f"`{solver['condition_number_max']:.6f}`",
            f"- Effective rank: `{solver['effective_rank_mean']:.6f}`",
            f"- Truncated directions: `{solver['truncated_direction_count']}`",
            "",
            "References were opened only after all predictions completed. No Minimal "
            "Validity Target test asset was used. Results were not used to select or "
            "tune a checkpoint, method, threshold, step size, or configuration.",
            "",
            "formal_test_authorized=true",
            f"test_records_read={report['test_records_read']}",
            "test_assets_opened=true",
            "training_performed=false",
            "checkpoint_or_config_selected_from_test=false",
            "cohort_selection_from_test_metrics=false",
            "parameter_tuning_from_test=false",
            "",
        ]
    )


def build_report(
    output_dir: Path,
    *,
    seed: int,
    bootstrap_draws: int,
    expected_records: int = 23_882,
    expected_molecules: int = 100,
) -> dict[str, Any]:
    records = pd.read_csv(output_dir / "formal_test_per_record.csv")
    molecules = pd.read_csv(output_dir / "formal_test_per_molecule.csv")
    if set(records.method.unique()) != set(METHODS):
        raise RuntimeError("formal-test method set is incomplete")
    method_records = {
        method: _records(records, method, expected_records) for method in METHODS
    }
    method_molecules = {
        method: _molecules(molecules, method, expected_molecules)
        for method in METHODS
    }
    source_records = method_records["Source"]
    source_molecules = method_molecules["Source"]
    for method in METHODS:
        if not method_records[method].index.equals(source_records.index):
            raise RuntimeError(f"paired sample identity differs: {method}")
        if not method_molecules[method].index.equals(source_molecules.index):
            raise RuntimeError(f"paired molecule identity differs: {method}")
    absolute = [
        _absolute_summary(
            method,
            source_records,
            method_records[method],
            method_molecules[method],
        )
        for method in METHODS
    ]
    comparison = [
        _delta_summary(
            method,
            source_records,
            method_records[method],
            source_molecules,
            method_molecules[method],
        )
        for method in CANDIDATES
    ]
    paired = {
        "D1-minus-Source": _paired(
            source_records,
            source_records,
            method_records["D1"],
            source_molecules,
            method_molecules["D1"],
            seed=seed * 1000 - 1,
            draws=bootstrap_draws,
        ),
        "V5-B-minus-Source": _paired(
            source_records,
            source_records,
            method_records["V5-B"],
            source_molecules,
            method_molecules["V5-B"],
            seed=seed * 1000,
            draws=bootstrap_draws,
        ),
        "V7-minus-Source": _paired(
            source_records,
            source_records,
            method_records["V7"],
            source_molecules,
            method_molecules["V7"],
            seed=seed * 1000 + 3,
            draws=bootstrap_draws,
        ),
        "V7-minus-D1": _paired(
            source_records,
            method_records["D1"],
            method_records["V7"],
            method_molecules["D1"],
            method_molecules["V7"],
            seed=seed * 1000 + 1,
            draws=bootstrap_draws,
        ),
        "V7-minus-V5-B": _paired(
            source_records,
            method_records["V5-B"],
            method_records["V7"],
            method_molecules["V5-B"],
            method_molecules["V7"],
            seed=seed * 1000 + 2,
            draws=bootstrap_draws,
        ),
    }
    angle_solver = json.loads(
        (output_dir / "angle_solver_summary.json").read_text(encoding="utf-8")
    )
    components = json.loads(
        (output_dir / "component_summary.json").read_text(encoding="utf-8")
    )
    report = {
        "schema_version": "mcvr-v7-formal-test-summary-v1",
        "status": "COMPLETED",
        "formal_test_authorized": True,
        "one_time_evaluation": True,
        "seed": seed,
        "methods": list(METHODS),
        "molecules": expected_molecules,
        "records": expected_records,
        "absolute": absolute,
        "comparison": comparison,
        "paired": paired,
        "angle_solver": angle_solver,
        "components": components,
        "bootstrap_draws": bootstrap_draws,
        "same_sample_identity": True,
        "reference_policy": "post_prediction_metrics_only",
        "predictions_complete_before_reference_access": True,
        "minimal_validity_target_test_used": False,
        "test_records_read": expected_records,
        "test_assets_opened": True,
        "training_performed": False,
        "checkpoint_or_config_selected_from_test": False,
        "cohort_selection_from_test_metrics": False,
        "parameter_tuning_from_test": False,
        "method_selection_from_test": False,
    }
    _atomic_json(output_dir / "summary.json", report)
    (output_dir / "summary.md").write_text(_markdown(report), encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(43,), default=43)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_draws != 10_000:
        raise RuntimeError("formal-test bootstrap draws are frozen to 10000")
    report = build_report(
        args.output_dir.expanduser().resolve(),
        seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
