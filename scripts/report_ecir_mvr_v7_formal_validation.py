#!/usr/bin/env python3
"""Summarize the frozen D1/V5-B/V7 formal-large validation comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


METHODS = ("D1", "V5-B", "V7")
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
        raise RuntimeError("formal validation bootstrap subset is empty")
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


def _records(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    selected = frame.loc[frame.method == method].set_index("sample_id").sort_index()
    if len(selected) != 10_000 or selected.index.nunique() != 10_000:
        raise RuntimeError(f"formal validation sample set changed: {method}")
    return selected


def _molecules(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    selected = frame.loc[
        (frame.method == method) & (frame.group == "all")
    ].set_index("molecule_id").sort_index()
    if len(selected) != 5_000 or selected.index.nunique() != 5_000:
        raise RuntimeError(f"formal validation molecule set changed: {method}")
    return selected


def _method_summary(
    method: str,
    source_records: pd.DataFrame,
    candidate_records: pd.DataFrame,
    source_molecules: pd.DataFrame,
    candidate_molecules: pd.DataFrame,
) -> dict[str, Any]:
    delta = candidate_molecules[list(METRICS.values())] - source_molecules[
        list(METRICS.values())
    ]
    means = delta.mean()
    active_ids = source_records.index[source_records.angle_outlier_rate > 0]
    active = (
        candidate_records.loc[active_ids, ["molecule_id", "angle_outlier_rate"]]
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
        "acceptance": float(candidate_molecules.accepted.mean()),
        "rollback": 1.0 - float(candidate_molecules.accepted.mean()),
        "mean_displacement": float(candidate_molecules.molecule_rms_displacement.mean()),
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
    result: dict[str, Any] = {}
    all_values = pd.DataFrame(
        {
            metric: candidate_molecules[column].to_numpy(dtype=np.float64)
            - baseline_molecules[column].to_numpy(dtype=np.float64)
            for metric, column in METRICS.items()
        },
        index=baseline_molecules.index,
    )
    result["all"] = {
        "molecules": len(all_values),
        "records": len(source_records),
        "metrics": _bootstrap(all_values, seed=seed, draws=draws),
    }
    active_ids = source_records.index[source_records.angle_outlier_rate > 0]
    active_values = pd.DataFrame(
        {
            metric: candidate_records.loc[active_ids, column].to_numpy(dtype=np.float64)
            - baseline_records.loc[active_ids, column].to_numpy(dtype=np.float64)
            for metric, column in RECORD_METRICS.items()
        },
        index=active_ids,
    )
    active_values["molecule_id"] = source_records.loc[
        active_ids, "molecule_id"
    ].to_numpy()
    active_values = active_values.groupby("molecule_id").mean(numeric_only=True)
    result["angle_active"] = {
        "molecules": len(active_values),
        "records": len(active_ids),
        "metrics": _bootstrap(active_values, seed=seed + 10_000, draws=draws),
    }
    return result


def _markdown(report: dict[str, Any]) -> str:
    header = (
        "| Method | Bond | Angle | Active Angle | Clash | Weighted BAC | "
        "Acceptance | Rollback | Displacement | Ring | Chirality | RMSD | "
        "MAT-P | MAT-R | COV-P | COV-R |"
    )
    divider = "|" + "---|" * 17
    rows = []
    for row in report["comparison"]:
        rows.append(
            "| {method} | {bond_delta:.6f} | {angle_delta:.6f} | "
            "{active_angle_delta:.6f} | {clash_delta:.3e} | "
            "{weighted_bac_delta:.6f} | {acceptance:.4%} | {rollback:.4%} | "
            "{mean_displacement:.6f} | {ring_delta:.6f} | {chirality_delta:.6f} | "
            "{rmsd_delta:.6f} | {mat_p_delta:.6f} | {mat_r_delta:.6f} | "
            "{cov_p_delta:.6f} | {cov_r_delta:.6f} |".format(**row)
        )
    solver = report["angle_solver"]
    return "\n".join(
        [
            "# MCVR V7 Formal-Large Validation",
            "",
            f"Seed: `{report['seed']}`",
            f"Molecules: `{report['molecules']}`",
            f"Records: `{report['records']}`",
            "",
            header,
            divider,
            *rows,
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
            "No formal test or frozen holdout was read.",
            "",
            "test_records_read=0",
            "test_assets_opened=false",
            "frozen_holdout_records_opened=0",
            "",
        ]
    )


def build_report(output_dir: Path, *, seed: int, bootstrap_draws: int) -> dict[str, Any]:
    records = pd.read_csv(output_dir / "formal_validation_per_record.csv")
    molecules = pd.read_csv(output_dir / "formal_validation_per_molecule.csv")
    if set(records.method.unique()) != {"Source", *METHODS}:
        raise RuntimeError("formal validation method set is incomplete")
    source_records = _records(records, "Source")
    source_molecules = _molecules(molecules, "Source")
    candidates_records = {method: _records(records, method) for method in METHODS}
    candidates_molecules = {method: _molecules(molecules, method) for method in METHODS}
    for method in METHODS:
        if not candidates_records[method].index.equals(source_records.index):
            raise RuntimeError(f"paired sample identity differs: {method}")
        if not candidates_molecules[method].index.equals(source_molecules.index):
            raise RuntimeError(f"paired molecule identity differs: {method}")
    comparison = [
        _method_summary(
            method,
            source_records,
            candidates_records[method],
            source_molecules,
            candidates_molecules[method],
        )
        for method in METHODS
    ]
    paired = {
        "V7-minus-D1": _paired(
            source_records,
            candidates_records["D1"],
            candidates_records["V7"],
            candidates_molecules["D1"],
            candidates_molecules["V7"],
            seed=seed * 1000 + 1,
            draws=bootstrap_draws,
        ),
        "V7-minus-V5-B": _paired(
            source_records,
            candidates_records["V5-B"],
            candidates_records["V7"],
            candidates_molecules["V5-B"],
            candidates_molecules["V7"],
            seed=seed * 1000 + 2,
            draws=bootstrap_draws,
        ),
    }
    solver = json.loads(
        (output_dir / "angle_solver_summary.json").read_text(encoding="utf-8")
    )
    components = json.loads(
        (output_dir / "component_summary.json").read_text(encoding="utf-8")
    )
    report = {
        "schema_version": "mcvr-v7-formal-validation-summary-v1",
        "seed": seed,
        "methods": list(METHODS),
        "molecules": 5_000,
        "records": 10_000,
        "comparison": comparison,
        "paired": paired,
        "angle_solver": solver,
        "components": components,
        "bootstrap_draws": bootstrap_draws,
        "same_sample_identity": True,
        "same_source_metrics": True,
        "configuration_selected_from_results": False,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "formal_test_run": False,
        "training_performed": False,
        "validation_only": True,
    }
    _atomic_json(output_dir / "method_summary.json", report)
    (output_dir / "formal_validation_report.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43), required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_draws < 1:
        raise RuntimeError("bootstrap draws must be positive")
    report = build_report(
        args.output_dir.expanduser().resolve(),
        seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
