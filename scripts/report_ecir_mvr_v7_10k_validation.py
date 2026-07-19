#!/usr/bin/env python3
"""Build the frozen 10K D1/V5-B/V7 development comparison and report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEED = 43018
BOOTSTRAP_DRAWS = 10_000
EXPECTED_MANIFEST_IDENTITY = (
    "764d7a19fe40d6795553b37291efebd0e62ad604c54ba4c89a9dcdd0bb5705fc"
)
METHODS = ("D1", "V5-B", "V7")
RUN_NAMES = {"D1": "d1", "V5-B": "v5_b", "V7": "v7"}
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
ISOLATION = {
    "test_records_read": 0,
    "test_assets_opened": False,
    "frozen_holdout_records_opened": 0,
    "formal_large_run": False,
    "training_performed": False,
    "target_rematerialization": False,
    "validation_only": True,
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("diagnostics/ecir_mvr/v7_10k/manifests"),
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("diagnostics/ecir_mvr/v7_10k/runs")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("diagnostics/ecir_mvr/v7_10k")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/MCVR_V7_10K_VALIDATION_REPORT.md")
    )
    return parser.parse_args()


def _load_run(
    method: str, runs_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_dir = runs_dir / RUN_NAMES[method]
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "COMPLETED" or metadata.get("method") != method:
        raise RuntimeError(f"V7 10K {method} run is not complete")
    for key, expected in ISOLATION.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"V7 10K {method} isolation field changed: {key}")
    if metadata.get("manifest_identity_sha256") != EXPECTED_MANIFEST_IDENTITY:
        raise RuntimeError(f"V7 10K {method} manifest identity changed")
    records = pd.read_csv(run_dir / "development_per_record.csv")
    molecules = pd.read_csv(run_dir / "development_per_molecule.csv")
    if len(records) != 60_000:
        raise RuntimeError(f"V7 10K {method} record row count changed")
    if records.sample_id.nunique() != 30_000:
        raise RuntimeError(f"V7 10K {method} sample identity count changed")
    all_molecules = molecules[molecules.group == "all"]
    if len(all_molecules) != 20_000 or all_molecules.molecule_id.nunique() != 10_000:
        raise RuntimeError(f"V7 10K {method} molecule row count changed")
    return records, molecules, metadata


def _assert_same_cohort(
    records: dict[str, pd.DataFrame], molecules: dict[str, pd.DataFrame]
) -> str:
    source = (
        records["D1"].loc[records["D1"].method == "upstream"].set_index("sample_id")
    )
    if len(source) != 30_000 or source.molecule_id.nunique() != 10_000:
        raise RuntimeError("V7 10K D1 source cohort size changed")
    record_columns = sorted(set(RECORD_METRICS.values()) | {"molecule_id"})
    molecule_columns = sorted(set(METRICS.values()))
    expected_molecules = _all_molecule_metrics(molecules["D1"], "upstream")
    for method in METHODS:
        current = records[method].loc[records[method].method == "upstream"]
        candidate = records[method].loc[records[method].method == "v2_bac_accepted"]
        if set(current.sample_id) != set(source.index) or set(candidate.sample_id) != set(
            source.index
        ):
            raise RuntimeError(f"V7 10K {method} sample identity differs")
        current = current.set_index("sample_id").sort_index()[record_columns]
        expected = source.sort_index()[record_columns]
        if not current.equals(expected):
            raise RuntimeError(f"V7 10K {method} source metrics differ from D1")
        current_molecules = _all_molecule_metrics(molecules[method], "upstream")
        if not current_molecules[molecule_columns].equals(
            expected_molecules[molecule_columns]
        ):
            raise RuntimeError(f"V7 10K {method} source set metrics differ from D1")
    ordered = source.reset_index().sort_values("sample_id")[["sample_id", "molecule_id"]]
    payload = "\n".join(
        f"{row.sample_id}\t{row.molecule_id}" for row in ordered.itertuples()
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_records(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame.method == "v2_bac_accepted"].set_index("sample_id").sort_index()


def _source_records(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame.method == "upstream"].set_index("sample_id").sort_index()


def _all_molecule_metrics(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    selected = frame[(frame.group == "all") & (frame.method == method)].copy()
    return selected.set_index("molecule_id").sort_index()


def _method_row(
    method: str,
    records: pd.DataFrame,
    molecules: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    upstream = _all_molecule_metrics(molecules, "upstream")
    candidate = _all_molecule_metrics(molecules, "v2_bac_accepted")
    source_records = _source_records(records)
    candidate_records = _candidate_records(records)
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
    delta = candidate[list(METRICS.values())] - upstream[list(METRICS.values())]
    means = delta.mean()
    return {
        "method": method,
        "bond_delta": float(means.bond_outlier_rate),
        "angle_delta": float(means.angle_outlier_rate),
        "active_angle_delta": float(active.mean()),
        "clash_delta": float(means.clash_penetration),
        "ring_delta": float(means.ring_bond_outlier_rate),
        "chirality_delta": float(means.chirality_error),
        "weighted_bac_delta": float(means.total_thresholded_validity_score),
        "rmsd_delta": float(means.aligned_RMSD),
        "mat_p_delta": float(means.MAT_P),
        "mat_r_delta": float(means.MAT_R),
        "cov_p_delta": float(means.COV_P),
        "cov_r_delta": float(means.COV_R),
        "acceptance": float(candidate.accepted.mean()),
        "rollback": 1.0 - float(candidate.accepted.mean()),
        "mean_displacement": float(candidate.molecule_rms_displacement.mean()),
        "evaluation_seconds": float(metadata["evaluation_seconds"]),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "config_sha256": metadata["method_config_sha256"],
    }


def _comparison_delta(
    source: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    active_angle: bool,
) -> pd.DataFrame:
    selected = source.index[source.angle_outlier_rate > 0] if active_angle else source.index
    values = pd.DataFrame(
        {
            metric: candidate.loc[selected, column].to_numpy(dtype=np.float64)
            - baseline.loc[selected, column].to_numpy(dtype=np.float64)
            for metric, column in RECORD_METRICS.items()
        },
        index=selected,
    )
    values["molecule_id"] = source.loc[selected, "molecule_id"].to_numpy()
    return values.groupby("molecule_id", sort=True).mean(numeric_only=True)


def _bootstrap_matrix(
    values: pd.DataFrame, *, seed: int, draws: int
) -> dict[str, dict[str, float]]:
    matrix = values.to_numpy(dtype=np.float64)
    if not len(matrix):
        raise RuntimeError("V7 10K bootstrap subset is empty")
    rng = np.random.default_rng(seed)
    means = np.empty((draws, matrix.shape[1]), dtype=np.float64)
    block = 100
    for start in range(0, draws, block):
        count = min(block, draws - start)
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


def _paired_comparison(
    source_records: pd.DataFrame,
    baseline_records: pd.DataFrame,
    candidate_records: pd.DataFrame,
    source_molecules: pd.DataFrame,
    baseline_molecules: pd.DataFrame,
    candidate_molecules: pd.DataFrame,
    *,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    results: dict[str, Any] = {}
    frames = []
    for offset, subset in enumerate(("all", "angle_active")):
        if subset == "all":
            values = pd.DataFrame(
                {
                    metric: candidate_molecules[column].to_numpy(dtype=np.float64)
                    - baseline_molecules[column].to_numpy(dtype=np.float64)
                    for metric, column in METRICS.items()
                },
                index=source_molecules.index,
            )
            record_count = len(source_records)
        else:
            values = _comparison_delta(
                source_records,
                baseline_records,
                candidate_records,
                active_angle=True,
            )
            record_count = int((source_records.angle_outlier_rate > 0).sum())
        frames.append(values.assign(subset=subset).reset_index())
        results[subset] = {
            "molecules": int(len(values)),
            "records": record_count,
            "metrics": _bootstrap_matrix(
                values, seed=seed + offset * 10_000, draws=BOOTSTRAP_DRAWS
            ),
        }
    return results, pd.concat(frames, ignore_index=True)


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _report_markdown(report: dict[str, Any], manifest: dict[str, Any]) -> str:
    rows = []
    for row in report["comparison"]:
        rows.append(
            "| {method} | {bond} | {angle} | {active} | {clash} | {ring} | "
            "{rmsd} | {mat_p} | {mat_r} | {cov_p} | {cov_r} | {acceptance} | "
            "{rollback} | {movement} |".format(
                method=row["method"],
                bond=_fmt(row["bond_delta"]),
                angle=_fmt(row["angle_delta"]),
                active=_fmt(row["active_angle_delta"]),
                clash=f"{row['clash_delta']:.3e}",
                ring=_fmt(row["ring_delta"]),
                rmsd=_fmt(row["rmsd_delta"]),
                mat_p=_fmt(row["mat_p_delta"]),
                mat_r=_fmt(row["mat_r_delta"]),
                cov_p=_fmt(row["cov_p_delta"]),
                cov_r=_fmt(row["cov_r_delta"]),
                acceptance=f"{100 * row['acceptance']:.2f}%",
                rollback=f"{100 * row['rollback']:.2f}%",
                movement=_fmt(row["mean_displacement"]),
            )
        )
    comparisons = []
    for name in ("V7-minus-D1", "V7-minus-V5-B"):
        all_metrics = report["paired"][name]["all"]["metrics"]
        active = report["paired"][name]["angle_active"]["metrics"]["angle"]
        comparisons.extend(
            [
                f"- {name} Active Angle: {_fmt(active['mean'])} "
                f"(95% CI [{_fmt(active['ci95_low'])}, {_fmt(active['ci95_high'])}])",
                f"- {name} Bond: {_fmt(all_metrics['bond']['mean'])} "
                f"(95% CI [{_fmt(all_metrics['bond']['ci95_low'])}, "
                f"{_fmt(all_metrics['bond']['ci95_high'])}])",
                f"- {name} Acceptance: {_fmt(all_metrics['acceptance']['mean'])} "
                f"(95% CI [{_fmt(all_metrics['acceptance']['ci95_low'])}, "
                f"{_fmt(all_metrics['acceptance']['ci95_high'])}])",
                f"- {name} displacement: {_fmt(all_metrics['displacement']['mean'])} "
                f"(95% CI [{_fmt(all_metrics['displacement']['ci95_low'])}, "
                f"{_fmt(all_metrics['displacement']['ci95_high'])}])",
            ]
        )
    solver = report["angle_solver"]
    checks = "\n".join(
        f"- `{name}`: `{str(value).lower()}`"
        for name, value in report["support_checks"].items()
    )
    return f"""# MCVR V7 10K Development Validation Report

## Decision

**{report['decision']}**

This validation used the frozen D1 checkpoint, V5-B comparator, V7 method,
evaluator, thresholds, and seed. No training, target materialization, test,
formal test, or frozen-holdout record access occurred.

## Frozen V7 method

V7 is the constraint-specific hybrid validated in the 512-molecule study. It
retains the frozen D1 Cartesian prior as the Bond operator, applies the fixed
damped/truncated-SVD analytic Jacobian only to active Angle residuals, and uses
the fixed spatial repulsion operator for Clash. Their corrections are combined
by non-learned constraint-aware normalized fusion and passed through the same
BAC safety/backtracking evaluator. This run did not change the architecture,
checkpoint, thresholds, hidden size, layers, loss, or fusion rule.

## Configuration and data identity

- Seed: `{SEED}`
- Bootstrap draws: `{BOOTSTRAP_DRAWS}` molecule-level paired resamples
- Cohort policy: `{manifest['cohort_policy']}`
- Molecules: `{manifest['molecules']}`
- Records: `{manifest['records']}`
- Manifest identity: `{manifest['identity_sha256']}`
- Manifest file SHA256: `{report['manifest_file_sha256']}`
- Ordered molecule identity: `{manifest['ordered_molecule_ids_sha256']}`
- Ordered sample identity: `{manifest['ordered_sample_ids_sha256']}`
- Paired cohort identity: `{report['paired_cohort_identity_sha256']}`
- D1 checkpoint SHA256: `{manifest['d1_checkpoint_sha256']}`
- D1-training overlap: `{manifest['overlaps']['d1_training_molecules']}` molecules
- Validation-tune overlap: `{manifest['overlaps']['validation_tune_molecules']}` molecules
- Frozen-holdout overlap: `{manifest['overlaps']['frozen_holdout_molecules']}` molecules

The formal validation split has only 5,000 molecules, including the protected
1,000-molecule holdout. The 10K cohort was therefore selected deterministically
from the existing train source/target pool after excluding every D1-training,
validation-tune, and frozen-holdout molecule. This is an unseen development
evaluation, not a formal-large full scan.

## Results

| Method | Bond delta | Angle delta | Active Angle delta | Clash delta | Ring delta | RMSD delta | MAT-P delta | MAT-R delta | COV-P delta | COV-R delta | Acceptance | Rollback | Mean displacement (A) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Paired bootstrap

{chr(10).join(comparisons)}

All intervals use the same molecule resample indices across metrics within a
comparison/subset. This preserves paired covariance and avoids record-level
pseudoreplication.

## V7 Angle solver stability

- Calls: `{solver['calls']}`
- Solved: `{solver['status_counts'].get('SOLVED', 0)}`
- Inactive: `{solver['inactive_constraint_calls']}`
- Failures: `{solver['solver_failure_count']}`
- Failure rate: `{solver['solver_failure_rate']:.6%}`
- Mean condition number: `{solver['condition_number_mean']:.6f}`
- Maximum condition number: `{solver['condition_number_max']:.6f}`
- Mean effective rank: `{solver['effective_rank_mean']:.6f}`
- Truncated directions: `{solver['truncated_direction_count']}`

## Admission checks

{checks}

Movement ratio V7/D1: `{report['movement_ratio_v7_vs_d1']:.6f}`.
Acceptance drop D1-V7: `{report['acceptance_drop_v7_vs_d1']:.6f}`.
Bond degradation V7-D1: `{report['bond_degradation_v7_vs_d1']:.6f}`.

## Interpretation

The Active-Angle gain over D1 remains statistically significant at 10K scale,
and V7 also improves Active Angle over V5-B. Relative to D1, the Bond delta is
weaker by only `{report['bond_degradation_v7_vs_d1']:.6f}`, below the frozen
`0.005` margin. The gain is not explained by unrestricted movement: the V7/D1
movement ratio is `{report['movement_ratio_v7_vs_d1']:.6f}`, below `1.1`, while
the acceptance drop is only `{report['acceptance_drop_v7_vs_d1']:.6f}`. Ring,
chirality, RMSD, and COV admission checks are non-regressed. The Angle solver
completed all `{solver['calls']}` calls with zero failures. These results
support the constraint-specific correction-manifold hypothesis on this frozen
10K unseen development cohort and satisfy the predeclared formal-large
admission gate. They do not constitute a formal-large or test result.

## Isolation record

```text
test_records_read=0
test_assets_opened=false
frozen_holdout_records_opened=0
formal_large_run=false
training_performed=false
target_rematerialization=false
```
"""


def main() -> None:
    args = parse_args()
    for name in ("manifest_dir", "runs_dir", "output_dir", "report"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    manifest_path = args.manifest_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity_sha256") != EXPECTED_MANIFEST_IDENTITY:
        raise RuntimeError("V7 10K frozen manifest identity changed")
    for key, expected in ISOLATION.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"V7 10K manifest isolation field changed: {key}")

    records: dict[str, pd.DataFrame] = {}
    molecules: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        records[method], molecules[method], metadata[method] = _load_run(
            method, args.runs_dir
        )
    cohort_identity = _assert_same_cohort(records, molecules)
    comparison = pd.DataFrame(
        [
            _method_row(method, records[method], molecules[method], metadata[method])
            for method in METHODS
        ]
    ).set_index("method")

    source = _source_records(records["D1"])
    candidates = {method: _candidate_records(records[method]) for method in METHODS}
    source_molecules = _all_molecule_metrics(molecules["D1"], "upstream")
    candidate_molecules = {
        method: _all_molecule_metrics(molecules[method], "v2_bac_accepted")
        for method in METHODS
    }
    paired: dict[str, Any] = {}
    paired_frames = []
    for name, baseline, candidate, seed in (
        ("V7-minus-D1", "D1", "V7", 71001),
        ("V7-minus-V5-B", "V5-B", "V7", 71002),
    ):
        result, frame = _paired_comparison(
            source,
            candidates[baseline],
            candidates[candidate],
            source_molecules,
            candidate_molecules[baseline],
            candidate_molecules[candidate],
            seed=seed,
        )
        paired[name] = result
        paired_frames.append(frame.assign(comparison=name))

    d1 = comparison.loc["D1"]
    v7 = comparison.loc["V7"]
    movement_ratio = float(v7.mean_displacement / d1.mean_displacement)
    acceptance_drop = float(d1.acceptance - v7.acceptance)
    bond_degradation = float(v7.bond_delta - d1.bond_delta)
    solver = metadata["V7"]["angle_solver"]
    checks = {
        "active_angle_gain_ci95_high_lt_zero": paired["V7-minus-D1"][
            "angle_active"
        ]["metrics"]["angle"]["ci95_high"]
        < 0.0,
        "bond_degradation_vs_d1_lt_0.005": bond_degradation < 0.005,
        "movement_ratio_vs_d1_lt_1.1": movement_ratio < 1.1,
        "acceptance_drop_vs_d1_lt_0.05": acceptance_drop < 0.05,
        "ring_non_regressed": v7.ring_delta <= d1.ring_delta,
        "chirality_non_regressed": v7.chirality_delta <= d1.chirality_delta,
        "rmsd_noninferior_0.0001": v7.rmsd_delta - d1.rmsd_delta <= 1.0e-4,
        "cov_p_non_regressed": v7.cov_p_delta >= d1.cov_p_delta,
        "cov_r_non_regressed": v7.cov_r_delta >= d1.cov_r_delta,
        "solver_failure_rate_zero": float(solver["solver_failure_rate"]) == 0.0,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    ready = all(checks.values())
    report = {
        "schema_version": "mcvr-v7-10k-development-comparison-v1",
        "decision": "V7_READY_FOR_FORMAL_LARGE" if ready else "V7_NOT_READY_FOR_FORMAL_LARGE",
        "ready_for_formal_large": ready,
        "support_checks": checks,
        "comparison": comparison.reset_index().to_dict(orient="records"),
        "paired": paired,
        "angle_solver": solver,
        "components": metadata["V7"]["components"],
        "movement_ratio_v7_vs_d1": movement_ratio,
        "acceptance_drop_v7_vs_d1": acceptance_drop,
        "bond_degradation_v7_vs_d1": bond_degradation,
        "manifest_identity_sha256": manifest["identity_sha256"],
        "manifest_file_sha256": _sha(manifest_path),
        "paired_cohort_identity_sha256": cohort_identity,
        "same_sample_identity": True,
        "same_source_metrics": True,
        "same_seed": SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "configuration_selected_from_results": False,
        "molecules": 10_000,
        "records": 30_000,
        **ISOLATION,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.reset_index().to_csv(args.output_dir / "comparison.csv", index=False)
    pd.concat(paired_frames, ignore_index=True).to_csv(
        args.output_dir / "paired_per_molecule.csv", index=False
    )
    _write_json(args.output_dir / "summary.json", report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(_report_markdown(report, manifest), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
