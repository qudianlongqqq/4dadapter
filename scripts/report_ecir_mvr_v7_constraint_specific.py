#!/usr/bin/env python3
"""Build the frozen D1/V5-B/V7 paired development comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "diagnostics/ecir_mvr/v7_constraint_specific"
DEVELOPMENT_IDENTITY = (
    "3241a30c47a532dfbee4448bcd6556f0e5a67b752d17f85505e1f99f8a48ec51"
)
RUN_PATHS = {
    "D1": ROOT
    / "diagnostics/ecir_mvr/v2_bac_recovery/runs/d1_pilot_1000step_seed43018",
    "V5-B": ROOT
    / "diagnostics/ecir_mvr/v5_constraint_hybrid/runs/v5_b_pilot_seed43018",
    "V7": OUTPUT / "runs/v7_constraint_specific_pilot_seed43018",
}
METRICS = {
    "bond": "bond_outlier_rate",
    "angle": "angle_outlier_rate",
    "clash": "clash_penetration",
    "ring": "ring_bond_outlier_rate",
    "chirality": "chirality_error",
    "weighted_bac": "total_thresholded_validity_score",
    "rmsd": "aligned_RMSD",
    "displacement": "molecule_rms_displacement",
    "acceptance": "accepted",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _load_run(name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run = RUN_PATHS[name]
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "COMPLETED":
        raise RuntimeError(f"{name} run is not complete")
    checks = {
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "validation_only": True,
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"{name} isolation field {key} changed")
    if metadata.get("formal_large_run", False):
        raise RuntimeError(f"{name} unexpectedly reports formal-large execution")
    config = yaml.safe_load((run / "config.resolved.yaml").read_text(encoding="utf-8"))
    if config["data"]["development_manifest_identity_sha256"] != DEVELOPMENT_IDENTITY:
        raise RuntimeError(f"{name} development manifest identity changed")
    records = pd.read_csv(run / "development_per_record.csv")
    source = records.loc[records.method == "upstream"].copy()
    candidate = records.loc[records.method == "v2_bac_accepted"].copy()
    if len(source) != 1024 or len(candidate) != 1024:
        raise RuntimeError(f"{name} development record count changed")
    if source.sample_id.duplicated().any() or candidate.sample_id.duplicated().any():
        raise RuntimeError(f"{name} contains duplicate sample IDs")
    candidate["method"] = name
    return source, candidate, metadata


def _cohort_identity(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values("sample_id")[["sample_id", "molecule_id"]]
    payload = "\n".join(f"{row.sample_id}\t{row.molecule_id}" for row in ordered.itertuples())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assert_same_cohort(
    sources: dict[str, pd.DataFrame], candidates: dict[str, pd.DataFrame]
) -> str:
    reference_ids = set(sources["D1"].sample_id)
    reference = sources["D1"].set_index("sample_id").sort_index()
    if len(reference_ids) != 1024 or reference.molecule_id.nunique() != 512:
        raise RuntimeError("V7 comparison cohort size changed")
    columns = sorted(set(METRICS.values()) | {"molecule_id"})
    for name in RUN_PATHS:
        if set(sources[name].sample_id) != reference_ids:
            raise RuntimeError(f"{name} source sample identity differs from D1")
        if set(candidates[name].sample_id) != reference_ids:
            raise RuntimeError(f"{name} candidate sample identity differs from D1")
        current = sources[name].set_index("sample_id").sort_index()[columns]
        if not current.equals(reference[columns]):
            raise RuntimeError(f"{name} source metrics differ from D1")
    return _cohort_identity(sources["D1"])


def _paired(
    source: pd.DataFrame,
    d1: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    seed: int,
    draws: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    source = source.set_index("sample_id")
    d1 = d1.set_index("sample_id")
    candidate = candidate.set_index("sample_id")
    sample_ids = source.index.sort_values()
    subsets = {
        "all": sample_ids,
        "angle_active": sample_ids[
            (source.loc[sample_ids, "angle_outlier_rate"] > 0).to_numpy()
        ],
    }
    result = {}
    molecule_frames = []
    for subset_index, (subset, selected) in enumerate(subsets.items()):
        frame = pd.DataFrame(
            {
                metric: candidate.loc[selected, column].to_numpy(dtype=np.float64)
                - d1.loc[selected, column].to_numpy(dtype=np.float64)
                for metric, column in METRICS.items()
            },
            index=selected,
        )
        frame["molecule_id"] = source.loc[selected, "molecule_id"].to_numpy()
        molecules = frame.groupby("molecule_id", sort=True).mean(numeric_only=True)
        molecules.insert(0, "subset", subset)
        molecule_frames.append(molecules.reset_index())
        statistics = {}
        for metric_index, metric in enumerate(METRICS):
            values = molecules[metric].to_numpy(dtype=np.float64)
            rng = np.random.default_rng(seed + subset_index * 100 + metric_index)
            sampled = np.asarray(
                [rng.choice(values, len(values), replace=True).mean() for _ in range(draws)]
            )
            statistics[metric] = {
                "mean": float(values.mean()),
                "ci95_low": float(np.quantile(sampled, 0.025)),
                "ci95_high": float(np.quantile(sampled, 0.975)),
            }
        result[subset] = {
            "records": int(len(selected)),
            "molecules": int(molecules.shape[0]),
            "metrics": statistics,
        }
    return result, pd.concat(molecule_frames, ignore_index=True)


def _comparison_row(name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    metrics = metadata["metrics"]
    subsets = metadata["active_subsets"]
    training_seconds = metadata.get("training_elapsed_seconds")
    evaluation_seconds = metadata.get("evaluation_seconds")
    total_seconds = metadata.get("elapsed_seconds")
    if total_seconds is None:
        total_seconds = float(training_seconds or 0.0) + float(evaluation_seconds or 0.0)
    return {
        "method": name,
        "bond_delta": metrics["bond_delta"],
        "angle_delta": metrics["angle_delta"],
        "active_angle_delta": subsets["angle_active"]["metrics"]["angle_delta"]["mean"],
        "clash_delta": metrics["clash_delta"],
        "ring_delta": metrics["ring_delta"],
        "chirality_delta": subsets["all"]["metrics"]["chirality_delta"]["mean"],
        "weighted_bac_delta": subsets["all"]["metrics"]["weighted_bac_delta"]["mean"],
        "rmsd_delta": metrics["rmsd_delta"],
        "mat_p_delta": metrics["mat_p_delta"],
        "mat_r_delta": metrics["mat_r_delta"],
        "cov_p_delta": metrics["cov_p_delta"],
        "cov_r_delta": metrics["cov_r_delta"],
        "acceptance": metrics["accepted_fraction"],
        "rollback": metrics["rollback_fraction"],
        "mean_displacement": metrics["mean_displacement"],
        "training_seconds": float(training_seconds or 0.0),
        "evaluation_seconds": float(evaluation_seconds or 0.0),
        "total_runtime_seconds": float(total_seconds),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
    }


def main() -> None:
    sources = {}
    candidates = {}
    metadata = {}
    for name in RUN_PATHS:
        sources[name], candidates[name], metadata[name] = _load_run(name)
    identity = _assert_same_cohort(sources, candidates)
    paired = {}
    paired_molecules = []
    for index, name in enumerate(("V5-B", "V7")):
        result, molecules = _paired(
            sources["D1"],
            candidates["D1"],
            candidates[name],
            seed=47000 + index * 1000,
            draws=10_000,
        )
        paired[name] = result
        molecules.insert(0, "candidate", name)
        paired_molecules.append(molecules)
    v7_minus_v5_b, molecules = _paired(
        sources["D1"],
        candidates["V5-B"],
        candidates["V7"],
        seed=49000,
        draws=10_000,
    )
    molecules.insert(0, "candidate", "V7-minus-V5-B")
    paired_molecules.append(molecules)
    comparison = pd.DataFrame(
        [_comparison_row(name, metadata[name]) for name in RUN_PATHS]
    ).set_index("method")
    d1 = comparison.loc["D1"]
    v7 = comparison.loc["V7"]
    checks = {
        "active_angle_gain_ci95_high_lt_zero": paired["V7"]["angle_active"][
            "metrics"
        ]["angle"]["ci95_high"]
        < 0.0,
        "bond_degradation_vs_d1_le_0.005": v7.bond_delta - d1.bond_delta <= 0.005,
        "movement_ratio_vs_d1_le_1.1": v7.mean_displacement <= 1.1 * d1.mean_displacement,
        "acceptance_drop_vs_d1_le_0.05": d1.acceptance - v7.acceptance <= 0.05,
        "ring_non_regressed": v7.ring_delta <= d1.ring_delta,
        "chirality_non_regressed": v7.chirality_delta <= d1.chirality_delta,
        "rmsd_noninferior_0.0001": v7.rmsd_delta - d1.rmsd_delta <= 1.0e-4,
        "cov_p_non_regressed": v7.cov_p_delta >= d1.cov_p_delta,
        "cov_r_non_regressed": v7.cov_r_delta >= d1.cov_r_delta,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    supported = all(checks.values())
    report = {
        "schema_version": "mcvr-v7-constraint-specific-comparison-v1",
        "decision": (
            "V7_CONSTRAINT_SPECIFIC_SUPPORTED"
            if supported
            else "V7_CONSTRAINT_SPECIFIC_NOT_SUPPORTED"
        ),
        "ready_for_10k_or_formal_large": supported,
        "support_checks": checks,
        "comparison": comparison.reset_index().to_dict(orient="records"),
        "paired_candidate_minus_d1": paired,
        "paired_v7_minus_v5_b": v7_minus_v5_b,
        "angle_solver": metadata["V7"]["angle_solver"],
        "components": metadata["V7"]["components"],
        "development_manifest_identity_sha256": DEVELOPMENT_IDENTITY,
        "paired_cohort_identity_sha256": identity,
        "records": 1024,
        "molecules": 512,
        "same_sample_identity": True,
        "same_source_metrics": True,
        "same_seed": 43018,
        "bootstrap_draws": 10_000,
        "configuration_selected_from_results": False,
        "training_performed": False,
        "learned_fusion": False,
        "hidden_or_layer_change": False,
        "target_rematerialization": False,
        "formal_large_run": False,
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "validation_only": True,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparison.reset_index().to_csv(OUTPUT / "comparison.csv", index=False)
    pd.concat(
        [sources["D1"].assign(method="upstream"), *candidates.values()],
        ignore_index=True,
    ).to_csv(OUTPUT / "paired_per_record.csv", index=False)
    pd.concat(paired_molecules, ignore_index=True).to_csv(
        OUTPUT / "paired_per_molecule.csv", index=False
    )
    _write_json(OUTPUT / "summary.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
