#!/usr/bin/env python3
"""Build the frozen D1/A/B paired comparison for MCVR V5."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "diagnostics/ecir_mvr/v5_constraint_hybrid/runs"
D1_RUN = ROOT / "diagnostics/ecir_mvr/v2_bac_recovery/runs/d1_pilot_1000step_seed43018"
OUTPUT = ROOT / "diagnostics/ecir_mvr/v5_constraint_hybrid"
RUN_PATHS = {
    "D1": D1_RUN,
    "A": RUNS / "v5_a_pilot_1000step_seed43018",
    "B": RUNS / "v5_b_pilot_seed43018",
}
METRICS = {
    "bond": "bond_outlier_rate",
    "angle": "angle_outlier_rate",
    "clash": "clash_penetration",
    "ring": "ring_bond_outlier_rate",
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


def _load_candidate(name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    run = RUN_PATHS[name]
    records = pd.read_csv(run / "development_per_record.csv")
    candidate = records[records.method == "v2_bac_accepted"].copy()
    candidate["method"] = name
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    return candidate, metadata


def _paired(
    source: pd.DataFrame,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    seed: int,
    draws: int = 1000,
) -> dict[str, Any]:
    source = source.set_index("sample_id")
    baseline = baseline.set_index("sample_id")
    candidate = candidate.set_index("sample_id")
    common = source.index.intersection(baseline.index).intersection(candidate.index)
    subsets = {
        "all": common,
        "angle_active": common[(source.loc[common].angle_outlier_rate > 0).to_numpy()],
    }
    result = {}
    for subset_index, (subset, selected) in enumerate(subsets.items()):
        frame = pd.DataFrame(
            {
                name: candidate.loc[selected, column].to_numpy(dtype=np.float64)
                - baseline.loc[selected, column].to_numpy(dtype=np.float64)
                for name, column in METRICS.items()
            },
            index=selected,
        )
        frame["molecule_id"] = source.loc[selected].molecule_id.to_numpy()
        molecules = frame.groupby("molecule_id").mean(numeric_only=True)
        statistics = {}
        for metric_index, name in enumerate(METRICS):
            values = molecules[name].to_numpy(dtype=np.float64)
            rng = np.random.default_rng(seed + subset_index * 100 + metric_index)
            sampled = np.asarray(
                [rng.choice(values, len(values), replace=True).mean() for _ in range(draws)]
            )
            statistics[name] = {
                "mean": float(values.mean()),
                "ci95_low": float(np.quantile(sampled, 0.025)),
                "ci95_high": float(np.quantile(sampled, 0.975)),
            }
        result[subset] = {
            "records": int(len(selected)),
            "molecules": int(molecules.shape[0]),
            "metrics": statistics,
        }
    return result


def main() -> None:
    d1_records, d1_meta = _load_candidate("D1")
    source_all = pd.read_csv(D1_RUN / "development_per_record.csv")
    source = source_all[source_all.method == "upstream"].copy()
    candidates = {"D1": d1_records}
    metadata = {"D1": d1_meta}
    for name in ("A", "B"):
        candidates[name], metadata[name] = _load_candidate(name)
    expected = set(d1_records.sample_id)
    if len(expected) != 1024:
        raise RuntimeError("V5 D1 comparison record count changed")
    for name, frame in candidates.items():
        if set(frame.sample_id) != expected:
            raise RuntimeError(f"V5 {name} sample identity differs from D1")
    paired = {
        name: _paired(source, d1_records, candidates[name], seed=45000 + index * 1000)
        for index, name in enumerate(("A", "B"))
    }
    table = []
    for name in ("D1", "A", "B"):
        metrics = metadata[name]["metrics"]
        active = metadata[name]["active_subsets"]["angle_active"]["metrics"]
        table.append(
            {
                "method": name,
                "bond_delta": metrics["bond_delta"],
                "angle_delta": metrics["angle_delta"],
                "active_angle_delta": active["angle_delta"]["mean"],
                "clash_delta": metrics["clash_delta"],
                "ring_delta": metrics["ring_delta"],
                "rmsd_delta": metrics["rmsd_delta"],
                "mat_p_delta": metrics["mat_p_delta"],
                "mat_r_delta": metrics["mat_r_delta"],
                "cov_p_delta": metrics["cov_p_delta"],
                "cov_r_delta": metrics["cov_r_delta"],
                "acceptance": metrics["accepted_fraction"],
                "rollback": metrics["rollback_fraction"],
                "mean_displacement": metrics["mean_displacement"],
                "active_angle_per_displacement": abs(active["angle_delta"]["mean"])
                / max(metrics["mean_displacement"], 1.0e-30),
            }
        )
    comparison = pd.DataFrame(table).set_index("method")
    d1 = comparison.loc["D1"]
    decisions = {}
    for name in ("A", "B"):
        row = comparison.loc[name]
        checks = {
            "active_angle_gain_ci95_high_lt_zero": paired[name]["angle_active"]["metrics"]["angle"][
                "ci95_high"
            ]
            < 0.0,
            "bond_degradation_vs_d1_le_0.005": row.bond_delta - d1.bond_delta <= 0.005,
            "acceptance_drop_vs_d1_le_0.05": d1.acceptance - row.acceptance <= 0.05,
            "movement_ratio_vs_d1_le_1.1": row.mean_displacement <= 1.1 * d1.mean_displacement,
            "ring_non_regressed": row.ring_delta <= d1.ring_delta,
            "rmsd_noninferior_0.0001": row.rmsd_delta - d1.rmsd_delta <= 1.0e-4,
            "cov_p_non_regressed": row.cov_p_delta >= d1.cov_p_delta,
            "cov_r_non_regressed": row.cov_r_delta >= d1.cov_r_delta,
        }
        checks = {key: bool(value) for key, value in checks.items()}
        decisions[name] = {"supported": all(checks.values()), "checks": checks}
    strict_supported = [name for name in ("A", "B") if decisions[name]["supported"]]
    report = {
        "schema_version": "mcvr-v5-constraint-hybrid-comparison-v1",
        "decision": (
            "V5_CONSTRAINT_HYBRID_SUPPORTED"
            if strict_supported
            else "V5_CONSTRAINT_HYBRID_NOT_YET_SUPPORTED"
        ),
        "strict_supported_prototypes": strict_supported,
        "paper_direction": "B_NEURAL_PRIOR_PLUS_JACOBIAN",
        "paper_direction_ready_for_scale": "B" in strict_supported,
        "comparison": comparison.reset_index().to_dict(orient="records"),
        "paired_candidate_minus_d1": paired,
        "support_checks": decisions,
        "solver_b": metadata["B"]["solver"],
        "records": 1024,
        "molecules": 512,
        "same_sample_identity": True,
        "same_seed": 43018,
        "configuration_selected_from_results": False,
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
    pd.concat([source.assign(method="upstream"), *candidates.values()]).to_csv(
        OUTPUT / "paired_per_record.csv", index=False
    )
    _write_json(OUTPUT / "summary.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
