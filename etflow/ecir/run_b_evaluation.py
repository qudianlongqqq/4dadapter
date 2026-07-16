"""Paired validation-only evaluation for conservative-torsion Run B."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .run_a_evaluation import (
    accuracy_gate,
    build_clean_control_items,
    infer_mvr,
    method_rows,
    paired_bootstrap,
    summarize_groups,
)


INCREMENTAL_METRICS = (
    "total_thresholded_validity_score", "bond_outlier_rate",
    "angle_outlier_rate", "ring_bond_outlier_rate",
    "torsion_prior_outlier_score", "severe_clash_rate", "chirality_error",
    "aligned_RMSD", "MAT_P", "MAT_R", "COV_P", "COV_R", "diversity",
    "accepted", "validity_worsened_fraction", "RMSD_worsened_fraction",
    "molecule_rms_displacement", "mean_torsion_change",
    "high_flex_torsion_change", "torsion_gate_active_fraction",
)


def paired_group_bootstrap(
    molecule_rows: pd.DataFrame,
    *, candidate: str, baseline: str, group: str = "all",
    draws: int = 1000, seed: int = 42,
) -> dict[str, dict[str, float]]:
    frame = molecule_rows[molecule_rows.group == group]
    result = {}
    for metric in INCREMENTAL_METRICS:
        if metric not in frame.columns:
            result[metric] = {"mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
            continue
        pivot = frame.pivot(index="molecule_id", columns="method", values=metric).dropna()
        if candidate not in pivot or baseline not in pivot:
            result[metric] = {"mean": math.nan, "ci95_low": math.nan, "ci95_high": math.nan}
            continue
        delta = pivot[candidate].to_numpy() - pivot[baseline].to_numpy()
        rng = np.random.default_rng(seed)
        means = np.asarray([
            rng.choice(delta, size=len(delta), replace=True).mean()
            for _ in range(int(draws))
        ])
        result[metric] = {
            "mean": float(delta.mean()),
            "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975)),
            "improved_molecules": int((delta < -1e-12).sum()),
            "worsened_molecules": int((delta > 1e-12).sum()),
        }
    return result


def incremental_accuracy_gate(
    bootstrap: Mapping[str, Mapping[str, float]],
    margins: Mapping[str, float],
) -> dict[str, bool]:
    return {
        "rmsd_mean": bootstrap["aligned_RMSD"]["mean"] <= float(margins["rmsd_mean_delta_max"]),
        "rmsd_ci": bootstrap["aligned_RMSD"]["ci95_high"] <= float(margins["rmsd_ci_upper_max"]),
        "mat_p_mean": bootstrap["MAT_P"]["mean"] <= float(margins["mat_p_mean_delta_max"]),
        "mat_p_ci": bootstrap["MAT_P"]["ci95_high"] <= float(margins["mat_p_ci_upper_max"]),
        "mat_r_mean": bootstrap["MAT_R"]["mean"] <= float(margins["mat_r_mean_delta_max"]),
        "mat_r_ci": bootstrap["MAT_R"]["ci95_high"] <= float(margins["mat_r_ci_upper_max"]),
        "cov_p": bootstrap["COV_P"]["mean"] >= -float(margins["cov_absolute_drop_max"]),
        "cov_r": bootstrap["COV_R"]["mean"] >= -float(margins["cov_absolute_drop_max"]),
    }


def _summary_row(summary: pd.DataFrame, group: str, method: str):
    row = summary[(summary.group == group) & (summary.method == method)]
    return None if row.empty or int(row.iloc[0].get("molecules", 0)) == 0 else row.iloc[0]


def evaluate_three_way(
    run_b_model,
    run_a_model,
    items: Sequence[dict[str, Any]],
    validity,
    *,
    device,
    inference: Mapping[str, Any],
    upstream_margins: Mapping[str, float],
    incremental_margins: Mapping[str, float],
    bootstrap_draws: int = 500,
    clean_control_items=None,
) -> dict[str, Any]:
    steps = int(inference["teacher_steps"])
    step_size = float(inference["step_size"])
    mode = str(inference["acceptance_mode"])
    run_a_raw, run_a_accepted, run_a_meta = infer_mvr(
        run_a_model, items, validity, device=device, steps=steps,
        step_size=step_size, acceptance_mode=mode,
    )
    run_b_raw, run_b_accepted, run_b_meta = infer_mvr(
        run_b_model, items, validity, device=device, steps=steps,
        step_size=step_size, acceptance_mode=mode,
        acceptance_config=inference.get("acceptance"),
    )
    coordinates = {
        "upstream": [item["input"] for item in items],
        "run_a_accepted": run_a_accepted,
        "run_b_raw": run_b_raw,
        "run_b_accepted": run_b_accepted,
    }
    rows = method_rows(
        items, coordinates, validity,
        {"run_a_accepted": run_a_meta, "run_b_raw": run_b_meta,
         "run_b_accepted": run_b_meta},
    )
    summary, molecules = summarize_groups(rows, items, coordinates)
    versus_upstream = paired_bootstrap(
        molecules, candidate="run_b_accepted", baseline="upstream",
        draws=bootstrap_draws,
    )
    versus_run_a = paired_group_bootstrap(
        molecules, candidate="run_b_accepted", baseline="run_a_accepted",
        draws=bootstrap_draws,
    )
    high_flex = paired_group_bootstrap(
        molecules, candidate="run_b_accepted", baseline="run_a_accepted",
        group="rotatable_ge_6", draws=bootstrap_draws,
    )
    upstream_accuracy = {
        name: bool(value) for name, value in accuracy_gate(
            summary, versus_upstream, upstream_margins,
            method="run_b_accepted",
        ).items()
    }
    incremental_accuracy = {
        name: bool(value) for name, value in incremental_accuracy_gate(
            versus_run_a, incremental_margins
        ).items()
    }

    clean_controls = list(clean_control_items or build_clean_control_items(items, validity))
    _, clean_b, clean_meta = infer_mvr(
        run_b_model, clean_controls, validity, device=device, steps=steps,
        step_size=step_size, acceptance_mode=mode,
        acceptance_config=inference.get("acceptance"),
    )
    clean_identity = float(np.mean([
        float(np.array_equal(candidate.numpy(), item["input"].numpy()))
        for candidate, item in zip(clean_b, clean_controls)
    ]))
    all_b = _summary_row(summary, "all", "run_b_accepted")
    all_a = _summary_row(summary, "all", "run_a_accepted")
    high_b = _summary_row(summary, "rotatable_ge_6", "run_b_accepted")
    return {
        "record_rows": rows, "summary": summary, "molecule_rows": molecules,
        "coordinates": coordinates,
        "bootstrap_vs_upstream": versus_upstream,
        "bootstrap_vs_run_a": versus_run_a,
        "bootstrap_high_flex_vs_run_a": high_flex,
        "upstream_accuracy_gate": upstream_accuracy,
        "incremental_accuracy_gate": incremental_accuracy,
        "upstream_accuracy_noninferior": all(upstream_accuracy.values()),
        "incremental_accuracy_noninferior": all(incremental_accuracy.values()),
        "clean_identity_fraction": clean_identity,
        "total_validity_delta_vs_run_a": float(
            all_b.total_thresholded_validity_score - all_a.total_thresholded_validity_score
        ),
        "torsion_delta_vs_run_a": float(
            all_b.torsion_prior_outlier_score - all_a.torsion_prior_outlier_score
        ),
        "high_flex_total_delta_vs_run_a": (
            float(high_b.total_thresholded_validity_score - _summary_row(
                summary, "rotatable_ge_6", "run_a_accepted"
            ).total_thresholded_validity_score) if high_b is not None else math.nan
        ),
        "mean_displacement": float(all_b.molecule_rms_displacement),
        "acceptance_fraction": float(all_b.accepted_fraction),
        "torsion_gate_mean": float(all_b.torsion_gate_mean),
        "torsion_gate_active_fraction": float(all_b.torsion_gate_active_fraction),
        "torsion_velocity_norm": float(all_b.torsion_velocity_norm),
        "torsion_velocity_fraction": float(all_b.torsion_velocity_fraction),
        "high_flex_mean_torsion_change": float(high_b.high_flex_torsion_change) if high_b is not None else math.nan,
        "high_flex_p95_torsion_change": float(high_b.high_flex_p95_torsion_change) if high_b is not None else math.nan,
        "run_b_metadata": run_b_meta,
        "clean_metadata": clean_meta,
    }
