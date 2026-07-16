"""Decision helpers for the ECIR MCVR Stage B inference sweep."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_PAIR_KEYS = (
    "teacher_steps",
    "update_scale",
    "trust_radius_scale",
    "gate_threshold",
    "phase",
    "acceptance_mode",
)


def compare_train_range_to_legacy(
    frame: pd.DataFrame,
    *,
    atol: float = 1.0e-6,
    rtol: float = 1.0e-6,
) -> tuple[bool, dict[str, Any]]:
    """Compare like-for-like coarse configurations for equivalent schedules.

    A checkpoint trained on ``[0, 1]`` produces the same time values under
    ``train_range`` and ``legacy_full``. The paired comparison prevents an
    asymmetric fine-search expansion from changing this decision. Small GPU
    reduction drift is handled by an explicit numerical tolerance.
    """

    coarse = frame.loc[frame["phase"].eq("coarse")]
    legacy = coarse.loc[coarse["time_schedule_mode"].eq("legacy_full")]
    train = coarse.loc[coarse["time_schedule_mode"].eq("train_range")]
    paired = legacy.merge(train, on=list(_PAIR_KEYS), suffixes=("_legacy", "_train"))
    if paired.empty:
        return False, {
            "status": "FAIL_NO_MATCHED_CONFIGS",
            "matched_configurations": 0,
            "atol": atol,
            "rtol": rtol,
        }

    metric = "delta_total_thresholded_validity_score"
    legacy_values = paired[f"{metric}_legacy"].to_numpy(dtype=float)
    train_values = paired[f"{metric}_train"].to_numpy(dtype=float)
    finite = np.isfinite(legacy_values) & np.isfinite(train_values)
    if not finite.all():
        return False, {
            "status": "FAIL_NONFINITE",
            "matched_configurations": int(len(paired)),
            "finite_configurations": int(finite.sum()),
            "atol": atol,
            "rtol": rtol,
        }

    tolerance = atol + rtol * np.abs(legacy_values)
    deltas = train_values - legacy_values
    nonworse = deltas <= tolerance
    passed = bool(nonworse.all())
    return passed, {
        "status": "PASS_PAIRED_NUMERICALLY_NONWORSE" if passed else "FAIL_PAIRED_WORSE",
        "metric": metric,
        "matched_configurations": int(len(paired)),
        "max_absolute_delta": float(np.abs(deltas).max()),
        "max_train_minus_legacy": float(deltas.max()),
        "min_train_minus_legacy": float(deltas.min()),
        "atol": atol,
        "rtol": rtol,
    }
