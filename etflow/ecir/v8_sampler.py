"""Train-only rare-error stratification for MCVR V8."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


COHORTS = (
    "natural",
    "active_angle",
    "active_clash",
    "ring_risk",
    "high_flexibility",
    "low_error_minimal_movement",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_train_only(frame: pd.DataFrame) -> None:
    if "split" not in frame or set(frame["split"].astype(str)) != {"train"}:
        raise RuntimeError("V8 stratified cohorts must be built from train only")
    for name in frame.columns:
        lowered = str(name).lower()
        if ("test" in lowered or "holdout" in lowered) and bool(
            frame[name].fillna(False).astype(bool).any()
        ):
            raise RuntimeError(f"V8 sampler manifest contains forbidden flag: {name}")


def cohort_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    _assert_train_only(frame)
    zeros = pd.Series(False, index=frame.index)
    angle = frame.get("source_angle_outlier_rate", zeros).fillna(0).astype(float) > 0
    clash = (frame.get("source_clash_penetration", zeros).fillna(0).astype(float) > 0) | (
        frame.get("source_severe_clash_rate", zeros).fillna(0).astype(float) > 0
    )
    ring = (frame.get("source_ring_bond_outlier_rate", zeros).fillna(0).astype(float) > 0) | (
        frame.get("source_ring_planarity_outlier_rate", zeros).fillna(0).astype(float) > 0
    )
    rotatable = frame.get("rotatable_group", pd.Series("", index=frame.index)).astype(str)
    high_flex = rotatable.str.contains("ge_6|6_plus|high", regex=True)
    score = (
        frame.get(
            "source_total_thresholded_validity_score", pd.Series(float("inf"), index=frame.index)
        )
        .fillna(float("inf"))
        .astype(float)
    )
    finite_score = score[score.map(math.isfinite)]
    threshold = float(finite_score.quantile(0.25)) if len(finite_score) else 0.0
    low_error = score <= threshold
    return {
        "natural": pd.Series(True, index=frame.index),
        "active_angle": angle,
        "active_clash": clash,
        "ring_risk": ring,
        "high_flexibility": high_flex,
        "low_error_minimal_movement": low_error,
    }


def derive_cohort_weights(counts: Mapping[str, int], total: int) -> dict[str, float]:
    """Choose count-driven, capped boosts without validation/test prevalence."""

    result = {"natural": 1.0}
    for name in COHORTS[1:]:
        count = max(int(counts.get(name, 0)), 1)
        result[name] = min(4.0, math.sqrt(max(int(total), 1) / count))
    return result


def build_stratified_payload(
    source_manifest: str | Path,
    *,
    cohort_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    frame = pd.read_parquet(source_manifest)
    masks = cohort_masks(frame)
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    weights = dict(cohort_weights or derive_cohort_weights(counts, len(frame)))
    if set(weights) != set(COHORTS) or any(float(value) <= 0 for value in weights.values()):
        raise ValueError("V8 sampler cohort weights must be positive and complete")
    records = []
    for index, row in frame.reset_index(drop=True).iterrows():
        memberships = [name for name in COHORTS if bool(masks[name].iloc[index])]
        # One row per sample avoids implicit duplicate sampling of overlapping cohorts.
        sampling_weight = sum(float(weights[name]) for name in memberships)
        records.append(
            {
                "sample_id": str(row["sample_id"]),
                "molecule_id": str(row["molecule_id"]),
                "cohorts": memberships,
                "sampling_weight": sampling_weight,
            }
        )
    stable = {
        "schema_version": "mcvr-v8-train-stratified-manifest-v1",
        "split": "train",
        "source_manifest": str(Path(source_manifest).resolve()),
        "source_manifest_sha256": _sha256(source_manifest),
        "record_count": len(records),
        "cohort_counts": counts,
        "cohort_weights": weights,
        "records": records,
        "validation_used": False,
        "test_used": False,
        "frozen_holdout_used": False,
    }
    stable["identity_sha256"] = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return stable


def sampler_from_payload(
    payload: Mapping[str, Any], *, num_samples: int, seed: int
) -> WeightedRandomSampler:
    if payload.get("split") != "train" or payload.get("test_used") is not False:
        raise RuntimeError("V8 sampler payload violates data isolation")
    weights = torch.tensor(
        [float(row["sampling_weight"]) for row in payload["records"]], dtype=torch.double
    )
    generator = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(weights, int(num_samples), replacement=True, generator=generator)
