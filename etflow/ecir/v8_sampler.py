"""Train-only rare-error stratification for MCVR V8."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
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
    low_error = frame.get("low_error_minimal_movement_flag", score <= threshold)
    low_error = low_error.fillna(False).astype(bool)
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
    target_manifest: str | Path | None = None,
    target_cache_root: str | Path | None = None,
    molecule_exposure_cap: float = 4.0,
) -> dict[str, Any]:
    frame = pd.read_parquet(source_manifest)
    target_sha256 = None
    if target_manifest is not None:
        if target_cache_root is None:
            raise ValueError("formal stratification requires target_cache_root")
        targets = pd.read_parquet(target_manifest)
        _assert_train_only(targets)
        if set(frame.sample_id.astype(str)) != set(targets.sample_id.astype(str)):
            raise RuntimeError("V8 sampler source-target identity differs")
        indexed = targets.set_index("sample_id")
        values: dict[str, list[float]] = {
            "source_angle_outlier_rate": [],
            "source_clash_penetration": [],
            "source_severe_clash_rate": [],
            "source_ring_bond_outlier_rate": [],
            "source_ring_planarity_outlier_rate": [],
            "source_total_thresholded_validity_score": [],
            "target_movement": [],
        }
        root = Path(target_cache_root)
        for offset, row in enumerate(frame.itertuples(index=False), start=1):
            target = indexed.loc[str(row.sample_id)]
            path = root / str(row.split) / Path(str(target.target_cache_path)).name
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if str(payload.get("sample_id")) != str(row.sample_id):
                raise RuntimeError(f"V8 sampler target payload identity differs: {row.sample_id}")
            if int(payload.get("test_records_read", -1)) != 0:
                raise RuntimeError(f"V8 sampler target isolation changed: {row.sample_id}")
            initial = payload["target_metadata"]["initial_validity"]
            values["source_angle_outlier_rate"].append(float(initial["angle_outlier_rate"]))
            values["source_clash_penetration"].append(float(initial["clash_penetration"]))
            values["source_severe_clash_rate"].append(float(initial["severe_clash_rate"]))
            values["source_ring_bond_outlier_rate"].append(float(initial["ring_bond_outlier_rate"]))
            values["source_ring_planarity_outlier_rate"].append(
                float(initial["ring_planarity_outlier_rate"])
            )
            values["source_total_thresholded_validity_score"].append(
                float(initial["total_thresholded_validity_score"])
            )
            values["target_movement"].append(float(target.initial_to_target_rmsd))
            if offset % 1000 == 0 or offset == len(frame):
                print(f"formal_stratification_progress={offset}/{len(frame)}", flush=True)
        frame = frame.copy()
        for name, column in values.items():
            frame[name] = column
        score_limit = float(frame["source_total_thresholded_validity_score"].quantile(0.25))
        movement_limit = float(frame["target_movement"].quantile(0.25))
        frame["low_error_minimal_movement_flag"] = (
            frame["source_total_thresholded_validity_score"] <= score_limit
        ) & (frame["target_movement"] <= movement_limit)
        frame["rotatable_group"] = (
            frame["num_rotatable_bonds"]
            .fillna(0)
            .astype(int)
            .map(lambda value: "ge_6" if value >= 6 else "lt_6")
        )
        target_sha256 = _sha256(target_manifest)
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
                "uncapped_sampling_weight": sampling_weight,
                "sampling_weight": sampling_weight,
            }
        )
    molecule_totals: dict[str, float] = defaultdict(float)
    for record in records:
        molecule_totals[record["molecule_id"]] += float(record["sampling_weight"])
    for record in records:
        total = molecule_totals[record["molecule_id"]]
        record["sampling_weight"] *= min(1.0, float(molecule_exposure_cap) / total)
    stable = {
        "schema_version": "mcvr-v8-train-stratified-manifest-v1",
        "split": "train",
        "source_manifest": str(Path(source_manifest).resolve()),
        "source_manifest_sha256": _sha256(source_manifest),
        "target_manifest_sha256": target_sha256,
        "record_count": len(records),
        "records_scanned": len(records),
        "cohort_counts": counts,
        "overlap_counts": {
            f"{left}&{right}": int((masks[left] & masks[right]).sum())
            for left, right in combinations(COHORTS[1:], 2)
        },
        "cohort_weights": weights,
        "molecule_exposure_cap": float(molecule_exposure_cap),
        "molecule_count": len(molecule_totals),
        "capped_molecule_count": sum(
            total > float(molecule_exposure_cap) for total in molecule_totals.values()
        ),
        "records": records,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_used": False,
        "validation_records_read": 0,
        "test_used": False,
        "formal_test_records_read": 0,
        "formal_test_assets_opened": False,
        "minimal_validity_target_test_used": False,
        "frozen_holdout_used": False,
        "frozen_holdout_records_read": 0,
        "parameter_selection_from_formal_test": False,
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
