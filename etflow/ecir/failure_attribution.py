"""Validation-only metric and attribution primitives for the final Medium audit."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

from .geometry import bond_lengths


STAGE_ORDER = (
    "upstream", "raw_proposal", "atom_clipped_proposal",
    "trust_clipped_proposal", "safety_gated_proposal", "accepted", "minimal_target",
)
THRESHOLD_BINS = (-math.inf, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, math.inf)
THRESHOLD_LABELS = (
    "lt_-20pct", "-20_to_-10pct", "-10_to_-5pct", "-5_to_0pct",
    "0_to_5pct", "5_to_10pct", "10_to_20pct", "gt_20pct",
)


def relative_improvement(upstream: float, candidate: float) -> float:
    upstream = float(upstream)
    if abs(upstream) <= 1.0e-12:
        return 0.0
    return (upstream - float(candidate)) / upstream


def molecule_equal_aggregate(
    records: pd.DataFrame,
    value_columns: Sequence[str],
    *,
    molecule_column: str = "molecule_id",
) -> tuple[pd.DataFrame, dict[str, float]]:
    if records[molecule_column].isna().any():
        raise ValueError("molecule aggregation cannot include missing molecule IDs")
    molecules = records.groupby(molecule_column, sort=True)[list(value_columns)].mean()
    return molecules, {name: float(molecules[name].mean()) for name in value_columns}


def threshold_margin(length: float, lower: float, upper: float) -> float:
    center = 0.5 * (float(lower) + float(upper))
    threshold = max(0.5 * (float(upper) - float(lower)), 1.0e-12)
    return (abs(float(length) - center) - threshold) / threshold


def threshold_bucket(margin: float) -> str:
    index = int(np.searchsorted(np.asarray(THRESHOLD_BINS[1:-1]), float(margin), side="right"))
    return THRESHOLD_LABELS[index]


def bond_observations(validity, coordinates: torch.Tensor, record: Any) -> pd.DataFrame:
    prepared = validity._prepare(record)
    coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
    lengths = bond_lengths(coordinates, prepared["bonds"]).detach().cpu().numpy()
    stats = prepared["bond_stats"].detach().cpu().numpy()
    ring = prepared["ring_mask"].detach().cpu().numpy().astype(bool)
    atom, bond_metadata, _ = validity._environment(record)
    degrees = torch.bincount(
        prepared["bonds"].reshape(-1), minlength=len(atom)
    ).cpu().numpy()
    rows = []
    for index, ((left, right), length, stat, is_ring) in enumerate(zip(
        prepared["bonds"].t().tolist(), lengths, stats, ring
    )):
        lower, upper, robust_scale = (float(value) for value in stat)
        excess = max(lower - float(length), float(length) - upper, 0.0)
        left_z, right_z = int(atom[left][0]), int(atom[right][0])
        bond_type, aromatic, metadata_ring = bond_metadata[tuple(sorted((left, right)))]
        rows.append({
            "bond_index": index, "left_atom": int(left), "right_atom": int(right),
            "left_atomic_number": left_z, "right_atomic_number": right_z,
            "bond_type": str(bond_type), "aromatic": bool(aromatic),
            "ring": bool(is_ring or metadata_ring),
            "heteroatom_bond": bool(left_z not in {1, 6} or right_z not in {1, 6}),
            "branched": bool(degrees[left] >= 3 or degrees[right] >= 3),
            "length": float(length), "lower": lower, "upper": upper,
            "robust_scale": robust_scale, "outlier": bool(excess > 0.0),
            "threshold_excess": excess,
            "normalized_excess": excess / max(robust_scale, 1.0e-8),
            "threshold_margin": threshold_margin(float(length), lower, upper),
            "threshold_bucket": threshold_bucket(threshold_margin(float(length), lower, upper)),
        })
    return pd.DataFrame(rows)


def transition_labels(upstream_flags: Sequence[bool], candidate_flags: Sequence[bool]) -> np.ndarray:
    upstream = np.asarray(upstream_flags, dtype=bool)
    candidate = np.asarray(candidate_flags, dtype=bool)
    if upstream.shape != candidate.shape:
        raise ValueError("bond transition arrays must have identical shapes")
    labels = np.empty(upstream.shape, dtype=object)
    labels[~upstream & ~candidate] = "normal_to_normal"
    labels[upstream & ~candidate] = "outlier_to_normal"
    labels[upstream & candidate] = "outlier_to_outlier"
    labels[~upstream & candidate] = "normal_to_outlier"
    return labels


def stage_gain_decomposition(values: Mapping[str, float]) -> dict[str, float]:
    upstream = float(values["upstream"])
    raw = float(values["raw_proposal"])
    atom = float(values["atom_clipped_proposal"])
    clipped = float(values["trust_clipped_proposal"])
    safe = float(values["safety_gated_proposal"])
    accepted = float(values["accepted"])
    target = float(values["minimal_target"])
    raw_gain = upstream - raw
    atom_gain = upstream - atom
    clipped_gain = upstream - clipped
    safe_gain = upstream - safe
    accepted_gain = upstream - accepted
    return {
        "target_available_gain": upstream - target,
        "raw_potential_gain": raw_gain,
        "atom_clipping_loss": raw_gain - atom_gain,
        "graph_clipping_loss": atom_gain - clipped_gain,
        "clipping_loss": raw_gain - clipped_gain,
        "safety_gate_loss": clipped_gain - safe_gain,
        "acceptance_loss": safe_gain - accepted_gain,
        "accepted_gain": accepted_gain,
        "target_gap": accepted - target,
    }


DEFAULT_CLASSIFICATION_RULES = {
    "headroom_rate_max": 0.01,
    "target_relative_min": 0.10,
    "dominant_share_min": 0.50,
    "model_recovery_min": 0.50,
    "cancellation_fraction_min": 0.50,
    "threshold_near_fraction_min": 0.50,
    "threshold_near_absolute_margin": 0.05,
}


def classify_failure(
    values: Mapping[str, float],
    *,
    rules: Mapping[str, float] | None = None,
) -> str:
    settings = {**DEFAULT_CLASSIFICATION_RULES, **dict(rules or {})}
    upstream = float(values["upstream_bond_outlier_rate"])
    available = float(values["target_available_gain"])
    if upstream <= settings["headroom_rate_max"] or available <= 1.0e-12:
        return "ALREADY_VALID_OR_NO_HEADROOM"
    if relative_improvement(upstream, upstream - available) < settings["target_relative_min"]:
        return "TARGET_LIMITED"

    losses = {
        "MODEL_PROPOSAL_LIMITED": max(available - float(values["raw_potential_gain"]), 0.0),
        "TRUST_CLIP_LIMITED": max(float(values["clipping_loss"]), 0.0),
        "SAFETY_GATE_LIMITED": max(float(values["safety_gate_loss"]), 0.0),
        "ACCEPTANCE_LIMITED": max(float(values["acceptance_loss"]), 0.0),
    }
    positive_total = sum(losses.values())
    cancellation = float(values.get("new_outlier_count", 0.0)) / max(
        float(values.get("repaired_bond_count", 0.0)), 1.0
    )
    if cancellation >= settings["cancellation_fraction_min"] and float(values.get("new_outlier_count", 0.0)) > 0:
        return "CANCELLATION_OR_NEW_OUTLIER"
    if (
        float(values.get("threshold_near_fraction", 0.0)) >= settings["threshold_near_fraction_min"]
        and float(values.get("bond_magnitude_improvement", 0.0)) > 0.0
        and float(values.get("accepted_gain", 0.0)) <= 0.0
    ):
        return "THRESHOLD_EDGE"
    if positive_total <= 1.0e-12:
        return "MIXED"
    dominant, amount = max(losses.items(), key=lambda item: item[1])
    if amount / positive_total >= settings["dominant_share_min"]:
        return dominant
    return "MIXED"


def paired_relative_bootstrap(
    upstream: Sequence[float],
    candidate: Sequence[float],
    *,
    draws: int = 10_000,
    seed: int = 42,
    threshold: float = 0.10,
) -> dict[str, float]:
    upstream = np.asarray(upstream, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if upstream.shape != candidate.shape or upstream.ndim != 1 or upstream.size == 0:
        raise ValueError("paired bootstrap inputs must be nonempty matched vectors")
    rng = np.random.default_rng(seed)
    values = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        indices = rng.integers(0, upstream.size, upstream.size)
        values[draw] = relative_improvement(upstream[indices].mean(), candidate[indices].mean())
    return {
        "draws": int(draws), "seed": int(seed),
        "point_estimate": relative_improvement(upstream.mean(), candidate.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "probability_ge_10pct": float(np.mean(values >= float(threshold))),
        "bootstrap_mean": float(values.mean()),
    }


def leave_one_out_influence(
    molecule_ids: Sequence[str], upstream: Sequence[float], candidate: Sequence[float]
) -> pd.DataFrame:
    molecule_ids = np.asarray(molecule_ids, dtype=object)
    upstream = np.asarray(upstream, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if not (len(molecule_ids) == upstream.size == candidate.size):
        raise ValueError("influence inputs must have equal lengths")
    full = relative_improvement(upstream.mean(), candidate.mean())
    rows = []
    for index, molecule_id in enumerate(molecule_ids):
        mask = np.ones(upstream.size, dtype=bool)
        mask[index] = False
        leave_one_out = relative_improvement(upstream[mask].mean(), candidate[mask].mean())
        rows.append({
            "molecule_id": str(molecule_id), "full_relative_improvement": full,
            "leave_one_out_relative_improvement": leave_one_out,
            "influence_score": full - leave_one_out,
            "absolute_influence": abs(full - leave_one_out),
        })
    return pd.DataFrame(rows).sort_values("absolute_influence", ascending=False).reset_index(drop=True)
