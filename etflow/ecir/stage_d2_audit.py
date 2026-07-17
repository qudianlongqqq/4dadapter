"""Pure analysis helpers for the validation-only MCVR Stage D2 audit."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np


def stable_average_ranks(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def safe_correlation(left: Sequence[float], right: Sequence[float], *, rank: bool = False) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if left.size < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return 0.0
    if rank:
        left = stable_average_ranks(left)
        right = stable_average_ranks(right)
    left = left - left.mean()
    right = right - right.mean()
    left_energy = float(np.sum(left * left))
    right_energy = float(np.sum(right * right))
    denominator = math.sqrt(left_energy * right_energy)
    return float(np.sum(left * right) / denominator) if denominator > 0.0 else 0.0


def binary_classification(
    truth: Sequence[bool], predicted: Sequence[bool]
) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    if truth.shape != predicted.shape:
        raise ValueError("binary classification inputs must have identical shapes")
    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    tn = int((~truth & ~predicted).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "true_negative": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def calibration_table(
    scores: Sequence[float], outcomes: Sequence[float], *, bins: int = 10
) -> list[dict[str, float | int]]:
    scores = np.asarray(scores, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if scores.shape != outcomes.shape:
        raise ValueError("calibration inputs must have identical shapes")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    result = []
    for index in range(int(bins)):
        keep = (scores >= edges[index]) & (
            scores <= edges[index + 1] if index == bins - 1 else scores < edges[index + 1]
        )
        result.append({
            "bin": index, "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "count": int(keep.sum()),
            "mean_score": float(scores[keep].mean()) if keep.any() else None,
            "mean_outcome": float(outcomes[keep].mean()) if keep.any() else None,
        })
    return result


def expected_calibration_error(table: Iterable[Mapping[str, float]]) -> float:
    rows = list(table)
    total = sum(int(row["count"]) for row in rows)
    if total == 0:
        return 0.0
    return float(sum(
        int(row["count"]) / total * abs(float(row["mean_score"]) - float(row["mean_outcome"]))
        for row in rows if int(row["count"]) and math.isfinite(float(row["mean_score"]))
    ))


def branch_interference_flags(
    target: Sequence[float], cartesian: Sequence[float], bond: Sequence[float]
) -> dict[str, np.ndarray]:
    target = np.asarray(target, dtype=np.float64)
    cartesian = np.asarray(cartesian, dtype=np.float64)
    bond = np.asarray(bond, dtype=np.float64)
    if not (target.shape == cartesian.shape == bond.shape):
        raise ValueError("branch arrays must have identical shapes")
    active = np.abs(target) > 1.0e-6
    cart_correct = active & (np.sign(cartesian) == np.sign(target))
    bond_correct = active & (np.sign(bond) == np.sign(target))
    combined = cartesian + bond
    return {
        "active": active,
        "constructive": active & (cartesian * bond > 0.0),
        "cancellation": active & (cartesian * bond < 0.0),
        "same_sign": active & (np.sign(cartesian) == np.sign(bond)),
        "opposite_sign": active & (np.sign(cartesian) == -np.sign(bond)),
        "overcorrection": active & (np.abs(combined) > np.abs(target)) & (np.sign(combined) == np.sign(target)),
        "cartesian_already_fixed_bond_duplicates": cart_correct & bond_correct,
        "cartesian_correct_bond_harms": cart_correct & ~bond_correct & (np.abs(combined - target) > np.abs(cartesian - target)),
        "bond_correct_cartesian_cancels": bond_correct & ~cart_correct & (np.abs(combined - target) > np.abs(bond - target)),
        "both_wrong": active & ~cart_correct & ~bond_correct,
    }


def top_k_capture(target: Sequence[float], prediction: Sequence[float], *, fraction: float = 0.10) -> float:
    target = np.abs(np.asarray(target, dtype=np.float64))
    prediction = np.abs(np.asarray(prediction, dtype=np.float64))
    if target.size == 0:
        return 0.0
    count = max(1, int(math.ceil(target.size * float(fraction))))
    target_top = set(np.argpartition(target, -count)[-count:].tolist())
    predicted_top = set(np.argpartition(prediction, -count)[-count:].tolist())
    return len(target_top & predicted_top) / count


def mask_bond_residuals(
    residual: Sequence[float], *, ring: Sequence[bool], mode: str,
    active: Sequence[bool] | None = None,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64)
    ring = np.asarray(ring, dtype=bool)
    if residual.shape != ring.shape:
        raise ValueError("residual and ring masks must have identical shapes")
    if mode == "ring_only":
        keep = ring
    elif mode == "nonring_only":
        keep = ~ring
    elif mode == "oracle_active":
        if active is None:
            raise ValueError("oracle_active mode requires an active mask")
        keep = np.asarray(active, dtype=bool)
        if keep.shape != residual.shape:
            raise ValueError("active mask must match residual shape")
    else:
        raise ValueError(f"unknown residual mask mode: {mode}")
    return residual * keep


def approximate_gap_decomposition(
    total_gap: float, components: Mapping[str, float]
) -> dict[str, object]:
    attributable = float(sum(float(value) for value in components.values()))
    return {
        "total_gap": float(total_gap),
        "components": {name: float(value) for name, value in components.items()},
        "attributable_sum": attributable,
        "nonadditive_remainder": float(total_gap) - attributable,
        "exactly_additive": False,
    }
