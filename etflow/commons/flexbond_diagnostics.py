"""Pure metric helpers for label-aware FlexBond diagnostics.

This module contains no dataset access, which keeps the label-aware diagnostic
path separate from normal inference.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def vector_rms(value: Tensor) -> float:
    """Root mean squared Cartesian component magnitude."""

    return float(value.square().mean().sqrt()) if value.numel() else 0.0


def safe_cosine(left: Tensor, right: Tensor, eps: float = 1.0e-12) -> float:
    left_flat, right_flat = left.reshape(-1), right.reshape(-1)
    denominator = torch.linalg.norm(left_flat) * torch.linalg.norm(right_flat)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= eps:
        return 0.0
    value = torch.dot(left_flat, right_flat) / denominator
    return float(value) if bool(torch.isfinite(value)) else 0.0


def projection_quality(
    residual: Tensor,
    v4d_pred: Tensor,
    v4d_star: Tensor,
    *,
    correction_scale: float,
) -> dict[str, float]:
    """Compute scale-aware branch quality metrics without throwing on zeros."""

    residual_error = vector_rms(residual)
    raw_error = vector_rms(residual - v4d_star)
    scaled_star = float(correction_scale) * v4d_star
    scaled_pred = float(correction_scale) * v4d_pred
    scaled_error = vector_rms(residual - scaled_star)
    denominator = max(residual_error, 1.0e-12)
    return {
        "residual_norm": residual_error,
        "oracle_raw_error": raw_error,
        "oracle_scaled_error": scaled_error,
        "oracle_explained_ratio": 1.0 - raw_error**2 / max(residual_error**2, 1.0e-24),
        "v4d_pred_norm": vector_rms(v4d_pred),
        "v4d_star_norm": vector_rms(v4d_star),
        "scaled_v4d_pred_norm": vector_rms(scaled_pred),
        "scaled_v4d_star_norm": vector_rms(scaled_star),
        "cosine_pred_to_residual": safe_cosine(v4d_pred, residual),
        "cosine_star_to_residual": safe_cosine(v4d_star, residual),
        "pred_norm_to_residual_ratio": vector_rms(v4d_pred) / denominator,
        "scaled_pred_norm_to_residual_ratio": vector_rms(scaled_pred) / denominator,
        "oracle_norm_to_residual_ratio": vector_rms(v4d_star) / denominator,
    }


def finite_or_blank(value) -> float | str:
    """Return a CSV-safe scalar while retaining invalid-value counts elsewhere."""

    number = float(value)
    return number if math.isfinite(number) else ""

