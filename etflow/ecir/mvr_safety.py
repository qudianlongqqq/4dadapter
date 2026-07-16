"""Raw-versus-clipped velocity diagnostics and fail-closed MCVR safety rules."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


def _distribution(velocity: Tensor, atom_batch: Tensor) -> dict[str, Any]:
    norms = torch.linalg.vector_norm(velocity, dim=-1)
    graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
    energy = velocity.new_zeros(graphs)
    energy.index_add_(0, atom_batch, velocity.square().sum(-1))
    counts = torch.bincount(atom_batch, minlength=graphs).clamp_min(1).to(velocity.dtype)
    graph_rms = torch.sqrt(energy / counts + 1.0e-12)
    return {
        "atom_mean": float(norms.mean()),
        "atom_p95": float(torch.quantile(norms, 0.95)),
        "atom_max": float(norms.max()),
        "graph_rms": float(graph_rms.max()),
        "graph_rms_mean": float(graph_rms.mean()),
        "graph_rms_per_graph": [float(value) for value in graph_rms],
    }


def trust_clip_with_diagnostics(
    raw_velocity: Tensor,
    atom_batch: Tensor,
    *,
    max_atom_norm: float,
    max_graph_rms: float,
) -> tuple[Tensor, dict[str, Any]]:
    """Apply the frozen trust clipping math and expose its two scale stages."""

    raw_norms = torch.linalg.vector_norm(raw_velocity, dim=-1)
    atom_scale = torch.clamp(float(max_atom_norm) / raw_norms.clamp_min(1.0e-12), max=1.0)
    atom_clipped = raw_velocity * atom_scale[:, None]
    graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
    energy = atom_clipped.new_zeros(graphs)
    energy.index_add_(0, atom_batch, atom_clipped.square().sum(-1))
    counts = torch.bincount(atom_batch, minlength=graphs).clamp_min(1).to(atom_clipped.dtype)
    graph_rms_before_scale = torch.sqrt(energy / counts + 1.0e-12)
    graph_scale = torch.clamp(
        float(max_graph_rms) / graph_rms_before_scale.clamp_min(1.0e-12), max=1.0
    )
    clipped = atom_clipped * graph_scale[atom_batch, None]
    atom_clipped_mask = atom_scale < 1.0
    graph_clipped_mask = graph_scale < 1.0
    return clipped, {
        "raw": _distribution(raw_velocity, atom_batch),
        "clipped": _distribution(clipped, atom_batch),
        "atom_clip_scale": float(atom_scale.min()),
        "atom_clip_scale_mean": float(atom_scale.mean()),
        "graph_clip_scale": float(graph_scale.min()),
        "graph_clip_scale_mean": float(graph_scale.mean()),
        "atom_clipped_fraction": float(atom_clipped_mask.float().mean()),
        "graph_clipped_fraction": float(graph_clipped_mask.float().mean()),
        "atom_clip_scale_per_atom": [float(value) for value in atom_scale],
        "graph_clip_scale_per_graph": [float(value) for value in graph_scale],
    }


def clipped_limit_tolerance(limit: float) -> float:
    return max(1.0e-6, float(limit) * 1.0e-5)


def evaluate_velocity_safety(
    metrics: Mapping[str, float],
    *,
    max_velocity_graph_rms_after_clip: float,
    max_velocity_atom_norm_after_clip: float,
    recent_raw_metrics: Sequence[Mapping[str, float]] = (),
    severe_multiplier: float = 4.0,
    large_area_clipping_fraction: float = 0.80,
    severe_windows: int = 5,
) -> dict[str, Any]:
    """Separate trust-limit diagnostics from actual early-stop conditions."""

    finite_fields = (
        "raw_velocity_atom_mean", "raw_velocity_atom_p95", "raw_velocity_atom_max",
        "raw_velocity_graph_rms", "clipped_velocity_atom_mean",
        "clipped_velocity_atom_p95", "clipped_velocity_atom_max",
        "clipped_velocity_graph_rms",
    )
    if any(not math.isfinite(float(metrics[name])) for name in finite_fields):
        return {"status": "HARD_STOP", "reason": "velocity_nan_or_inf"}
    graph_limit = float(max_velocity_graph_rms_after_clip)
    atom_limit = float(max_velocity_atom_norm_after_clip)
    if float(metrics["clipped_velocity_graph_rms"]) > graph_limit + clipped_limit_tolerance(graph_limit):
        return {"status": "HARD_STOP", "reason": "graph_trust_clipping_failed"}
    if float(metrics["clipped_velocity_atom_max"]) > atom_limit + clipped_limit_tolerance(atom_limit):
        return {"status": "HARD_STOP", "reason": "atom_trust_clipping_failed"}

    history = [*recent_raw_metrics, metrics][-int(severe_windows):]
    if len(history) == int(severe_windows):
        sustained_severe = all(
            (
                float(row["raw_velocity_graph_rms"]) >= severe_multiplier * graph_limit
                or float(row["raw_velocity_atom_max"]) >= severe_multiplier * atom_limit
            )
            and max(
                float(row["graph_clipped_fraction"]),
                float(row["atom_clipped_fraction"]),
            ) >= large_area_clipping_fraction
            for row in history
        )
        if sustained_severe:
            return {"status": "HARD_STOP", "reason": "sustained_severe_raw_divergence_with_large_area_clipping"}

    raw_over = (
        float(metrics["raw_velocity_graph_rms"]) >= graph_limit
        or float(metrics["raw_velocity_atom_max"]) >= atom_limit
    )
    clipping = max(
        float(metrics["graph_clipped_fraction"]), float(metrics["atom_clipped_fraction"])
    )
    if raw_over or clipping > 0.0:
        return {"status": "WARNING", "reason": "raw_velocity_clipped_output_safe"}
    return {"status": "INFO", "reason": "velocity_within_trust_region"}


def evaluate_validation_safety(
    validation_history: Sequence[Mapping[str, Any]],
    *,
    clean_identity_min: float = 0.90,
) -> dict[str, str]:
    """Apply unchanged chemical rules plus the two-transition composite rule."""

    if not validation_history:
        return {"status": "INFO", "reason": "no_validation"}
    current = validation_history[-1]
    identity = float(current.get("identity_fraction", math.nan))
    if math.isfinite(identity) and identity < clean_identity_min:
        return {"status": "HARD_STOP", "reason": "clean_identity_below_90pct"}
    if float(current.get("severe_clash_delta", 0.0)) > 1.0e-9:
        return {"status": "HARD_STOP", "reason": "severe_clash_increased"}
    if float(current.get("chirality_delta", 0.0)) > 1.0e-9:
        return {"status": "HARD_STOP", "reason": "chirality_worsened"}
    if len(validation_history) < 3:
        return {"status": "INFO", "reason": "validation_safe"}
    transitions = []
    for previous, value in zip(validation_history[-3:-1], validation_history[-2:]):
        accuracy_worse = all(
            float(value["bootstrap"][metric]["mean"]) > 0.0
            for metric in ("aligned_RMSD", "MAT_P", "MAT_R")
        )
        transitions.append(
            float(value["validity_delta"]) > 0.0
            and float(value["mean_displacement"]) > float(previous["mean_displacement"]) + 1.0e-4
            and accuracy_worse
        )
    if all(transitions):
        return {
            "status": "HARD_STOP",
            "reason": "two_validations_joint_validity_displacement_accuracy_worsening",
        }
    return {"status": "INFO", "reason": "validation_safe"}
