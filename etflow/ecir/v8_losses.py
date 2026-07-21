"""Independently aggregated training losses for MCVR V8 Full v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .audit import field
from .bac_constraints import sparse_clash_edges
from .geometry import bond_angles, bond_lengths
from .model import _atom_batch
from .v8_constraint_normalization import FrozenResidualScales
from .v8_error_state import confidence_regularization


def active_only_mean(values: Tensor, active_weight: Tensor) -> Tensor:
    """Return a weighted active mean; inactive graphs/rows never enter its denominator."""

    values = torch.as_tensor(values)
    active_weight = torch.as_tensor(
        active_weight, device=values.device, dtype=values.dtype
    ).reshape(-1)
    values = values.reshape(-1)
    if values.numel() != active_weight.numel():
        raise ValueError("active-only value and weight counts differ")
    denominator = active_weight.sum()
    if not values.numel() or not bool(denominator.detach() > 0):
        return values.new_zeros(())
    return (values * active_weight).sum() / denominator


def _interval_violation(values: Tensor, ranges: Tensor) -> Tensor:
    ranges = torch.as_tensor(ranges, device=values.device, dtype=values.dtype).reshape(-1, 3)
    return torch.maximum(ranges[:, 0] - values, values - ranges[:, 1]).clamp_min(0.0)


def _constraints(batch: Any, name: str, width: int, device: torch.device) -> Tensor:
    value = field(batch, name)
    if value is None:
        return torch.empty((width, 0), device=device, dtype=torch.long)
    result = torch.as_tensor(value, device=device, dtype=torch.long)
    if width == 3 and result.ndim == 2 and result.size(0) != 3:
        result = result.t()
    return result.reshape(width, -1)


def smooth_clash_loss(
    coordinates: Tensor,
    batch: Any,
    *,
    safe_distance: float = 1.0,
    cutoff: float = 2.0,
    temperature: float = 0.05,
    exclude_topology_distance: int = 2,
    max_edges_per_graph: int = 128,
    residual_scale: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    bonds = _constraints(batch, "active_bond_constraint_index", 2, coordinates.device)
    atom_batch = _atom_batch(batch, coordinates)
    clash = sparse_clash_edges(
        coordinates,
        bonds,
        atom_batch=atom_batch,
        cutoff=cutoff,
        allowed_contact=safe_distance,
        exclude_topology_distance=exclude_topology_distance,
        max_edges_per_graph=max_edges_per_graph,
    )
    penetration = clash["penetration"]
    active = clash["active_mask"].to(coordinates.dtype)
    barrier = F.softplus((clash["allowed_contact"] - clash["distance"]) / temperature)
    loss = active_only_mean((barrier / float(residual_scale)).square(), active)
    return loss, {
        "clash_pair_count": coordinates.new_tensor(clash["edge_index"].size(1)),
        "active_clash_pair_count": active.sum(),
        "degenerate_clash_pair_count": (clash["distance"] <= 1.0e-8).sum().to(coordinates.dtype),
        "clash_penetration_rms": torch.sqrt(active_only_mean(penetration.square(), active)),
    }


def ring_loss(
    coordinates: Tensor,
    source: Tensor,
    batch: Any,
    *,
    target: Tensor | None = None,
    residual_scale: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    ring_bonds = _constraints(batch, "protected_ring_bond_index", 2, coordinates.device)
    if not ring_bonds.numel():
        zero = coordinates.new_zeros(())
        return zero, {
            "applicable_ring_bond_count": zero,
            "ring_source_deformation": zero,
            "ring_target_loss": zero,
            "ring_source_noninferiority_loss": zero,
        }
    current = bond_lengths(coordinates, ring_bonds)
    baseline = bond_lengths(source, ring_bonds).detach()
    # Permit train-scale source-relative motion; supervise target and non-inferiority separately.
    deformation = (current - baseline).abs()
    tolerance = 2.0 * float(residual_scale)
    source_excess = torch.relu(deformation - tolerance) / float(residual_scale)
    source_noninferiority = F.smooth_l1_loss(source_excess, torch.zeros_like(source_excess))
    target_term = coordinates.new_zeros(())
    if target is not None:
        target_lengths = bond_lengths(target, ring_bonds).detach()
        target_term = F.smooth_l1_loss(
            current / float(residual_scale), target_lengths / float(residual_scale)
        )
    loss = target_term + source_noninferiority
    return loss, {
        "applicable_ring_bond_count": coordinates.new_tensor(current.numel()),
        "ring_source_deformation": deformation.mean(),
        "ring_target_loss": target_term,
        "ring_source_noninferiority_loss": source_noninferiority,
    }


def _signed_volumes(coordinates: Tensor, quads: Tensor) -> Tensor:
    if not quads.numel():
        return coordinates.new_empty(0)
    center, first, second, third = quads
    matrices = torch.stack(
        (
            coordinates[first] - coordinates[center],
            coordinates[second] - coordinates[center],
            coordinates[third] - coordinates[center],
        ),
        dim=1,
    )
    return torch.linalg.det(matrices)


def chirality_barrier(
    coordinates: Tensor,
    source: Tensor,
    batch: Any,
    *,
    source_epsilon: float = 1.0e-5,
    margin_fraction: float = 0.05,
    temperature: float = 0.02,
) -> tuple[Tensor, dict[str, Tensor]]:
    quads = _constraints(batch, "protected_chirality_constraint_index", 4, coordinates.device)
    if not quads.numel():
        zero = coordinates.new_zeros(())
        return zero, {
            "applicable_stereocenter_count": zero,
            "near_degenerate_chirality_count": zero,
            "chirality_sign_flip_count": zero,
        }
    source_volume = _signed_volumes(source, quads).detach()
    current_volume = _signed_volumes(coordinates, quads)
    applicable = source_volume.abs() > float(source_epsilon)
    scale = source_volume.abs().clamp_min(float(source_epsilon))
    signed_fraction = source_volume.sign() * current_volume / scale
    barrier = F.softplus((float(margin_fraction) - signed_fraction) / float(temperature))
    weight = applicable.to(coordinates.dtype)
    loss = active_only_mean(barrier.square(), weight)
    return loss, {
        "applicable_stereocenter_count": weight.sum(),
        "near_degenerate_chirality_count": (
            applicable & (current_volume.abs() <= float(source_epsilon))
        )
        .sum()
        .to(coordinates.dtype),
        "chirality_sign_flip_count": (applicable & (source_volume.sign() != current_volume.sign()))
        .sum()
        .to(coordinates.dtype),
    }


@dataclass(frozen=True)
class V8LossWeights:
    target: float = 1.0
    movement: float = 0.1
    error_state: float = 0.1
    confidence_regularization: float = 0.01
    bond: float = 0.1
    angle: float = 0.1
    clash: float = 0.1
    ring: float = 0.05
    chirality: float = 0.05
    step_consistency: float = 0.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "V8LossWeights":
        values = values or {}
        aliases = {
            "target_weight": "target",
            "movement_weight": "movement",
            "error_state_weight": "error_state",
            "confidence_regularization_weight": "confidence_regularization",
            "bond_weight": "bond",
            "angle_weight": "angle",
            "clash_weight": "clash",
            "ring_weight": "ring",
            "chirality_weight": "chirality",
            "step_consistency_weight": "step_consistency",
        }
        payload = {aliases.get(key, key): value for key, value in values.items()}
        payload = {key: value for key, value in payload.items() if key in cls.__dataclass_fields__}
        result = cls(**payload)
        if any(float(value) < 0 for value in result.__dict__.values()):
            raise ValueError("V8 loss weights must be nonnegative")
        return result


class MCVRV8Loss(nn.Module):
    """Full V8 objective with separate type denominators and diagnostics."""

    def __init__(
        self,
        weights: V8LossWeights | Mapping[str, Any] | None = None,
        *,
        confidence_min: float = 0.25,
        confidence_max: float = 4.0,
        clash_settings: Mapping[str, Any] | None = None,
        residual_scales: FrozenResidualScales | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.weights = (
            weights if isinstance(weights, V8LossWeights) else V8LossWeights.from_mapping(weights)
        )
        self.confidence_min = float(confidence_min)
        self.confidence_max = float(confidence_max)
        self.clash_settings = dict(clash_settings or {})
        self.residual_scales = (
            residual_scales
            if isinstance(residual_scales, FrozenResidualScales)
            else FrozenResidualScales.from_mapping(residual_scales or {"bond": 1.0, "angle": 1.0})
        )

    def forward(self, output: Mapping[str, Any], batch: Any) -> dict[str, Tensor]:
        final = output["x_final"]
        source = torch.as_tensor(field(batch, "x_input"), device=final.device, dtype=final.dtype)
        target = torch.as_tensor(field(batch, "x_target"), device=final.device, dtype=final.dtype)
        atom_batch = _atom_batch(batch, final)
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        target_loss = F.smooth_l1_loss(final, target)
        displacement = final - source
        movement_loss = displacement.square().sum(-1).mean()
        target_magnitude = torch.linalg.vector_norm(target - source, dim=-1)
        error_atom = F.smooth_l1_loss(output["predicted_error_magnitude"], target_magnitude)
        target_energy = final.new_zeros(graphs)
        target_energy.index_add_(0, atom_batch, (target - source).square().sum(-1))
        counts = torch.bincount(atom_batch, minlength=graphs).clamp_min(1).to(final.dtype)
        target_graph_rms = torch.sqrt(target_energy / counts)
        target_max = []
        for graph in range(graphs):
            values = target_magnitude[atom_batch == graph]
            target_max.append(values.max() if values.numel() else final.new_zeros(()))
        target_max_tensor = torch.stack(target_max)
        error_state_loss = (
            error_atom
            + F.smooth_l1_loss(output["predicted_graph_correction_rms"], target_graph_rms)
            + F.smooth_l1_loss(output["predicted_max_atom_correction"], target_max_tensor)
        )
        confidence = output["bounded_prior_confidence"]
        confidence_loss = confidence_regularization(
            confidence,
            confidence_min=self.confidence_min,
            confidence_max=self.confidence_max,
        )

        bonds = _constraints(batch, "active_bond_constraint_index", 2, final.device)
        bond_ranges = field(batch, "bond_allowed_range")
        if bonds.numel() and bond_ranges is not None:
            bond_violation = _interval_violation(bond_lengths(final, bonds), bond_ranges)
            bond_active = (bond_violation > 0).to(final.dtype)
            bond_loss = active_only_mean(
                (bond_violation / self.residual_scales.bond).square(), bond_active
            )
        else:
            bond_violation = final.new_empty(0)
            bond_active = final.new_empty(0)
            bond_loss = final.new_zeros(())
        angles = _constraints(batch, "active_angle_constraint_index", 3, final.device).t()
        angle_ranges = field(batch, "angle_allowed_range")
        if angles.numel() and angle_ranges is not None:
            angle_values = bond_angles(final, angles)
            angle_ranges_tensor = torch.as_tensor(
                angle_ranges, device=final.device, dtype=final.dtype
            ).reshape(-1, 3)
            angle_violation = _interval_violation(angle_values, angle_ranges_tensor)
            angle_active = (angle_violation > 0).to(final.dtype)
            angle_boundary = torch.where(
                angle_values < angle_ranges_tensor[:, 0],
                angle_ranges_tensor[:, 0],
                torch.where(
                    angle_values > angle_ranges_tensor[:, 1],
                    angle_ranges_tensor[:, 1],
                    angle_values,
                ),
            )
            cosine_residual = torch.cos(angle_values) - torch.cos(angle_boundary)
            angle_loss = active_only_mean(
                (cosine_residual / self.residual_scales.angle).square(), angle_active
            )
        else:
            angle_violation = final.new_empty(0)
            angle_active = final.new_empty(0)
            angle_loss = final.new_zeros(())
        clash_loss, clash_diag = smooth_clash_loss(
            final,
            batch,
            residual_scale=self.residual_scales.clash,
            **self.clash_settings,
        )
        ring_value, ring_diag = ring_loss(
            final,
            source,
            batch,
            target=target,
            residual_scale=self.residual_scales.ring,
        )
        chirality_value, chirality_diag = chirality_barrier(final, source, batch)
        deltas = output.get("step_deltas", ())
        step_consistency = (
            F.smooth_l1_loss(deltas[1], deltas[0]) if len(deltas) > 1 else final.new_zeros(())
        )
        losses = {
            "target_loss": target_loss,
            "movement_loss": movement_loss,
            "error_state_loss": error_state_loss,
            "confidence_regularization_loss": confidence_loss,
            "bond_loss": bond_loss,
            "angle_loss": angle_loss,
            "clash_loss": clash_loss,
            "ring_loss": ring_value,
            "chirality_loss": chirality_value,
            "step_consistency_loss": step_consistency,
        }
        d1_only_objective = all(
            float(getattr(self.weights, name)) == 0.0
            for name in (
                "error_state",
                "confidence_regularization",
                "bond",
                "angle",
                "clash",
                "ring",
                "chirality",
                "step_consistency",
            )
        )
        if d1_only_objective:
            zero = final.new_zeros(())
            for name in (
                "error_state_loss",
                "confidence_regularization_loss",
                "bond_loss",
                "angle_loss",
                "clash_loss",
                "ring_loss",
                "chirality_loss",
                "step_consistency_loss",
            ):
                losses[name] = zero
        total = sum(
            getattr(self.weights, name) * losses[f"{name}_loss"]
            for name in (
                "target",
                "movement",
                "error_state",
                "confidence_regularization",
                "bond",
                "angle",
                "clash",
                "ring",
                "chirality",
                "step_consistency",
            )
        )
        displacement_norm = torch.linalg.vector_norm(displacement, dim=-1)
        graph_energy = final.new_zeros(graphs)
        graph_energy.index_add_(0, atom_batch, displacement.square().sum(-1))
        graph_rms = torch.sqrt(graph_energy / counts)
        step_outputs = output.get("step_outputs", ())
        solver_failures = sum(
            (step["solver_failure"].sum() for step in step_outputs), final.new_zeros(())
        )
        bond_contributions = [
            step["solver_bond_contribution"].mean()
            for step in step_outputs
            if "solver_bond_contribution" in step
        ]
        angle_contributions = [
            step["solver_angle_contribution"].mean()
            for step in step_outputs
            if "solver_angle_contribution" in step
        ]
        bond_contribution = (
            torch.stack(bond_contributions).mean() if bond_contributions else final.new_zeros(())
        )
        angle_contribution = (
            torch.stack(angle_contributions).mean() if angle_contributions else final.new_zeros(())
        )
        conditions = [
            step["condition_estimate"].reshape(-1)
            for step in step_outputs
            if "condition_estimate" in step
        ]
        condition_values = torch.cat(conditions) if conditions else final.new_empty(0)

        def step_mean(name: str) -> Tensor:
            values = [step[name].float().mean() for step in step_outputs if name in step]
            return torch.stack(values).mean() if values else final.new_zeros(())

        def step_sum(name: str) -> Tensor:
            values = [step[name].float().sum() for step in step_outputs if name in step]
            return torch.stack(values).sum() if values else final.new_zeros(())

        def active_graph_mean(name: str) -> Tensor:
            values = [(step[name] > 0).float().sum() for step in step_outputs if name in step]
            return torch.stack(values).mean() if values else final.new_zeros(())

        confidence_error_correlation = final.new_zeros(())
        if confidence.numel() > 1:
            confidence_error_correlation = torch.nan_to_num(
                torch.corrcoef(torch.stack((confidence.reshape(-1), target_magnitude.reshape(-1))))[
                    0, 1
                ]
            )
        step_zero_contribution = (
            torch.linalg.vector_norm(deltas[0], dim=-1).mean()
            if len(deltas) > 0
            else final.new_zeros(())
        )
        step_one_contribution = (
            torch.linalg.vector_norm(deltas[1], dim=-1).mean()
            if len(deltas) > 1
            else final.new_zeros(())
        )
        return {
            "loss": total,
            **losses,
            "active_bond_count": bond_active.sum(),
            "active_angle_count": angle_active.sum(),
            **clash_diag,
            **ring_diag,
            **chirality_diag,
            "displacement_mean": displacement_norm.mean(),
            "displacement_max": displacement_norm.max(),
            "displacement_p50": torch.quantile(displacement_norm, 0.50),
            "displacement_p95": torch.quantile(displacement_norm, 0.95),
            "graph_displacement_rms_mean": graph_rms.mean(),
            "graph_displacement_rms_max": graph_rms.max(),
            "solver_failure_count": solver_failures,
            "solver_call_count": final.new_tensor(
                sum(
                    sum(status != "DISABLED" for status in step.get("solver_status", ()))
                    for step in step_outputs
                )
            ),
            "solver_fallback_count": solver_failures,
            "solver_condition_mean": condition_values.mean()
            if condition_values.numel()
            else final.new_zeros(()),
            "solver_condition_max": condition_values.max()
            if condition_values.numel()
            else final.new_zeros(()),
            "solver_nonfinite_count": final.new_zeros(()),
            "solver_duration_seconds": step_sum("solver_duration_seconds"),
            "solver_bond_contribution": bond_contribution,
            "solver_angle_contribution": angle_contribution,
            "solver_contribution_ratio": bond_contribution
            / angle_contribution.clamp_min(torch.finfo(final.dtype).eps),
            "active_bond_graph_count": active_graph_mean("bond_hard_active_count"),
            "active_angle_graph_count": active_graph_mean("angle_hard_active_count"),
            "active_clash_graph_count": (clash_diag["active_clash_pair_count"] > 0).to(final.dtype),
            "raw_bond_residual_rms": step_mean("bond_raw_residual_rms"),
            "normalized_bond_residual_rms": step_mean("bond_normalized_residual_rms"),
            "raw_angle_residual_rms": step_mean("angle_raw_residual_rms"),
            "normalized_angle_residual_rms": step_mean("angle_normalized_residual_rms"),
            "raw_bond_jacobian_norm": step_mean("bond_raw_jacobian_norm"),
            "normalized_bond_jacobian_norm": step_mean("bond_normalized_jacobian_norm"),
            "raw_angle_jacobian_norm": step_mean("angle_raw_jacobian_norm"),
            "normalized_angle_jacobian_norm": step_mean("angle_normalized_jacobian_norm"),
            "near_linear_angle_count": step_sum("angle_near_linear_count"),
            "confidence_mean": confidence.mean(),
            "confidence_std": confidence.std(unbiased=False),
            "confidence_min": confidence.min(),
            "confidence_max": confidence.max(),
            "confidence_lower_saturation_fraction": (confidence <= self.confidence_min + 1.0e-4)
            .float()
            .mean(),
            "confidence_upper_saturation_fraction": (confidence >= self.confidence_max - 1.0e-4)
            .float()
            .mean(),
            "confidence_target_error_correlation": confidence_error_correlation,
            "step0_displacement_contribution": step_zero_contribution,
            "step1_displacement_contribution": step_one_contribution,
        }
