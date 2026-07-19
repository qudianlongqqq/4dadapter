"""Inference-only constraint-specific Cartesian, Angle, and Clash hybrid."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .audit import field
from .bac_constraints import sparse_clash_edges
from .bac_jacobian import (
    ConstraintSystem,
    JacobianBACConfig,
    cosine_angle_residual_jacobian,
    remove_rigid_update,
    solve_damped_system,
)
from .geometry import bond_angles
from .model import _atom_batch
from .mvr_v2_bac import MCVRBACModel


V7_SCHEMA_VERSION = "mcvr-v7-constraint-specific-hybrid-v1"


def _trust_normalize(
    update: Tensor,
    *,
    max_graph_rms: float,
    max_atom: float,
) -> tuple[Tensor, float, float, float]:
    """Apply fixed graph/atom trust caps and report raw/scaled graph RMS."""

    if not update.numel():
        return update, 1.0, 0.0, 0.0
    raw_rms = float(torch.sqrt(update.square().sum(-1).mean()))
    atom_max = float(torch.linalg.vector_norm(update, dim=-1).max())
    if not math.isfinite(raw_rms) or not math.isfinite(atom_max):
        return torch.zeros_like(update), 0.0, raw_rms, 0.0
    if raw_rms <= 1.0e-30 and atom_max <= 1.0e-30:
        return update, 0.0, 0.0, 0.0
    scale = min(
        1.0,
        float(max_graph_rms) / max(raw_rms, 1.0e-30),
        float(max_atom) / max(atom_max, 1.0e-30),
    )
    scaled = update * scale
    scaled_rms = float(torch.sqrt(scaled.square().sum(-1).mean()))
    return scaled, scale, raw_rms, scaled_rms


def build_angle_constraint_system(
    coordinates: Tensor,
    angles: Tensor,
    angle_ranges: Tensor,
    config: JacobianBACConfig,
) -> ConstraintSystem:
    """Build a cosine-Angle-only analytic system for one graph."""

    config.validate()
    coordinates = torch.as_tensor(coordinates, dtype=torch.float64)
    angles = torch.as_tensor(angles, device=coordinates.device, dtype=torch.long)
    if angles.ndim == 2 and angles.size(0) == 3:
        angles = angles.t()
    angles = angles.reshape(-1, 3)
    ranges = torch.as_tensor(
        angle_ranges, device=coordinates.device, dtype=coordinates.dtype
    ).reshape(-1, 3)
    if angles.size(0) != ranges.size(0):
        raise ValueError("angle triplet and allowed-range counts differ")
    current = bond_angles(coordinates, angles)
    lower, upper = ranges[:, 0], ranges[:, 1]
    active = (current < lower) | (current > upper)
    boundary = torch.where(current < lower, lower, torch.where(current > upper, upper, current))
    active_angles = angles[active]
    residual, jacobian, degenerate, sine = cosine_angle_residual_jacobian(
        coordinates,
        active_angles,
        torch.cos(boundary[active]),
        minimum_distance=config.minimum_distance,
    )
    valid = ~degenerate
    residual = residual[valid]
    jacobian = jacobian[valid]
    sine = sine[valid]
    if residual.numel():
        weights = residual.new_full(
            residual.shape, float(config.angle_weight) / float(residual.numel())
        )
        near_linear = sine < config.near_linear_sine_threshold
        weights = weights * torch.where(
            near_linear,
            weights.new_full(weights.shape, config.near_linear_weight),
            torch.ones_like(weights),
        )
    else:
        weights = residual.new_empty(0)
        near_linear = torch.empty(0, device=coordinates.device, dtype=torch.bool)
    return ConstraintSystem(
        residual=residual,
        jacobian=jacobian,
        weights=weights,
        constraint_types=("angle",) * residual.numel(),
        counts={
            "total_bond": 0,
            "active_bond": 0,
            "total_angle": int(angles.size(0)),
            "active_angle": int(residual.numel()),
            "candidate_clash": 0,
            "active_clash": 0,
        },
        diagnostics={
            "degenerate_angle_count": int(degenerate.sum()),
            "near_linear_angle_count": int(near_linear.sum()),
            "jacobian_shape": [int(jacobian.size(0)), int(jacobian.size(1))],
        },
    )


class MCVRConstraintSpecificHybrid(nn.Module):
    """Frozen D1 Cartesian prior plus fixed Angle and Clash operators."""

    def __init__(
        self,
        prior: MCVRBACModel,
        *,
        jacobian_config: JacobianBACConfig | Mapping[str, Any] | None = None,
        integration_step_size: float = 0.25,
        angle_max_graph_rms: float = 0.01,
        angle_max_atom: float = 0.02,
        clash_max_graph_rms: float = 0.01,
        clash_max_atom: float = 0.02,
        clash_cutoff: float = 2.0,
        clash_allowed_contact: float = 1.0,
        clash_exclude_topology_distance: int = 2,
        max_clash_edges_per_graph: int = 128,
    ) -> None:
        super().__init__()
        self.prior = prior
        for parameter in self.prior.parameters():
            parameter.requires_grad_(False)
        self.jacobian_config = (
            jacobian_config
            if isinstance(jacobian_config, JacobianBACConfig)
            else JacobianBACConfig.from_mapping(jacobian_config)
        )
        self.integration_step_size = float(integration_step_size)
        self.angle_max_graph_rms = float(angle_max_graph_rms)
        self.angle_max_atom = float(angle_max_atom)
        self.clash_max_graph_rms = float(clash_max_graph_rms)
        self.clash_max_atom = float(clash_max_atom)
        self.clash_cutoff = float(clash_cutoff)
        self.clash_allowed_contact = float(clash_allowed_contact)
        self.clash_exclude_topology_distance = int(clash_exclude_topology_distance)
        self.max_clash_edges_per_graph = int(max_clash_edges_per_graph)
        positive = {
            "integration_step_size": self.integration_step_size,
            "angle_max_graph_rms": self.angle_max_graph_rms,
            "angle_max_atom": self.angle_max_atom,
            "clash_max_graph_rms": self.clash_max_graph_rms,
            "clash_max_atom": self.clash_max_atom,
            "clash_cutoff": self.clash_cutoff,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"V7 trust settings must be positive: {positive}")
        if self.clash_allowed_contact < 0:
            raise ValueError("clash_allowed_contact must be nonnegative")
        self.trace_enabled = True
        self.reset_statistics()

    def train(self, mode: bool = True) -> "MCVRConstraintSpecificHybrid":
        super().train(mode)
        self.prior.eval()
        return self

    def reset_statistics(self) -> None:
        self._angle_solver_trace: list[dict[str, Any]] = []
        self._component_trace: list[dict[str, Any]] = []

    def angle_solver_trace(self) -> list[dict[str, Any]]:
        return list(self._angle_solver_trace)

    def component_trace(self) -> list[dict[str, Any]]:
        return list(self._component_trace)

    def angle_solver_summary(self) -> dict[str, Any]:
        statuses = Counter(row["solver_status"] for row in self._angle_solver_trace)
        solved = [row for row in self._angle_solver_trace if row["solver_status"] == "SOLVED"]
        inactive = statuses.get("NO_ACTIVE_CONSTRAINT", 0)
        failures = len(self._angle_solver_trace) - len(solved) - inactive
        return {
            "calls": len(self._angle_solver_trace),
            "status_counts": dict(statuses),
            "inactive_constraint_calls": inactive,
            "solver_failure_count": failures,
            "solver_failure_rate": failures / max(len(self._angle_solver_trace), 1),
            "effective_rank_mean": (
                float(sum(row["effective_rank"] for row in solved) / len(solved))
                if solved
                else 0.0
            ),
            "condition_number_mean": (
                float(sum(row["condition_number"] for row in solved) / len(solved))
                if solved
                else 0.0
            ),
            "condition_number_max": float(
                max((row["condition_number"] for row in solved), default=0.0)
            ),
            "singular_value_max": float(
                max((row["singular_value_max"] for row in solved), default=0.0)
            ),
            "singular_value_min_retained": float(
                min(
                    (
                        row["singular_value_min_retained"]
                        for row in solved
                        if row["singular_value_min_retained"] > 0.0
                    ),
                    default=0.0,
                )
            ),
            "truncated_direction_count": int(
                sum(row["truncated_direction_count"] for row in solved)
            ),
        }

    def component_summary(self) -> dict[str, Any]:
        if not self._component_trace:
            return {"calls": 0}
        numeric = (
            "bond_alpha",
            "angle_alpha",
            "clash_alpha",
            "fusion_alpha",
            "bond_rms",
            "angle_rms",
            "clash_rms",
            "fused_rms",
            "active_angle_count",
            "active_clash_count",
            "degenerate_clash_count",
        )
        return {
            "calls": len(self._component_trace),
            **{
                name: float(
                    sum(float(row[name]) for row in self._component_trace)
                    / len(self._component_trace)
                )
                for name in numeric
            },
        }

    def _graph_constraints(
        self,
        batch: Any,
        left: int,
        right: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        bonds = torch.as_tensor(
            field(batch, "active_bond_constraint_index"), device=device, dtype=torch.long
        ).reshape(2, -1)
        bond_ranges = torch.as_tensor(
            field(batch, "bond_allowed_range"), device=device, dtype=torch.float64
        ).reshape(-1, 3)
        bond_mask = (
            (bonds[0] >= left)
            & (bonds[0] < right)
            & (bonds[1] >= left)
            & (bonds[1] < right)
        )
        graph_bonds = bonds[:, bond_mask] - left
        graph_bond_ranges = bond_ranges[bond_mask]
        angles = torch.as_tensor(
            field(batch, "active_angle_constraint_index"), device=device, dtype=torch.long
        )
        if angles.ndim == 2 and angles.size(0) == 3:
            angles = angles.t()
        angles = angles.reshape(-1, 3)
        angle_ranges = torch.as_tensor(
            field(batch, "angle_allowed_range"), device=device, dtype=torch.float64
        ).reshape(-1, 3)
        angle_mask = (angles[:, 1] >= left) & (angles[:, 1] < right)
        return (
            graph_bonds,
            graph_bond_ranges,
            angles[angle_mask] - left,
            angle_ranges[angle_mask],
        )

    def _angle_component(
        self,
        coordinates: Tensor,
        angles: Tensor,
        angle_ranges: Tensor,
    ) -> tuple[Tensor, dict[str, Any]]:
        system = build_angle_constraint_system(
            coordinates, angles, angle_ranges, self.jacobian_config
        )
        flat, solve = solve_damped_system(
            system, coordinates.size(0), self.jacobian_config
        )
        correction = coordinates.new_zeros(coordinates.shape)
        status = str(solve["solver_status"])
        if status == "SOLVED":
            correction, _ = remove_rigid_update(coordinates, flat.reshape_as(coordinates))
            projected_after = float(
                (
                    system.weights
                    * (system.residual + system.jacobian @ correction.reshape(-1)).square()
                ).sum()
            )
            if float(system.objective) - projected_after <= 0.0:
                correction = torch.zeros_like(correction)
                status = "RIGID_PROJECTION_REMOVED_REDUCTION"
        if not bool(torch.isfinite(correction).all()):
            correction = torch.zeros_like(correction)
            status = "NONFINITE_PROJECTED_UPDATE"
        row = {
            "solver_status": status,
            "effective_rank": int(solve["effective_rank"]),
            "singular_value_max": float(solve["singular_value_max"]),
            "singular_value_min_retained": float(solve["singular_value_min_retained"]),
            "condition_number": float(solve["condition_number"]),
            "truncated_direction_count": int(solve["truncated_direction_count"]),
            "constraint_count": int(system.residual.numel()),
            "active_angle_count": int(system.counts["active_angle"]),
            **system.diagnostics,
        }
        return correction, row

    def _clash_component(
        self,
        coordinates: Tensor,
        bonds: Tensor,
    ) -> tuple[Tensor, dict[str, Any]]:
        clash = sparse_clash_edges(
            coordinates,
            bonds,
            cutoff=self.clash_cutoff,
            allowed_contact=self.clash_allowed_contact,
            exclude_topology_distance=self.clash_exclude_topology_distance,
            max_edges_per_graph=self.max_clash_edges_per_graph,
        )
        correction = torch.zeros_like(coordinates)
        counts = coordinates.new_zeros(coordinates.size(0))
        active = clash["active_mask"]
        degenerate = active & (clash["distance"] <= self.jacobian_config.minimum_distance)
        valid = active & ~degenerate
        edges = clash["edge_index"][:, valid]
        if edges.numel():
            left, right = edges
            magnitude = 0.5 * clash["penetration"][valid]
            vectors = magnitude[:, None] * clash["direction"][valid]
            correction.index_add_(0, left, vectors)
            correction.index_add_(0, right, -vectors)
            ones = torch.ones_like(magnitude)
            counts.index_add_(0, left, ones)
            counts.index_add_(0, right, ones)
            correction = correction / counts.clamp_min(1.0)[:, None]
            correction, _ = remove_rigid_update(coordinates, correction)
        if not bool(torch.isfinite(correction).all()):
            correction = torch.zeros_like(correction)
        return correction, {
            "candidate_clash_count": int(clash["edge_index"].size(1)),
            "active_clash_count": int(active.sum()),
            "degenerate_clash_count": int(degenerate.sum()),
        }

    def forward(
        self,
        batch: Any,
        pos: Tensor,
        t: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        with torch.no_grad():
            base = self.prior(batch, pos, t, **kwargs)
        ptr = torch.as_tensor(field(batch, "ptr"), device=pos.device, dtype=torch.long)
        atom_batch = _atom_batch(batch, pos)
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        bond_limit_rms = self.integration_step_size * float(
            self.prior.max_velocity_graph_rms
        )
        bond_limit_atom = self.integration_step_size * float(
            self.prior.max_velocity_atom_norm
        )
        raw_components = []
        trusted_components = []
        final_components = []
        bond_components = []
        angle_components = []
        clash_components = []
        bond_alphas = []
        angle_alphas = []
        clash_alphas = []
        fusion_alphas = []
        angle_failures = []
        for graph_index in range(graphs):
            left = int(ptr[graph_index])
            right = int(ptr[graph_index + 1])
            bonds, _, angles, angle_ranges = self._graph_constraints(
                batch, left, right, pos.device
            )
            local = pos[left:right].detach().to(torch.float64)
            bond_raw = (
                self.integration_step_size * base["v_raw"][left:right].detach()
            ).to(torch.float64)
            angle_raw, solver_row = self._angle_component(local, angles, angle_ranges)
            clash_raw, clash_row = self._clash_component(local, bonds)
            bond_step, bond_alpha, _, bond_rms = _trust_normalize(
                bond_raw,
                max_graph_rms=bond_limit_rms,
                max_atom=bond_limit_atom,
            )
            angle_step, angle_alpha, _, angle_rms = _trust_normalize(
                angle_raw,
                max_graph_rms=self.angle_max_graph_rms,
                max_atom=self.angle_max_atom,
            )
            clash_step, clash_alpha, _, clash_rms = _trust_normalize(
                clash_raw,
                max_graph_rms=self.clash_max_graph_rms,
                max_atom=self.clash_max_atom,
            )
            fused_raw = bond_step + angle_step + clash_step
            fused, fusion_alpha, _, fused_rms = _trust_normalize(
                fused_raw,
                max_graph_rms=bond_limit_rms,
                max_atom=bond_limit_atom,
            )
            finite = bool(torch.isfinite(fused).all())
            if not finite:
                fused = torch.zeros_like(fused)
                fusion_alpha = 0.0
            raw_components.append(fused_raw.to(pos.dtype))
            trusted_components.append(fused.to(pos.dtype))
            bond_components.append(bond_step.to(pos.dtype))
            angle_components.append(angle_step.to(pos.dtype))
            clash_components.append(clash_step.to(pos.dtype))
            final_components.append(
                (
                    base["global_safety_gate"][graph_index]
                    * fused.to(pos.dtype)
                    / self.integration_step_size
                )
            )
            bond_alphas.append(bond_alpha)
            angle_alphas.append(angle_alpha)
            clash_alphas.append(clash_alpha)
            fusion_alphas.append(fusion_alpha)
            failure = solver_row["solver_status"] not in {
                "SOLVED",
                "NO_ACTIVE_CONSTRAINT",
            }
            angle_failures.append(float(failure))
            if self.trace_enabled:
                self._angle_solver_trace.append(
                    {
                        "graph_call": len(self._angle_solver_trace),
                        **solver_row,
                    }
                )
                self._component_trace.append(
                    {
                        "graph_call": len(self._component_trace),
                        "bond_alpha": bond_alpha,
                        "angle_alpha": angle_alpha,
                        "clash_alpha": clash_alpha,
                        "fusion_alpha": fusion_alpha,
                        "bond_rms": bond_rms,
                        "angle_rms": angle_rms,
                        "clash_rms": clash_rms,
                        "fused_rms": fused_rms,
                        "active_angle_count": solver_row["active_angle_count"],
                        **clash_row,
                    }
                )
        v_raw = torch.cat(raw_components, dim=0) / self.integration_step_size
        v_trust = torch.cat(trusted_components, dim=0) / self.integration_step_size
        v_final = torch.cat(final_components, dim=0)
        return {
            **base,
            "v_bond_cartesian": base["v_raw"],
            "v_bond_cartesian_coordinate": torch.cat(bond_components, dim=0),
            "v_angle_jacobian_coordinate": torch.cat(angle_components, dim=0),
            "v_clash_repulsion_coordinate": torch.cat(clash_components, dim=0),
            "constraint_alpha_bond": pos.new_tensor(bond_alphas),
            "constraint_alpha_angle": pos.new_tensor(angle_alphas),
            "constraint_alpha_clash": pos.new_tensor(clash_alphas),
            "constraint_alpha_fusion": pos.new_tensor(fusion_alphas),
            "angle_gate": pos.new_tensor(angle_alphas),
            "clash_gate": pos.new_tensor(clash_alphas),
            "angle_solver_failure": pos.new_tensor(angle_failures),
            "v_raw": v_raw,
            "v_trust_clipped": v_trust,
            "v_final": v_final,
            "velocity": v_final,
        }
