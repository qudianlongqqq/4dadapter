"""Constraint-space prototypes built around the frozen D1 Cartesian prior."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from etflow.models.components.light_egnn_refiner import _mlp

from .audit import field
from .bac_constraints import standardized_interval_residual
from .bac_jacobian import (
    JacobianBACConfig,
    build_constraint_system,
    remove_rigid_update,
    solve_damped_system,
)
from .model import _atom_batch
from .mvr_model import MCVRModel, trust_clip_velocity
from .mvr_v2_bac import (
    MCVRBACModel,
    V2_D_BOND_ANGLE_CLASH,
    _scatter_constraint_vectors,
    _zero_last,
)


V5_SCHEMA_VERSION = "mcvr-v5-constraint-hybrid-v1"


def _component_clip(
    value: Tensor,
    atom_batch: Tensor,
    *,
    max_graph_rms: float,
) -> Tensor:
    graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
    energy = value.new_zeros(graphs)
    energy.index_add_(0, atom_batch, value.square().sum(-1))
    counts = torch.bincount(atom_batch, minlength=graphs).clamp_min(1).to(value.dtype)
    rms = torch.sqrt(energy / counts + 1.0e-12)
    scale = torch.clamp(float(max_graph_rms) / rms.clamp_min(1.0e-12), max=1.0)
    return value * scale[atom_batch, None]


class MCVRConstraintMultiHeadModel(MCVRBACModel):
    """Prototype A: normalized Bond/Angle/Clash Cartesian correction heads."""

    def __init__(
        self,
        *args: Any,
        component_max_graph_rms: float = 0.02,
        **kwargs: Any,
    ) -> None:
        kwargs["bac_mode"] = V2_D_BOND_ANGLE_CLASH
        super().__init__(*args, **kwargs)
        self.component_max_graph_rms = float(component_max_graph_rms)
        if self.component_max_graph_rms <= 0.0:
            raise ValueError("component_max_graph_rms must be positive")
        hidden_dim = int(self.backbone.atom_embedding.out_features)
        edge_hidden_dim = int(self.backbone.layers[0].message_mlp[0].out_features)
        self.bond_constraint_encoder = _mlp(2 * hidden_dim + 3, edge_hidden_dim, hidden_dim, 0.0)
        self.bond_constraint_head = _mlp(hidden_dim, edge_hidden_dim, 3, 0.0)
        _zero_last(self.bond_constraint_head)
        # Three allocation logits, one activity gate, and one trust gate.
        self.multihead_fusion = _mlp(hidden_dim + 7, hidden_dim, 5, 0.0)
        _zero_last(self.multihead_fusion)
        with torch.no_grad():
            self.multihead_fusion[-1].bias[3] = -2.0
        # V5 replaces the V2 independent-sigmoid fusion surface.
        del self.constraint_fusion
        del self.constraint_type_embedding

    def _bond_branch(
        self, batch: Any, pos: Tensor, hidden: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        bonds = field(batch, "active_bond_constraint_index")
        ranges = field(batch, "bond_allowed_range")
        if bonds is None or ranges is None:
            return torch.zeros_like(pos), {
                "bond_constraint_strength": pos.new_empty(0),
                "bond_constraint_confidence": pos.new_empty(0),
                "bond_constraint_gate": pos.new_empty(0),
            }
        bonds = torch.as_tensor(bonds, device=pos.device, dtype=torch.long).reshape(2, -1)
        ranges = torch.as_tensor(ranges, device=pos.device, dtype=pos.dtype).reshape(-1, 3)
        if not bonds.numel():
            return torch.zeros_like(pos), {
                "bond_constraint_strength": pos.new_empty(0),
                "bond_constraint_confidence": pos.new_empty(0),
                "bond_constraint_gate": pos.new_empty(0),
            }
        left, right = bonds
        relative = pos[left] - pos[right]
        distance = torch.linalg.vector_norm(relative, dim=-1)
        direction = relative / distance.clamp_min(1.0e-8)[:, None]
        direction = torch.where(
            (distance > 1.0e-8)[:, None], direction, torch.zeros_like(direction)
        )
        residual, severity = standardized_interval_residual(distance, ranges)
        features = torch.cat(
            [
                hidden[left] + hidden[right],
                (hidden[left] - hidden[right]).abs(),
                distance[:, None],
                residual[:, None],
                severity[:, None],
            ],
            dim=-1,
        )
        raw = self.bond_constraint_head(self.bond_constraint_encoder(features))
        strength = torch.tanh(raw[:, 0])
        confidence = torch.sigmoid(raw[:, 1])
        gate = torch.sigmoid(raw[:, 2])
        active = severity > 0
        weights = self.bac_constraint_scale * strength * confidence * gate * active.to(pos.dtype)
        correction = _scatter_constraint_vectors(
            pos.size(0),
            bonds.t(),
            (direction, -direction),
            weights,
            pos,
            count_mask=active,
        )
        return correction, {
            "bond_constraint_strength": strength,
            "bond_constraint_confidence": confidence,
            "bond_constraint_gate": gate,
            "bond_constraint_residual": residual,
        }

    def forward(
        self,
        batch: Any,
        pos: Tensor,
        t: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        base = MCVRModel.forward(self, batch, pos, t, **kwargs)
        hidden = base["node_embedding"]
        atom_batch = base["atom_batch"]
        bond, bond_diag = self._bond_branch(batch, pos, hidden)
        angle, angle_diag = self._angle_branch(batch, pos, hidden)
        clash, clash_diag = self._clash_branch(batch, pos, hidden, atom_batch)
        components = [
            _component_clip(value, atom_batch, max_graph_rms=self.component_max_graph_rms)
            for value in (bond, angle, clash)
        ]
        component_norms = torch.stack(
            [torch.linalg.vector_norm(value, dim=-1) for value in components], dim=-1
        )
        mode_mask = field(batch, "active_mode_mask")
        if mode_mask is None:
            active = component_norms > 1.0e-12
        else:
            graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
            mode_mask = torch.as_tensor(mode_mask, device=pos.device, dtype=pos.dtype).reshape(
                graphs, 6
            )
            active_graph = torch.stack([mode_mask[:, 0], mode_mask[:, 1], mode_mask[:, 3]], dim=-1)
            active = active_graph[atom_batch] > 0
        fusion_features = torch.cat(
            [
                hidden,
                torch.linalg.vector_norm(base["v_raw"], dim=-1, keepdim=True),
                component_norms,
                active.to(pos.dtype),
            ],
            dim=-1,
        )
        fusion = self.multihead_fusion(fusion_features)
        logits = fusion[:, :3].masked_fill(~active, -1.0e4)
        allocation = torch.softmax(logits, dim=-1) * active.to(pos.dtype)
        allocation = allocation / allocation.sum(-1, keepdim=True).clamp_min(1.0)
        activity_gate = torch.sigmoid(fusion[:, 3:4])
        trust_gate = torch.sigmoid(fusion[:, 4:5])
        constraint_field = (
            activity_gate
            * trust_gate
            * sum(
                allocation[:, index : index + 1] * value for index, value in enumerate(components)
            )
        )
        unified_raw = base["v_raw"] + constraint_field
        unified_clipped = trust_clip_velocity(
            unified_raw,
            atom_batch,
            max_atom_norm=self.max_velocity_atom_norm,
            max_graph_rms=self.max_velocity_graph_rms,
        )
        unified_final = base["global_safety_gate"][atom_batch] * unified_clipped
        if not bool(torch.isfinite(unified_final).all()):
            unified_final = torch.zeros_like(unified_final)
        return {
            **base,
            **bond_diag,
            **angle_diag,
            **clash_diag,
            "v_bond_component": components[0],
            "v_angle_component": components[1],
            "v_clash_component": components[2],
            "constraint_component_norms": component_norms,
            "constraint_component_active": active.to(pos.dtype),
            "constraint_allocation": allocation,
            "constraint_activity_gate": activity_gate,
            "constraint_trust_gate": trust_gate,
            "v_constraint_fused": constraint_field,
            "v_raw": unified_raw,
            "v_trust_clipped": unified_clipped,
            "v_final": unified_final,
            "velocity": unified_final,
            "unified_delta_count": pos.new_tensor(1, dtype=torch.long),
        }

    def load_d1b_state_dict(
        self, state_dict: dict[str, Tensor], *, strict: bool = True
    ) -> tuple[list[str], list[str]]:
        current = self.state_dict()
        current_keys = set(current)
        checkpoint_keys = set(state_dict)
        new_prefixes = (
            "angle_constraint_",
            "clash_constraint_",
            "bond_constraint_",
            "multihead_fusion.",
        )
        missing_base = sorted(
            key for key in current_keys - checkpoint_keys if not key.startswith(new_prefixes)
        )
        unexpected = sorted(checkpoint_keys - current_keys)
        if strict and (missing_base or unexpected):
            raise RuntimeError(
                f"D1-B base state mismatch: missing={missing_base}, unexpected={unexpected}"
            )
        current.update({key: value for key, value in state_dict.items() if key in current_keys})
        self.load_state_dict(current, strict=True)
        return missing_base, unexpected


def _trust_scale_coordinate(
    update: Tensor, *, max_graph_rms: float, max_atom: float
) -> tuple[Tensor, float]:
    if not update.numel():
        return update, 1.0
    graph_rms = float(torch.sqrt(update.square().sum(-1).mean()))
    atom_max = float(torch.linalg.vector_norm(update, dim=-1).max())
    scale = min(
        1.0,
        float(max_graph_rms) / max(graph_rms, 1.0e-30),
        float(max_atom) / max(atom_max, 1.0e-30),
    )
    return update * scale, scale


class MCVRNeuralJacobianHybrid(nn.Module):
    """Prototype B: frozen neural prior plus bounded analytic BAC correction."""

    def __init__(
        self,
        prior: MCVRBACModel,
        *,
        jacobian_config: JacobianBACConfig | Mapping[str, Any] | None = None,
        correction_lambda: float = 1.0,
        integration_step_size: float = 0.25,
        max_correction_graph_rms: float = 0.01,
        max_correction_atom: float = 0.02,
    ) -> None:
        super().__init__()
        self.prior = prior
        self.jacobian_config = (
            jacobian_config
            if isinstance(jacobian_config, JacobianBACConfig)
            else JacobianBACConfig.from_mapping(jacobian_config)
        )
        self.jacobian_config.validate()
        self.correction_lambda = float(correction_lambda)
        self.integration_step_size = float(integration_step_size)
        self.max_correction_graph_rms = float(max_correction_graph_rms)
        self.max_correction_atom = float(max_correction_atom)
        if not 0.0 <= self.correction_lambda <= 1.0:
            raise ValueError("correction_lambda must be in [0, 1]")
        if self.integration_step_size <= 0.0:
            raise ValueError("integration_step_size must be positive")
        if self.max_correction_graph_rms <= 0.0 or self.max_correction_atom <= 0.0:
            raise ValueError("correction trust limits must be positive")
        self.reset_solver_statistics()

    @property
    def max_velocity_atom_norm(self) -> float:
        return float(self.prior.max_velocity_atom_norm)

    @property
    def max_velocity_graph_rms(self) -> float:
        return float(self.prior.max_velocity_graph_rms)

    def reset_solver_statistics(self) -> None:
        self._solver_trace: list[dict[str, Any]] = []

    def solver_trace(self) -> list[dict[str, Any]]:
        return list(self._solver_trace)

    def solver_summary(self) -> dict[str, Any]:
        statuses = Counter(row["solver_status"] for row in self._solver_trace)
        solved = [row for row in self._solver_trace if row["solver_status"] == "SOLVED"]
        inactive = statuses.get("NO_ACTIVE_CONSTRAINT", 0)
        failures = len(self._solver_trace) - len(solved) - inactive
        condition = [float(row["condition_number"]) for row in solved]
        return {
            "calls": len(self._solver_trace),
            "status_counts": dict(statuses),
            "inactive_constraint_calls": inactive,
            "solver_failure_count": failures,
            "solver_failure_rate": (
                failures / len(self._solver_trace)
                if self._solver_trace
                else 0.0
            ),
            "effective_rank_mean": (
                float(sum(row["effective_rank"] for row in solved) / len(solved)) if solved else 0.0
            ),
            "condition_number_mean": float(sum(condition) / len(condition)) if condition else 0.0,
            "condition_number_max": float(max(condition)) if condition else 0.0,
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

    def _graph_constraints(
        self,
        batch: Any,
        graph_index: int,
        left: int,
        right: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        bonds = torch.as_tensor(
            field(batch, "active_bond_constraint_index"),
            device=device,
            dtype=torch.long,
        ).reshape(2, -1)
        bond_ranges = torch.as_tensor(
            field(batch, "bond_allowed_range"), device=device, dtype=torch.float64
        ).reshape(-1, 3)
        bond_mask = (
            (bonds[0] >= left) & (bonds[0] < right) & (bonds[1] >= left) & (bonds[1] < right)
        )
        graph_bonds = bonds[:, bond_mask] - left
        graph_bond_ranges = bond_ranges[bond_mask]
        angles = torch.as_tensor(
            field(batch, "active_angle_constraint_index"),
            device=device,
            dtype=torch.long,
        )
        if angles.ndim == 2 and angles.size(0) == 3:
            angles = angles.t()
        angles = angles.reshape(-1, 3)
        angle_ranges = torch.as_tensor(
            field(batch, "angle_allowed_range"), device=device, dtype=torch.float64
        ).reshape(-1, 3)
        angle_mask = (angles[:, 1] >= left) & (angles[:, 1] < right)
        graph_angles = angles[angle_mask] - left
        graph_angle_ranges = angle_ranges[angle_mask]
        del graph_index
        return graph_bonds, graph_bond_ranges, graph_angles, graph_angle_ranges

    @torch.inference_mode()
    def forward(
        self,
        batch: Any,
        pos: Tensor,
        t: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        base = self.prior(batch, pos, t, **kwargs)
        atom_batch = _atom_batch(batch, pos)
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        ptr = torch.as_tensor(field(batch, "ptr"), device=pos.device, dtype=torch.long)
        prior_coordinate = pos + self.integration_step_size * base["v_final"]
        geometric_velocity = torch.zeros_like(pos)
        rank = pos.new_zeros(graphs)
        condition = pos.new_zeros(graphs)
        solver_failure = pos.new_zeros(graphs)
        correction_scale = pos.new_zeros(graphs)
        for graph_index in range(graphs):
            left = int(ptr[graph_index])
            right = int(ptr[graph_index + 1])
            bonds, bond_ranges, angles, angle_ranges = self._graph_constraints(
                batch, graph_index, left, right, pos.device
            )
            local = prior_coordinate[left:right].detach().to(torch.float64)
            system = build_constraint_system(
                local,
                bonds,
                bond_ranges,
                angles,
                angle_ranges,
                self.jacobian_config,
            )
            flat, diagnostics = solve_damped_system(system, local.size(0), self.jacobian_config)
            trace = {
                "graph_call": len(self._solver_trace),
                "solver_status": diagnostics["solver_status"],
                "effective_rank": int(diagnostics["effective_rank"]),
                "singular_value_max": float(diagnostics["singular_value_max"]),
                "singular_value_min_retained": float(diagnostics["singular_value_min_retained"]),
                "condition_number": float(diagnostics["condition_number"]),
                "truncated_direction_count": int(diagnostics["truncated_direction_count"]),
                "constraint_count": int(system.residual.numel()),
            }
            self._solver_trace.append(trace)
            rank[graph_index] = trace["effective_rank"]
            condition[graph_index] = min(
                trace["condition_number"] if math.isfinite(trace["condition_number"]) else 0.0,
                torch.finfo(pos.dtype).max,
            )
            if diagnostics["solver_status"] != "SOLVED":
                solver_failure[graph_index] = 1.0
                continue
            correction, _ = remove_rigid_update(local, flat.reshape_as(local))
            correction, scale = _trust_scale_coordinate(
                correction,
                max_graph_rms=self.max_correction_graph_rms,
                max_atom=self.max_correction_atom,
            )
            correction_scale[graph_index] = scale
            if not bool(torch.isfinite(correction).all()):
                solver_failure[graph_index] = 1.0
                self._solver_trace[-1]["solver_status"] = "NONFINITE_PROJECTED_UPDATE"
                continue
            geometric_velocity[left:right] = correction.to(pos.dtype) / self.integration_step_size
        combined_raw = base["v_raw"] + self.correction_lambda * geometric_velocity
        combined_clipped = trust_clip_velocity(
            combined_raw,
            atom_batch,
            max_atom_norm=self.max_velocity_atom_norm,
            max_graph_rms=self.max_velocity_graph_rms,
        )
        combined_final = base["global_safety_gate"][atom_batch] * combined_clipped
        if not bool(torch.isfinite(combined_final).all()):
            combined_final = base["v_final"]
        return {
            **base,
            "v_neural_prior": base["v_final"],
            "v_jacobian_geometry": geometric_velocity,
            "jacobian_effective_rank": rank,
            "jacobian_condition_number": condition,
            "jacobian_solver_failure": solver_failure,
            "jacobian_correction_scale": correction_scale,
            "v_raw": combined_raw,
            "v_trust_clipped": combined_clipped,
            "v_final": combined_final,
            "velocity": combined_final,
        }
