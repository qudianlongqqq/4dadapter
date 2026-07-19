"""Constraint-specialized supervision for MCVR V5 Prototype A."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .audit import field
from .bac_constraints import sparse_clash_edges, standardized_interval_residual
from .geometry import bond_angles, bond_lengths
from .model import _atom_batch
from .mvr_v2_bac_loss import MCVRBACLoss


DEFAULT_V5_HEAD_WEIGHTS = {
    "bond_specialization": 0.5,
    "angle_specialization": 0.5,
    "clash_specialization": 0.5,
    "cross_preservation": 0.25,
    "fusion_assignment": 0.1,
}


def _graph_mean(values: Tensor, graph_index: Tensor, graphs: int) -> Tensor:
    if not values.numel():
        return values.new_zeros(graphs)
    total = values.new_zeros(graphs)
    total.index_add_(0, graph_index, values)
    count = torch.bincount(graph_index, minlength=graphs).clamp_min(1).to(values.dtype)
    return total / count


def _active_graph_mean(values: Tensor, graph_index: Tensor, active: Tensor, graphs: int) -> Tensor:
    if not values.numel() or not bool(active.any()):
        return values.new_zeros(graphs)
    return _graph_mean(values[active], graph_index[active], graphs)


class MCVRConstraintMultiHeadLoss(nn.Module):
    """Unified BAC loss plus branch-specific geometric supervision."""

    def __init__(
        self,
        base_weights: Mapping[str, float] | None = None,
        bac_weights: Mapping[str, float] | None = None,
        head_weights: Mapping[str, float] | None = None,
        *,
        proposal_time: float = 1.0,
        proposal_step_size: float = 0.25,
    ) -> None:
        super().__init__()
        self.base = MCVRBACLoss(
            base_weights,
            bac_weights,
            proposal_mode="full_inference_field",
            proposal_time=proposal_time,
            proposal_step_size=proposal_step_size,
        )
        self.weights = {**DEFAULT_V5_HEAD_WEIGHTS, **dict(head_weights or {})}
        self.proposal_time = float(proposal_time)
        self.proposal_step_size = float(proposal_step_size)

    def forward(self, model: nn.Module, batch: Any) -> dict[str, Tensor]:
        base = self.base(model, batch)
        x_input = torch.as_tensor(field(batch, "x_input", field(batch, "x_init")))
        x_target = torch.as_tensor(
            field(batch, "x_target"), device=x_input.device, dtype=x_input.dtype
        )
        atom_batch = _atom_batch(batch, x_input)
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        output = model(batch, x_input, x_input.new_full((graphs,), self.proposal_time))
        proposals = {
            name: x_input + self.proposal_step_size * output[f"v_{name}_component"]
            for name in ("bond", "angle", "clash")
        }
        active_mask = torch.as_tensor(
            field(batch, "active_mode_mask"),
            device=x_input.device,
            dtype=x_input.dtype,
        ).reshape(graphs, 6)
        bonds = torch.as_tensor(
            field(batch, "active_bond_constraint_index"),
            device=x_input.device,
            dtype=torch.long,
        ).reshape(2, -1)
        bond_ranges = torch.as_tensor(
            field(batch, "bond_allowed_range"),
            device=x_input.device,
            dtype=x_input.dtype,
        ).reshape(-1, 3)
        angles = torch.as_tensor(
            field(batch, "active_angle_constraint_index"),
            device=x_input.device,
            dtype=torch.long,
        )
        if angles.ndim == 2 and angles.size(0) == 3:
            angles = angles.t()
        angles = angles.reshape(-1, 3)
        angle_ranges = torch.as_tensor(
            field(batch, "angle_allowed_range"),
            device=x_input.device,
            dtype=x_input.dtype,
        ).reshape(-1, 3)
        bond_graph = atom_batch[bonds[0]] if bonds.numel() else atom_batch.new_empty(0)
        angle_graph = atom_batch[angles[:, 1]] if angles.numel() else atom_batch.new_empty(0)
        input_bond, input_bond_severity = standardized_interval_residual(
            bond_lengths(x_input, bonds), bond_ranges
        )
        target_bond, _ = standardized_interval_residual(bond_lengths(x_target, bonds), bond_ranges)
        input_angle, input_angle_severity = standardized_interval_residual(
            bond_angles(x_input, angles), angle_ranges
        )
        target_angle, _ = standardized_interval_residual(
            bond_angles(x_target, angles), angle_ranges
        )
        bond_active = (input_bond_severity > 0) & (active_mask[bond_graph, 0] > 0)
        angle_active = (input_angle_severity > 0) & (active_mask[angle_graph, 1] > 0)
        bond_proposal, bond_proposal_severity = standardized_interval_residual(
            bond_lengths(proposals["bond"], bonds), bond_ranges
        )
        angle_proposal, angle_proposal_severity = standardized_interval_residual(
            bond_angles(proposals["angle"], angles), angle_ranges
        )
        bond_specialization = _active_graph_mean(
            F.smooth_l1_loss(bond_proposal, target_bond, reduction="none"),
            bond_graph,
            bond_active,
            graphs,
        ).mean()
        angle_specialization = _active_graph_mean(
            F.smooth_l1_loss(angle_proposal, target_angle, reduction="none"),
            angle_graph,
            angle_active,
            graphs,
        ).mean()
        clash = sparse_clash_edges(
            proposals["clash"],
            bonds,
            atom_batch=atom_batch,
            cutoff=float(model.clash_cutoff),
            allowed_contact=float(model.clash_allowed_contact),
            exclude_topology_distance=int(model.clash_exclude_topology_distance),
            max_edges_per_graph=int(model.max_clash_edges_per_graph),
        )
        clash_values = (
            clash["penetration"] / max(float(model.clash_allowed_contact), 1.0e-8)
        ).square()
        clash_specialization = (
            _graph_mean(clash_values, clash["graph_index"], graphs)
            * (active_mask[:, 3] > 0).to(x_input.dtype)
        ).mean()
        angle_under_bond, angle_under_bond_severity = standardized_interval_residual(
            bond_angles(proposals["bond"], angles), angle_ranges
        )
        bond_under_angle, bond_under_angle_severity = standardized_interval_residual(
            bond_lengths(proposals["angle"], bonds), bond_ranges
        )
        bond_under_clash, bond_under_clash_severity = standardized_interval_residual(
            bond_lengths(proposals["clash"], bonds), bond_ranges
        )
        cross_preservation = (
            _graph_mean(
                (angle_under_bond_severity - input_angle_severity).clamp_min(0.0),
                angle_graph,
                graphs,
            )
            + _graph_mean(
                (bond_under_angle_severity - input_bond_severity).clamp_min(0.0),
                bond_graph,
                graphs,
            )
            + _graph_mean(
                (bond_under_clash_severity - input_bond_severity).clamp_min(0.0),
                bond_graph,
                graphs,
            )
        ).mean()
        del input_bond, input_angle, angle_under_bond, bond_under_angle, bond_under_clash
        desired = torch.stack([active_mask[:, 0], active_mask[:, 1], active_mask[:, 3]], dim=-1)
        desired = desired / desired.sum(-1, keepdim=True).clamp_min(1.0)
        allocation = output["constraint_allocation"]
        allocation_graph = allocation.new_zeros((graphs, 3))
        allocation_graph.index_add_(0, atom_batch, allocation)
        atom_count = torch.bincount(atom_batch, minlength=graphs).clamp_min(1).to(x_input.dtype)
        allocation_graph = allocation_graph / atom_count[:, None]
        active_graph = desired.sum(-1) > 0
        fusion_assignment = (
            F.smooth_l1_loss(allocation_graph[active_graph], desired[active_graph])
            if bool(active_graph.any())
            else x_input.new_zeros(())
        )
        terms = {
            "v5_bond_specialization_loss": bond_specialization,
            "v5_angle_specialization_loss": angle_specialization,
            "v5_clash_specialization_loss": clash_specialization,
            "v5_cross_preservation_loss": cross_preservation,
            "v5_fusion_assignment_loss": fusion_assignment,
        }
        weighted = {
            "weighted_v5_bond_specialization": self.weights["bond_specialization"]
            * bond_specialization,
            "weighted_v5_angle_specialization": self.weights["angle_specialization"]
            * angle_specialization,
            "weighted_v5_clash_specialization": self.weights["clash_specialization"]
            * clash_specialization,
            "weighted_v5_cross_preservation": self.weights["cross_preservation"]
            * cross_preservation,
            "weighted_v5_fusion_assignment": self.weights["fusion_assignment"] * fusion_assignment,
        }
        del bond_proposal_severity, angle_proposal_severity
        return {**base, **terms, **weighted, "loss": base["loss"] + sum(weighted.values())}
