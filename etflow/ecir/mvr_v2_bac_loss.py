"""Per-record normalized losses for unified BAC refinement."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .audit import field
from .bac_constraints import (
    sparse_clash_edges,
    standardized_interval_residual,
)
from .geometry import bond_angles, bond_lengths
from .model import _atom_batch
from .mvr_loss import MCVRLoss


DEFAULT_BAC_LOSS_WEIGHTS = {
    "coordinate_target": 0.0,
    "bond_residual": 1.0,
    "angle_residual": 1.0,
    "clash_penetration": 1.0,
    "zero_error_noop": 0.25,
    "preservation": 0.25,
    "no_new_violation": 0.5,
    "confidence": 0.05,
    "gate": 0.05,
}


def _per_graph_mean(values: Tensor, graph_index: Tensor, graphs: int) -> Tensor:
    if not values.numel():
        return values.new_zeros(graphs)
    totals = values.new_zeros(graphs)
    totals.index_add_(0, graph_index, values)
    counts = torch.bincount(graph_index, minlength=graphs).to(values.dtype)
    return totals / counts.clamp_min(1.0)


def _masked_graph_mean(
    values: Tensor, graph_index: Tensor, mask: Tensor, graphs: int
) -> Tensor:
    if not values.numel() or not bool(mask.any()):
        return values.new_zeros(graphs)
    return _per_graph_mean(values[mask], graph_index[mask], graphs)


class MCVRBACLoss(nn.Module):
    """D1-B loss plus balanced BAC constraint objectives."""

    def __init__(
        self,
        base_weights: Mapping[str, float] | None = None,
        bac_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.base = MCVRLoss(base_weights)
        self.weights = {**DEFAULT_BAC_LOSS_WEIGHTS, **dict(bac_weights or {})}

    def forward(self, model: nn.Module, batch: Any) -> dict[str, Tensor]:
        base = self.base(model, batch)
        x_input = torch.as_tensor(field(batch, "x_input", field(batch, "x_init")))
        x_target = torch.as_tensor(
            field(batch, "x_target"), device=x_input.device, dtype=x_input.dtype
        )
        atom_batch = _atom_batch(batch, x_input)
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        t = x_input.new_zeros(graphs)
        output = model(batch, x_input, t)
        # New constraint objectives initially supervise only the new branches.
        # The unchanged D1-B loss remains responsible for the base Cartesian
        # and explicit-bond field, preventing a large new residual from
        # immediately overwriting the compatible initialization.
        proposal = x_input + output.get(
            "v_angle_fused", torch.zeros_like(x_input)
        ) + output.get("v_clash_fused", torch.zeros_like(x_input))
        active = torch.as_tensor(
            field(batch, "active_mode_mask"),
            device=x_input.device,
            dtype=x_input.dtype,
        ).reshape(graphs, 6)
        bonds = torch.as_tensor(
            field(batch, "active_bond_constraint_index"),
            device=x_input.device,
            dtype=torch.long,
        ).reshape(2, -1)
        angles = torch.as_tensor(
            field(batch, "active_angle_constraint_index"),
            device=x_input.device,
            dtype=torch.long,
        )
        if angles.ndim == 2 and angles.size(0) == 3:
            angles = angles.t()
        angles = angles.reshape(-1, 3)
        bond_ranges = torch.as_tensor(
            field(batch, "bond_allowed_range"),
            device=x_input.device,
            dtype=x_input.dtype,
        ).reshape(-1, 3)
        angle_ranges = torch.as_tensor(
            field(batch, "angle_allowed_range"),
            device=x_input.device,
            dtype=x_input.dtype,
        ).reshape(-1, 3)

        bond_graph = atom_batch[bonds[0]] if bonds.numel() else atom_batch.new_empty(0)
        angle_graph = (
            atom_batch[angles[:, 1]] if angles.numel() else atom_batch.new_empty(0)
        )
        input_bond_residual, input_bond_severity = standardized_interval_residual(
            bond_lengths(x_input, bonds), bond_ranges
        )
        proposal_bond_residual, proposal_bond_severity = (
            standardized_interval_residual(bond_lengths(proposal, bonds), bond_ranges)
        )
        target_bond_residual, _ = standardized_interval_residual(
            bond_lengths(x_target, bonds), bond_ranges
        )
        input_angle_residual, input_angle_severity = standardized_interval_residual(
            bond_angles(x_input, angles), angle_ranges
        )
        proposal_angle_residual, proposal_angle_severity = (
            standardized_interval_residual(bond_angles(proposal, angles), angle_ranges)
        )
        target_angle_residual, _ = standardized_interval_residual(
            bond_angles(x_target, angles), angle_ranges
        )
        bond_active = (input_bond_severity > 0) & (active[bond_graph, 0] > 0)
        angle_active = (input_angle_severity > 0) & (active[angle_graph, 1] > 0)
        bond_record = _masked_graph_mean(
            F.smooth_l1_loss(
                proposal_bond_residual,
                target_bond_residual,
                reduction="none",
            ),
            bond_graph,
            bond_active,
            graphs,
        )
        angle_record = _masked_graph_mean(
            F.smooth_l1_loss(
                proposal_angle_residual,
                target_angle_residual,
                reduction="none",
            ),
            angle_graph,
            angle_active,
            graphs,
        )

        clash = sparse_clash_edges(
            proposal,
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
        clash_record = _per_graph_mean(clash_values, clash["graph_index"], graphs)
        clash_record = clash_record * (active[:, 3] > 0).to(x_input.dtype)

        clean_graph = active[:, :4].sum(-1) <= 0
        noop = _per_graph_mean(
            output["v_final"].square().sum(-1), atom_batch, graphs
        )
        zero_error_noop = (
            noop[clean_graph].mean() if bool(clean_graph.any()) else noop.new_zeros(())
        )
        inactive_bond = ~bond_active
        inactive_angle = ~angle_active
        preservation = (
            _masked_graph_mean(
                (bond_lengths(proposal, bonds) - bond_lengths(x_input, bonds)).square(),
                bond_graph,
                inactive_bond,
                graphs,
            )
            + _masked_graph_mean(
                (bond_angles(proposal, angles) - bond_angles(x_input, angles)).square(),
                angle_graph,
                inactive_angle,
                graphs,
            )
        ).mean()
        new_bond = (proposal_bond_severity - input_bond_severity).clamp_min(0.0)
        new_angle = (proposal_angle_severity - input_angle_severity).clamp_min(0.0)
        no_new = (
            _per_graph_mean(new_bond, bond_graph, graphs)
            + _per_graph_mean(new_angle, angle_graph, graphs)
            + clash_record
        ).mean()
        confidence_terms = []
        gate_terms = []
        if output.get("angle_confidence", x_input.new_empty(0)).numel():
            confidence_terms.append(
                F.binary_cross_entropy(
                    output["angle_confidence"], angle_active.to(x_input.dtype)
                )
            )
            gate_terms.append(
                F.binary_cross_entropy(
                    output["angle_gate"], angle_active.to(x_input.dtype)
                )
            )
        if output.get("clash_confidence", x_input.new_empty(0)).numel():
            clash_active = output["active_mask"].to(x_input.dtype)
            confidence_terms.append(
                F.binary_cross_entropy(output["clash_confidence"], clash_active)
            )
            gate_terms.append(F.binary_cross_entropy(output["clash_gate"], clash_active))
        zero = x_input.new_zeros(())
        confidence = torch.stack(confidence_terms).mean() if confidence_terms else zero
        gate = torch.stack(gate_terms).mean() if gate_terms else zero
        coordinate_target = F.smooth_l1_loss(proposal, x_target)
        terms = {
            "bac_coordinate_target_loss": coordinate_target,
            "bac_bond_residual_loss": bond_record.mean(),
            "bac_angle_residual_loss": angle_record.mean(),
            "bac_clash_penetration_loss": clash_record.mean(),
            "bac_zero_error_noop_loss": zero_error_noop,
            "bac_preservation_loss": preservation,
            "bac_no_new_violation_loss": no_new,
            "bac_confidence_loss": confidence,
            "bac_gate_loss": gate,
        }
        key_map = {
            "coordinate_target": "bac_coordinate_target_loss",
            "bond_residual": "bac_bond_residual_loss",
            "angle_residual": "bac_angle_residual_loss",
            "clash_penetration": "bac_clash_penetration_loss",
            "zero_error_noop": "bac_zero_error_noop_loss",
            "preservation": "bac_preservation_loss",
            "no_new_violation": "bac_no_new_violation_loss",
            "confidence": "bac_confidence_loss",
            "gate": "bac_gate_loss",
        }
        weighted = {
            f"weighted_{name}": self.weights[name] * terms[key]
            for name, key in key_map.items()
        }
        total = base["loss"] + sum(weighted.values())
        return {
            **base,
            **terms,
            **weighted,
            "loss": total,
            "active_bond_constraints": bond_active.sum().to(x_input.dtype),
            "active_angle_constraints": angle_active.sum().to(x_input.dtype),
            "active_clash_constraints": clash["active_mask"].sum().to(x_input.dtype),
        }
