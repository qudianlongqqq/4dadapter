"""Loss decomposition for minimal-validity Cartesian flow training."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .geometry import (
    angle_triplets,
    internal_mode_velocities,
    torsion_quads,
    unique_bonds,
)
from .model import _atom_batch, _field


DEFAULT_LOSS_WEIGHTS = {
    "flow": 1.0,
    "validity": 0.25,
    "identity": 0.5,
    "anchor": 0.1,
    "sparse": 0.1,
    "torsion_anchor": 0.1,
    "error": 0.25,
    "uncertainty": 0.05,
    "trust": 0.1,
    "torsion_mode": 0.0,
    "torsion_gate_sparsity": 0.0,
    "high_flex_torsion_trust": 0.0,
}


def _masked_smooth_l1(predicted: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if predicted.numel() == 0 or not bool(mask.any()):
        return predicted.new_zeros(())
    return F.smooth_l1_loss(predicted[mask], target[mask])


class MCVRLoss(nn.Module):
    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        super().__init__()
        self.weights = {**DEFAULT_LOSS_WEIGHTS, **dict(weights or {})}

    def forward(self, model, batch: Any) -> dict[str, Tensor]:
        x_input = torch.as_tensor(_field(batch, "x_input", _field(batch, "x_init")))
        x_target = torch.as_tensor(_field(batch, "x_target"), device=x_input.device)
        atom_batch = _atom_batch(batch, x_input)
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        t = torch.rand(graphs, device=x_input.device, dtype=x_input.dtype)
        atom_t = t[atom_batch, None]
        x_t = (1.0 - atom_t) * x_input + atom_t * x_target
        target_velocity = x_target - x_input
        output = model(batch, x_t, t)
        predicted = output["v_final"]
        flow = F.smooth_l1_loss(predicted, target_velocity)

        active = torch.as_tensor(
            _field(batch, "active_mode_mask"), device=x_input.device, dtype=x_input.dtype
        ).reshape(graphs, 6)
        predicted_modes = internal_mode_velocities(x_t, predicted, batch)
        target_modes = internal_mode_velocities(x_t, target_velocity, batch)
        edge_index = torch.as_tensor(_field(batch, "edge_index"), device=x_input.device)
        rotatable = torch.as_tensor(
            _field(batch, "rotatable_bond_index", torch.empty(2, 0)), device=x_input.device
        )
        bonds = unique_bonds(edge_index)
        angles = angle_triplets(edge_index.cpu(), x_input.size(0)).to(x_input.device)
        torsions = torsion_quads(edge_index.cpu(), rotatable.cpu(), x_input.size(0)).to(x_input.device)
        mode_terms = [
            _masked_smooth_l1(
                predicted_modes["bond"], target_modes["bond"],
                active[atom_batch[bonds[0]], 0] > 0 if bonds.numel() else torch.zeros(0, dtype=torch.bool, device=x_input.device),
            ),
            _masked_smooth_l1(
                predicted_modes["angle"], target_modes["angle"],
                active[atom_batch[angles[:, 1]], 1] > 0 if angles.numel() else torch.zeros(0, dtype=torch.bool, device=x_input.device),
            ),
            _masked_smooth_l1(
                predicted_modes["torsion"], target_modes["torsion"],
                active[atom_batch[torsions[:, 1]], 4] > 0 if torsions.numel() else torch.zeros(0, dtype=torch.bool, device=x_input.device),
            ),
        ]
        local_active = (active[:, 2] > 0) | (active[:, 3] > 0)
        local_atoms = local_active[atom_batch]
        if bool(local_atoms.any()):
            mode_terms.append(F.smooth_l1_loss(predicted[local_atoms], target_velocity[local_atoms]))
        validity_mode = torch.stack(mode_terms).sum()

        clean = active[:, 5] > 0
        identity = predicted[clean[atom_batch]].square().mean() if bool(clean.any()) else flow.new_zeros(())
        anchor = predicted.square().sum(-1).mean()
        affected = torch.as_tensor(
            _field(batch, "affected_atom_mask", torch.ones(x_input.size(0))),
            device=x_input.device, dtype=x_input.dtype,
        ).reshape(-1)
        sparse_mask = affected <= 0
        sparse = predicted[sparse_mask].square().mean() if bool(sparse_mask.any()) else flow.new_zeros(())

        torsion_inactive = active[:, 4] <= 0
        if torsions.numel():
            torsion_graph = atom_batch[torsions[:, 1]]
            mask = torsion_inactive[torsion_graph]
            torsion_anchor = predicted_modes["torsion"][mask].square().mean() if bool(mask.any()) else flow.new_zeros(())
        else:
            torsion_anchor = flow.new_zeros(())
        torsion_contribution_modes = internal_mode_velocities(
            x_t, output["v_torsion_contribution"], batch
        )
        if torsions.numel():
            torsion_graph = atom_batch[torsions[:, 1]]
            torsion_active_mask = active[torsion_graph, 4] > 0
            torsion_mode = _masked_smooth_l1(
                torsion_contribution_modes["torsion"],
                target_modes["torsion"], torsion_active_mask,
            )
        else:
            torsion_mode = flow.new_zeros(())
        torsion_gate_sparsity = output["torsion_gate"].mean()
        high_flex = torch.as_tensor(
            _field(batch, "num_rotatable_bonds", torch.zeros(graphs)),
            device=x_input.device,
        ).reshape(graphs) >= 6
        high_flex_atoms = high_flex[atom_batch]
        high_flex_torsion_trust = (
            output["v_torsion_contribution"][high_flex_atoms].square().mean()
            if bool(high_flex_atoms.any()) else flow.new_zeros(())
        )
        error = F.binary_cross_entropy_with_logits(output["error_logits"], active)
        difficulty = torch.as_tensor(
            _field(batch, "difficulty_target", torch.zeros(graphs)),
            device=x_input.device, dtype=x_input.dtype,
        ).reshape(graphs, 1)
        uncertainty = F.smooth_l1_loss(output["uncertainty"], difficulty)
        atom_norm = torch.linalg.vector_norm(output["v_raw"], dim=-1)
        atom_excess = (atom_norm - model.max_velocity_atom_norm).clamp_min(0.0).square().mean()
        energy = predicted.new_zeros(graphs)
        energy.index_add_(0, atom_batch, output["v_raw"].square().sum(-1))
        counts = torch.bincount(atom_batch, minlength=graphs).clamp_min(1).to(predicted.dtype)
        graph_rms = torch.sqrt(energy / counts + 1.0e-12)
        trust = atom_excess + (graph_rms - model.max_velocity_graph_rms).clamp_min(0.0).square().mean()
        terms = {
            "flow_loss": flow,
            "validity_mode_loss": validity_mode,
            "identity_loss": identity,
            "anchor_loss": anchor,
            "sparse_loss": sparse,
            "torsion_anchor_loss": torsion_anchor,
            "torsion_mode_loss": torsion_mode,
            "torsion_gate_sparsity_loss": torsion_gate_sparsity,
            "high_flex_torsion_trust_loss": high_flex_torsion_trust,
            "error_loss": error,
            "uncertainty_loss": uncertainty,
            "trust_loss": trust,
        }
        weight_terms = {
            "flow": "flow_loss",
            "validity": "validity_mode_loss",
            "identity": "identity_loss",
            "anchor": "anchor_loss",
            "sparse": "sparse_loss",
            "torsion_anchor": "torsion_anchor_loss",
            "error": "error_loss",
            "uncertainty": "uncertainty_loss",
            "trust": "trust_loss",
            "torsion_mode": "torsion_mode_loss",
            "torsion_gate_sparsity": "torsion_gate_sparsity_loss",
            "high_flex_torsion_trust": "high_flex_torsion_trust_loss",
        }
        total = sum(self.weights[name] * terms[term] for name, term in weight_terms.items())
        return {"loss": total, **terms}
