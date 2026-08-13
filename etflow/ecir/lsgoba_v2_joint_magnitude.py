"""Development-only LSGO-BA v2 adaptive trust-magnitude components.

The module wraps the frozen v1 Bond/Angle predictor. It changes neither the
structured objective nor the Cartesian direction and deliberately detaches the
direction before the post-refinement loss.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from .learned_geometry import (
    GraphGeometry,
    LearnedGeometryObjective,
    gaussian_nll,
    geometry_values,
    remove_rigid_component,
)


SOURCE_STATE_FEATURES = (
    "mean_abs_rho_B", "rms_rho_B", "max_abs_rho_B", "p90_abs_rho_B",
    "mean_abs_rho_A", "rms_rho_A", "max_abs_rho_A", "p90_abs_rho_A",
    "E_B", "E_A", "E_BA",
    "graph_RMS_gradient", "mean_atom_gradient_norm", "max_atom_gradient_norm",
    "n_atoms", "n_bonds", "n_angles",
)


class AdaptiveTrustMagnitudeHead(nn.Module):
    def __init__(self, graph_dim: int, state_dim: int, *, initial_tau: float = 0.003, tau_max: float = 0.010) -> None:
        super().__init__()
        self.tau_max = float(tau_max)
        self.net = nn.Sequential(
            nn.Linear(int(graph_dim) + int(state_dim), 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, math.log(float(initial_tau) / (self.tau_max - float(initial_tau))))

    def forward(self, graph_embedding: Tensor, normalized_state: Tensor) -> Tensor:
        return self.tau_max * torch.sigmoid(self.net(torch.cat((graph_embedding, normalized_state), dim=-1)).squeeze(-1))


class JointMagnitudeLSGO(nn.Module):
    def __init__(self, *, hidden_dim: int = 128, layers: int = 3, initial_tau: float = 0.003, tau_max: float = 0.010) -> None:
        super().__init__()
        self.geometry = LearnedGeometryObjective(hidden_dim=hidden_dim, layers=layers, learned_sigma=False)
        self.magnitude = AdaptiveTrustMagnitudeHead(hidden_dim, len(SOURCE_STATE_FEATURES), initial_tau=initial_tau, tau_max=tau_max)
        self.register_buffer("state_mean", torch.zeros(len(SOURCE_STATE_FEATURES)))
        self.register_buffer("state_std", torch.ones(len(SOURCE_STATE_FEATURES)))

    def set_state_normalization(self, mean: Tensor, std: Tensor, floor: float = 1e-6) -> None:
        mean = torch.as_tensor(mean, dtype=self.state_mean.dtype, device=self.state_mean.device)
        std = torch.as_tensor(std, dtype=self.state_std.dtype, device=self.state_std.device).clamp_min(float(floor))
        if mean.shape != self.state_mean.shape or std.shape != self.state_std.shape:
            raise ValueError("source-state normalization shape changed")
        self.state_mean.copy_(mean); self.state_std.copy_(std)

    def normalized_state(self, state: Tensor) -> Tensor:
        return (state - self.state_mean.to(state)) / self.state_std.to(state)


def _family_features(z: Tensor) -> Tensor:
    if not z.numel():
        return z.new_zeros(4)
    absolute = z.abs()
    return torch.stack((absolute.mean(), z.square().mean().sqrt(), absolute.max(), torch.quantile(absolute, 0.90)))


def batch_source_directions_and_state(
    source: Tensor,
    graphs: Sequence[GraphGeometry],
    parameters: Mapping[str, Tensor],
    node_embedding: Tensor,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    """Compute current v1 directions and differentiable source-state features.

    Direction is computed from the exact per-molecule equal-family v1
    structured objective. The returned normalized directions are detached,
    preventing mixed second-order direction backpropagation.
    """
    source = source.detach().clone().requires_grad_(True)
    atom_offsets = [0]; bond_offsets = [0]; angle_offsets = [0]
    for graph in graphs:
        atom_offsets.append(atom_offsets[-1] + int(graph.atom_categorical.size(0)))
        bond_offsets.append(bond_offsets[-1] + int(graph.bonds.size(1)))
        angle_offsets.append(angle_offsets[-1] + int(graph.angles.size(0)))
    bonds, angles = geometry_values(source, _collated_geometry_view(graphs, source.device))
    bond_z = (bonds - parameters["bond_mu"]) / parameters["bond_sigma"]
    angle_z = (angles - parameters["angle_mu"]) / parameters["angle_sigma"]
    bond_nll = gaussian_nll(bonds, parameters["bond_mu"], parameters["bond_sigma"])
    angle_nll = gaussian_nll(angles, parameters["angle_mu"], parameters["angle_sigma"])
    objectives, family_groups = [], []
    for i in range(len(graphs)):
        bn = bond_nll[bond_offsets[i]:bond_offsets[i + 1]]
        an = angle_nll[angle_offsets[i]:angle_offsets[i + 1]]
        eb = bn.mean() if bn.numel() else source.new_zeros(())
        ea = an.mean() if an.numel() else source.new_zeros(())
        objectives.append(0.5 * (eb + ea)); family_groups.append((eb, ea))
    objective_vector = torch.stack(objectives)
    gradient, = torch.autograd.grad(objective_vector.sum(), source, retain_graph=True, create_graph=False)
    directions, states, graph_embeddings = [], [], []
    diagnostics = {"gradient_rms": [], "atom_gradient_mean": [], "atom_gradient_max": []}
    for i, graph in enumerate(graphs):
        aa = slice(atom_offsets[i], atom_offsets[i + 1]); bb = slice(bond_offsets[i], bond_offsets[i + 1]); cc = slice(angle_offsets[i], angle_offsets[i + 1])
        local_source = source[aa]; local_gradient = remove_rigid_component(gradient[aa], local_source)
        grad_norm = torch.linalg.vector_norm(local_gradient, dim=-1)
        grad_rms = local_gradient.square().sum(-1).mean().sqrt()
        if not bool(torch.isfinite(local_gradient).all()) or float(grad_rms.detach()) <= 1e-14:
            direction = torch.zeros_like(local_gradient)
        else:
            direction = -local_gradient / grad_rms
        directions.append(direction.detach())
        eb, ea = family_groups[i]
        gradient_features = torch.stack((grad_rms.detach(), grad_norm.mean().detach(), grad_norm.max().detach()))
        size_features = source.new_tensor((local_source.size(0), graph.bonds.size(1), graph.angles.size(0)))
        local_bond_z = bond_z[bb] / math.sqrt(2.0 * max(1, bond_offsets[i + 1] - bond_offsets[i]))
        local_angle_z = angle_z[cc] / math.sqrt(2.0 * max(1, angle_offsets[i + 1] - angle_offsets[i]))
        state = torch.cat((_family_features(local_bond_z), _family_features(local_angle_z), torch.stack((eb, ea, objectives[i])), gradient_features, size_features))
        states.append(state.to(node_embedding))
        graph_embeddings.append(node_embedding[aa].mean(0))
        diagnostics["gradient_rms"].append(grad_rms.detach()); diagnostics["atom_gradient_mean"].append(grad_norm.mean().detach()); diagnostics["atom_gradient_max"].append(grad_norm.max().detach())
    return torch.cat(directions), torch.stack(states), torch.stack(graph_embeddings), {key: torch.stack(value) for key, value in diagnostics.items()}


def _collated_geometry_view(graphs: Sequence[GraphGeometry], device: torch.device) -> GraphGeometry:
    """Minimal graph collation used only for coordinate primitive indexing."""
    atoms=[]; edges=[]; edge_codes=[]; bonds=[]; bond_codes=[]; angles=[]; angle_codes=[]; fixed_b=[]; fixed_a=[]; offset=0
    for graph in graphs:
        graph=graph.to(device); atoms.append(graph.atom_categorical); edges.append(graph.edge_index+offset); edge_codes.append(graph.edge_categorical)
        bonds.append(graph.bonds+offset); bond_codes.append(graph.bond_categorical); angles.append(graph.angles+offset); angle_codes.append(graph.angle_edge_categorical)
        fixed_b.append(graph.bond_fixed); fixed_a.append(graph.angle_fixed); offset += graph.atom_categorical.size(0)
    return GraphGeometry(atom_categorical=torch.cat(atoms), edge_index=torch.cat(edges,1), edge_categorical=torch.cat(edge_codes), bonds=torch.cat(bonds,1), bond_categorical=torch.cat(bond_codes), angles=torch.cat(angles), angle_edge_categorical=torch.cat(angle_codes), rings=(), chirality=(), bond_fixed=torch.cat(fixed_b), angle_fixed=torch.cat(fixed_a), ring_fixed=None, backoff=None)


def scaled_proposal(source: Tensor, directions: Tensor, tau: Tensor, graphs: Sequence[GraphGeometry], atom_cap: float = 0.03) -> tuple[Tensor, Tensor]:
    pieces=[]; cap_active=[]; offset=0
    for i, graph in enumerate(graphs):
        count=int(graph.atom_categorical.size(0)); delta=directions[offset:offset+count] * tau[i]
        norms=torch.linalg.vector_norm(delta,dim=-1); scales=torch.clamp(delta.new_tensor(float(atom_cap))/norms.clamp_min(1e-15),max=1.0)
        delta=delta*scales[:,None]; pieces.append(source[offset:offset+count]+delta); cap_active.append((scales<1).any()); offset+=count
    return torch.cat(pieces), torch.stack(cap_active).to(dtype=source.dtype)
