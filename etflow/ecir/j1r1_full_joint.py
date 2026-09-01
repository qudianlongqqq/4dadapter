"""Full-joint J1/R1 + Adaptive-BA + adaptive-magnitude components.

The module preserves the validated first-order R1 action graph. Cartesian
primitive derivatives, source defects, mu, sigma, magnitude inputs, and the
Adaptive-BA graph input follow their existing detach boundaries. No Hessian or
second-order Cartesian graph is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from .learned_geometry import GraphGeometry, gaussian_nll, geometry_values, remove_rigid_component
from .lsgoba_v2_joint_magnitude import AdaptiveTrustMagnitudeHead, SOURCE_STATE_FEATURES, _family_features, scaled_proposal
from .musigma_reliability import (
    DirectMuSigmaModel,
    PrimitiveReliabilityHead,
    _angle_descent,
    _bond_descent,
    _collated_geometry_view,
    _offsets,
)


class AdaptiveBAHead(nn.Module):
    """The audited Adaptive-BA-v2 128 -> 64 -> 2 source-free head."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden = nn.Linear(hidden_dim, 64)
        self.activation = nn.SiLU()
        self.output = nn.Linear(64, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, graph_embedding: Tensor) -> Tensor:
        return torch.softmax(self.output(self.activation(self.hidden(graph_embedding))), dim=-1)


class FullJointModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, layers: int = 3) -> None:
        super().__init__()
        self.belief = DirectMuSigmaModel(hidden_dim, layers)
        self.reliability = PrimitiveReliabilityHead(hidden_dim)
        self.adaptive_ba = AdaptiveBAHead(hidden_dim)
        self.magnitude = AdaptiveTrustMagnitudeHead(
            hidden_dim, len(SOURCE_STATE_FEATURES), initial_tau=0.003, tau_max=0.010
        )
        self.register_buffer("state_mean", torch.zeros(len(SOURCE_STATE_FEATURES)))
        self.register_buffer("state_std", torch.ones(len(SOURCE_STATE_FEATURES)))

    def set_state_normalization(self, mean: Tensor, std: Tensor) -> None:
        self.state_mean.copy_(torch.as_tensor(mean, dtype=self.state_mean.dtype).reshape_as(self.state_mean))
        self.state_std.copy_(torch.as_tensor(std, dtype=self.state_std.dtype).reshape_as(self.state_std).clamp_min(1.0e-6))

    def normalized_state(self, state: Tensor) -> Tensor:
        return (state - self.state_mean.to(state)) / self.state_std.to(state)

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "backbone": self.belief.backbone_parameters(),
            "mu": self.belief.mu_head_parameters(),
            "j1_sigma": self.belief.sigma_head_parameters(),
            "reliability": list(self.reliability.parameters()),
            "adaptive_ba": list(self.adaptive_ba.parameters()),
            "magnitude": list(self.magnitude.parameters()),
        }


@dataclass
class FullJointAction:
    direction: Tensor
    proposal: Tensor
    uncapped_proposal: Tensor
    cap_active: Tensor
    graph_rms: Tensor
    bond_reliability: Tensor
    angle_reliability: Tensor
    family_weights: Tensor
    tau: Tensor
    source_state: Tensor
    graph_embedding: Tensor


def graph_mean_embeddings(node_embedding: Tensor, graphs: Sequence[GraphGeometry]) -> Tensor:
    result, offset = [], 0
    for graph in graphs:
        count = int(graph.atom_categorical.size(0))
        result.append(node_embedding[offset : offset + count].detach().mean(0))
        offset += count
    return torch.stack(result)


def full_joint_action(
    model: FullJointModel,
    source: Tensor,
    graphs: Sequence[GraphGeometry],
    prediction: Mapping[str, Tensor],
    *,
    atom_cap: float = 0.03,
) -> FullJointAction:
    """Construct the differentiable first-order action with frozen math boundaries."""
    atom_offsets, bond_offsets, angle_offsets = _offsets(graphs)
    view = _collated_geometry_view(graphs, source.device)
    bond_q, angle_q = geometry_values(source, view)
    bond_r, angle_r = model.reliability(
        prediction["bond_features"],
        prediction["angle_features"],
        bond_q.detach().to(prediction["bond_mu"]),
        angle_q.detach().to(prediction["angle_mu"]),
        prediction["bond_mu"].detach(),
        prediction["angle_mu"].detach(),
        prediction["bond_sigma"].detach(),
        prediction["angle_sigma"].detach(),
    )
    graph_embedding = graph_mean_embeddings(prediction["node_embedding"], graphs)
    family_weights = model.adaptive_ba(graph_embedding)

    detached_bond_mu = prediction["bond_mu"].detach().to(bond_q)
    detached_angle_mu = prediction["angle_mu"].detach().to(angle_q)
    detached_bond_sigma = prediction["bond_sigma"].detach().to(bond_q)
    detached_angle_sigma = prediction["angle_sigma"].detach().to(angle_q)
    bond_z = (bond_q.detach() - detached_bond_mu) / detached_bond_sigma
    angle_z = (angle_q.detach() - detached_angle_mu) / detached_angle_sigma
    bond_nll = gaussian_nll(bond_q.detach(), detached_bond_mu, detached_bond_sigma)
    angle_nll = gaussian_nll(angle_q.detach(), detached_angle_mu, detached_angle_sigma)

    directions, states = [], []
    for i, graph_cpu in enumerate(graphs):
        graph = graph_cpu.to(source.device)
        aa = slice(atom_offsets[i], atom_offsets[i + 1])
        bb = slice(bond_offsets[i], bond_offsets[i + 1])
        cc = slice(angle_offsets[i], angle_offsets[i + 1])
        coordinates = source[aa]
        nb = max(1, bond_offsets[i + 1] - bond_offsets[i])
        na = max(1, angle_offsets[i + 1] - angle_offsets[i])
        w_b, w_a = family_weights[i, 0], family_weights[i, 1]
        bond_coefficient = (
            (w_b / nb)
            * bond_r[bb].to(coordinates)
            * (bond_q[bb].detach() - detached_bond_mu[bb])
            / detached_bond_sigma[bb].square()
        )
        angle_coefficient = (
            (w_a / na)
            * angle_r[cc].to(coordinates)
            * (angle_q[cc].detach() - detached_angle_mu[cc])
            / detached_angle_sigma[cc].square()
        )
        raw = _bond_descent(coordinates, graph, bond_coefficient)
        raw = raw + _angle_descent(coordinates, graph, angle_coefficient)
        projected = remove_rigid_component(raw, coordinates.detach())
        gradient_norm = torch.linalg.vector_norm(projected, dim=-1)
        gradient_rms = projected.square().sum(-1).mean().sqrt()
        direction = projected / gradient_rms.clamp_min(1.0e-14)
        direction = torch.where(
            torch.isfinite(projected).all() & (gradient_rms > 1.0e-14),
            direction,
            torch.zeros_like(direction),
        )
        directions.append(direction)

        eb = bond_nll[bb].mean() if bond_nll[bb].numel() else coordinates.new_zeros(())
        ea = angle_nll[cc].mean() if angle_nll[cc].numel() else coordinates.new_zeros(())
        local_bond_z = bond_z[bb] / (2.0 * nb) ** 0.5
        local_angle_z = angle_z[cc] / (2.0 * na) ** 0.5
        gradient_features = torch.stack(
            (gradient_rms.detach(), gradient_norm.mean().detach(), gradient_norm.max().detach())
        )
        size = coordinates.new_tensor((coordinates.size(0), graph.bonds.size(1), graph.angles.size(0)))
        state = torch.cat(
            (
                _family_features(local_bond_z),
                _family_features(local_angle_z),
                torch.stack((eb, ea, 0.5 * (eb + ea))),
                gradient_features,
                size,
            )
        )
        states.append(state.to(graph_embedding))

    direction = torch.cat(directions)
    source_state = torch.stack(states).detach()
    detached_embedding = graph_embedding.detach()
    tau = model.magnitude(detached_embedding, model.normalized_state(source_state))
    proposal, cap_active, graph_rms = scaled_proposal(
        source, direction, tau, graphs, atom_cap=float(atom_cap)
    )
    atom_counts = torch.tensor(
        [int(graph.atom_categorical.size(0)) for graph in graphs], device=tau.device
    )
    uncapped = source + torch.repeat_interleave(tau, atom_counts)[:, None] * direction
    return FullJointAction(
        direction=direction,
        proposal=proposal,
        uncapped_proposal=uncapped,
        cap_active=cap_active,
        graph_rms=graph_rms,
        bond_reliability=bond_r,
        angle_reliability=angle_r,
        family_weights=family_weights,
        tau=tau,
        source_state=source_state,
        graph_embedding=detached_embedding,
    )
