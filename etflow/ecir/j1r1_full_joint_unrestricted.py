"""Independent unrestricted-movement variant of the frozen J1/R1 Full Joint model.

Only the magnitude parameterization and final Cartesian proposal differ from
``j1r1_full_joint``: tau is positive and unbounded via softplus, and the
proposal is exactly ``source + tau * graph_rms_normalized_direction``.  No
movement cap, clipping, rollback, or second-order Cartesian graph is used.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from .j1r1_full_joint import FullJointAction, FullJointModel, graph_mean_embeddings
from .learned_geometry import GraphGeometry, gaussian_nll, geometry_values, remove_rigid_component
from .lsgoba_v2_joint_magnitude import SOURCE_STATE_FEATURES, _family_features
from .musigma_reliability import (
    _angle_descent,
    _bond_descent,
    _collated_geometry_view,
    _offsets,
)


INITIAL_TAU_ANGSTROM = 0.003
INITIAL_RAW_PARAMETER = math.log(math.expm1(INITIAL_TAU_ANGSTROM))


class UnboundedSoftplusMagnitudeHead(nn.Module):
    """Architecture-matched magnitude head with no finite tau ceiling."""

    def __init__(self, graph_dim: int, state_dim: int, *, initial_tau: float = INITIAL_TAU_ANGSTROM) -> None:
        super().__init__()
        if not math.isfinite(initial_tau) or initial_tau <= 0:
            raise ValueError("initial_tau must be finite and positive")
        self.initial_tau = float(initial_tau)
        self.initial_raw = math.log(math.expm1(self.initial_tau))
        self.net = nn.Sequential(
            nn.Linear(int(graph_dim) + int(state_dim), 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, self.initial_raw)

    def forward(self, graph_embedding: Tensor, normalized_state: Tensor) -> Tensor:
        raw = self.net(torch.cat((graph_embedding, normalized_state), dim=-1)).squeeze(-1)
        return torch.nn.functional.softplus(raw)


class UnrestrictedFullJointModel(FullJointModel):
    """Full Joint model with an architecture-matched unbounded magnitude head."""

    def __init__(self, hidden_dim: int = 128, layers: int = 3) -> None:
        super().__init__(hidden_dim, layers)
        bounded = self.magnitude
        unbounded = UnboundedSoftplusMagnitudeHead(
            hidden_dim, len(SOURCE_STATE_FEATURES), initial_tau=INITIAL_TAU_ANGSTROM
        )
        # Preserve the exact standard random initialization of the two hidden
        # magnitude layers. Only the scientific output parameterization differs.
        with torch.no_grad():
            for index in (0, 2):
                unbounded.net[index].weight.copy_(bounded.net[index].weight)
                unbounded.net[index].bias.copy_(bounded.net[index].bias)
        self.magnitude = unbounded


def unrestricted_full_joint_action(
    model: UnrestrictedFullJointModel,
    source: Tensor,
    graphs: Sequence[GraphGeometry],
    prediction: Mapping[str, Tensor],
) -> FullJointAction:
    """Exact first-order Full Joint action without movement clipping."""
    atom_offsets, bond_offsets, angle_offsets = _offsets(graphs)
    view = _collated_geometry_view(graphs, source.device)
    bond_q, angle_q = geometry_values(source, view)
    bond_r, angle_r = model.reliability(
        prediction["bond_features"], prediction["angle_features"],
        bond_q.detach().to(prediction["bond_mu"]),
        angle_q.detach().to(prediction["angle_mu"]),
        prediction["bond_mu"].detach(), prediction["angle_mu"].detach(),
        prediction["bond_sigma"].detach(), prediction["angle_sigma"].detach(),
    )
    graph_embedding = graph_mean_embeddings(prediction["node_embedding"], graphs)
    family_weights = model.adaptive_ba(graph_embedding)

    bond_mu = prediction["bond_mu"].detach().to(bond_q)
    angle_mu = prediction["angle_mu"].detach().to(angle_q)
    bond_sigma = prediction["bond_sigma"].detach().to(bond_q)
    angle_sigma = prediction["angle_sigma"].detach().to(angle_q)
    bond_z = (bond_q.detach() - bond_mu) / bond_sigma
    angle_z = (angle_q.detach() - angle_mu) / angle_sigma
    bond_nll = gaussian_nll(bond_q.detach(), bond_mu, bond_sigma)
    angle_nll = gaussian_nll(angle_q.detach(), angle_mu, angle_sigma)

    directions, states = [], []
    for index, graph_cpu in enumerate(graphs):
        graph = graph_cpu.to(source.device)
        atoms = slice(atom_offsets[index], atom_offsets[index + 1])
        bonds = slice(bond_offsets[index], bond_offsets[index + 1])
        angles = slice(angle_offsets[index], angle_offsets[index + 1])
        coordinates = source[atoms]
        nb = max(1, bond_offsets[index + 1] - bond_offsets[index])
        na = max(1, angle_offsets[index + 1] - angle_offsets[index])
        w_b, w_a = family_weights[index, 0], family_weights[index, 1]
        bond_coefficient = (
            (w_b / nb) * bond_r[bonds].to(coordinates)
            * (bond_q[bonds].detach() - bond_mu[bonds]) / bond_sigma[bonds].square()
        )
        angle_coefficient = (
            (w_a / na) * angle_r[angles].to(coordinates)
            * (angle_q[angles].detach() - angle_mu[angles]) / angle_sigma[angles].square()
        )
        raw = _bond_descent(coordinates, graph, bond_coefficient)
        raw = raw + _angle_descent(coordinates, graph, angle_coefficient)
        projected = remove_rigid_component(raw, coordinates.detach())
        atom_norm = torch.linalg.vector_norm(projected, dim=-1)
        gradient_rms = projected.square().sum(-1).mean().sqrt()
        direction = projected / gradient_rms.clamp_min(1.0e-14)
        direction = torch.where(
            torch.isfinite(projected).all() & (gradient_rms > 1.0e-14),
            direction, torch.zeros_like(direction),
        )
        directions.append(direction)

        eb = bond_nll[bonds].mean() if bond_nll[bonds].numel() else coordinates.new_zeros(())
        ea = angle_nll[angles].mean() if angle_nll[angles].numel() else coordinates.new_zeros(())
        state = torch.cat((
            _family_features(bond_z[bonds] / (2.0 * nb) ** 0.5),
            _family_features(angle_z[angles] / (2.0 * na) ** 0.5),
            torch.stack((eb, ea, 0.5 * (eb + ea))),
            torch.stack((gradient_rms.detach(), atom_norm.mean().detach(), atom_norm.max().detach())),
            coordinates.new_tensor((coordinates.size(0), graph.bonds.size(1), graph.angles.size(0))),
        ))
        states.append(state.to(graph_embedding))

    direction = torch.cat(directions)
    source_state = torch.stack(states).detach()
    detached_embedding = graph_embedding.detach()
    tau = model.magnitude(detached_embedding, model.normalized_state(source_state))
    atom_counts = torch.tensor(
        [int(graph.atom_categorical.size(0)) for graph in graphs], device=tau.device
    )
    delta = torch.repeat_interleave(tau, atom_counts)[:, None] * direction
    proposal = source + delta
    graph_rms, offset = [], 0
    for count in atom_counts.tolist():
        local = delta[offset : offset + count]
        graph_rms.append(local.square().sum(-1).mean().sqrt())
        offset += count
    return FullJointAction(
        direction=direction,
        proposal=proposal,
        uncapped_proposal=proposal,
        cap_active=torch.zeros_like(tau),
        graph_rms=torch.stack(graph_rms),
        bond_reliability=bond_r,
        angle_reliability=angle_r,
        family_weights=family_weights,
        tau=tau,
        source_state=source_state,
        graph_embedding=detached_embedding,
    )
