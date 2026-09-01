"""Direct mu/sigma and source-conditioned action-reliability components.

This module is deliberately independent of the Sigma-v2 teacher/student code.
It implements the frozen six-arm development factorial only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from .learned_geometry import (
    GraphGeometry,
    LearnedGeometryObjective,
    geometry_values,
    remove_rigid_component,
)


LOG_SIGMA_RATIO_LIMIT = 6.0
INITIAL_RELIABILITY = 0.999
RELIABILITY_INITIAL_LOGIT = math.log(INITIAL_RELIABILITY / (1.0 - INITIAL_RELIABILITY))


def primitive_features(
    geometry: LearnedGeometryObjective,
    graph: GraphGeometry,
    node_embedding: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the same source-free local representations used by the mu heads."""
    left, right = graph.bonds
    bond_edge = geometry._edge(graph.bond_categorical)
    bond = torch.cat(
        (
            node_embedding[left] + node_embedding[right],
            (node_embedding[left] - node_embedding[right]).abs(),
            node_embedding[left] * node_embedding[right],
            bond_edge,
        ),
        dim=-1,
    )
    if graph.angles.numel():
        aleft, center, aright = graph.angles.t()
        first = geometry._edge(graph.angle_edge_categorical[:, :3])
        second = geometry._edge(graph.angle_edge_categorical[:, 3:])
        angle = torch.cat(
            (
                node_embedding[center],
                node_embedding[aleft] + node_embedding[aright],
                (node_embedding[aleft] - node_embedding[aright]).abs(),
                node_embedding[aleft] * node_embedding[aright],
                first + second,
                (first - second).abs(),
            ),
            dim=-1,
        )
    else:
        angle = node_embedding.new_empty((0, 4 * node_embedding.size(-1) + 40))
    return bond, angle


class DirectMuSigmaModel(nn.Module):
    """Common large GNN, mu heads, and direct learned-sigma heads."""

    def __init__(self, hidden_dim: int = 128, layers: int = 3) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.geometry = LearnedGeometryObjective(
            hidden_dim=hidden_dim, layers=layers, learned_sigma=False
        )
        edge_dim = 20
        self.bond_sigma_head = nn.Sequential(
            nn.Linear(3 * hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.angle_sigma_head = nn.Sequential(
            nn.Linear(4 * hidden_dim + 2 * edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.bond_sigma_head[-1].weight)
        nn.init.zeros_(self.bond_sigma_head[-1].bias)
        nn.init.zeros_(self.angle_sigma_head[-1].weight)
        nn.init.zeros_(self.angle_sigma_head[-1].bias)

    def backbone_parameters(self) -> list[nn.Parameter]:
        excluded = {id(p) for p in self.mu_head_parameters()}
        return [p for p in self.geometry.parameters() if id(p) not in excluded]

    def mu_head_parameters(self) -> list[nn.Parameter]:
        return list(self.geometry.bond_head.parameters()) + list(
            self.geometry.angle_head.parameters()
        )

    def sigma_head_parameters(self) -> list[nn.Parameter]:
        return list(self.bond_sigma_head.parameters()) + list(
            self.angle_sigma_head.parameters()
        )

    def forward(
        self,
        graph: GraphGeometry,
        *,
        detach_sigma_features: bool = False,
    ) -> dict[str, Tensor]:
        prediction = self.geometry(graph)
        bond_features, angle_features = primitive_features(
            self.geometry, graph, prediction["node_embedding"]
        )
        sigma_bond_features = bond_features.detach() if detach_sigma_features else bond_features
        sigma_angle_features = angle_features.detach() if detach_sigma_features else angle_features
        bond_raw = self.bond_sigma_head(sigma_bond_features).squeeze(-1)
        angle_raw = (
            self.angle_sigma_head(sigma_angle_features).squeeze(-1)
            if sigma_angle_features.numel()
            else bond_raw.new_empty(0)
        )
        bond_delta = LOG_SIGMA_RATIO_LIMIT * torch.tanh(
            bond_raw / LOG_SIGMA_RATIO_LIMIT
        )
        angle_delta = LOG_SIGMA_RATIO_LIMIT * torch.tanh(
            angle_raw / LOG_SIGMA_RATIO_LIMIT
        )
        bond_stat = graph.bond_fixed[:, 1].to(bond_delta)
        angle_stat = graph.angle_fixed[:, 1].to(angle_delta)
        bond_sigma = bond_stat * torch.exp(bond_delta)
        angle_sigma = angle_stat * torch.exp(angle_delta)
        return {
            **prediction,
            "bond_features": bond_features,
            "angle_features": angle_features,
            "bond_log_sigma_ratio": bond_delta,
            "angle_log_sigma_ratio": angle_delta,
            "bond_sigma": bond_sigma,
            "angle_sigma": angle_sigma,
        }


class PrimitiveReliabilityHead(nn.Module):
    """One shared bounded gate head after family-specific input projection."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.bond_projection = nn.Sequential(
            nn.Linear(3 * hidden_dim + 20, hidden_dim), nn.SiLU()
        )
        self.angle_projection = nn.Sequential(
            nn.Linear(4 * hidden_dim + 40, hidden_dim), nn.SiLU()
        )
        # Scalars: q, mu, log(sigma), signed defect, abs defect,
        # standardized defect; followed by a two-value family indicator.
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + 8, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        # A strictly zero final weight would make the initial gate constant but
        # would also block the required action -> shared-backbone gradient on
        # the first update.  This deterministic microscopic initialization
        # keeps normalized R0/R1 actions equivalent at the frozen tolerance
        # while preserving a nonzero first-step route.
        nn.init.normal_(self.shared[-1].weight, mean=0.0, std=1.0e-10)
        nn.init.constant_(self.shared[-1].bias, RELIABILITY_INITIAL_LOGIT)

    @staticmethod
    def _scalars(q: Tensor, mu: Tensor, sigma: Tensor, family: int) -> Tensor:
        # mu/sigma are detached at the action-update callsite.  Values are
        # represented in native family units and by their standardized defect.
        defect = q - mu
        standardized = defect / sigma.clamp_min(1.0e-12)
        family_values = q.new_zeros((q.numel(), 2))
        family_values[:, int(family)] = 1.0
        return torch.cat(
            (
                q[:, None],
                mu[:, None],
                sigma.log()[:, None],
                defect[:, None],
                defect.abs()[:, None],
                standardized[:, None],
                family_values,
            ),
            dim=-1,
        )

    def forward(
        self,
        bond_features: Tensor,
        angle_features: Tensor,
        bond_q: Tensor,
        angle_q: Tensor,
        bond_mu: Tensor,
        angle_mu: Tensor,
        bond_sigma: Tensor,
        angle_sigma: Tensor,
    ) -> tuple[Tensor, Tensor]:
        bond_input = torch.cat(
            (
                self.bond_projection(bond_features),
                self._scalars(bond_q, bond_mu, bond_sigma, 0),
            ),
            dim=-1,
        )
        bond = torch.sigmoid(self.shared(bond_input).squeeze(-1))
        if angle_features.numel():
            angle_input = torch.cat(
                (
                    self.angle_projection(angle_features),
                    self._scalars(angle_q, angle_mu, angle_sigma, 1),
                ),
                dim=-1,
            )
            angle = torch.sigmoid(self.shared(angle_input).squeeze(-1))
        else:
            angle = bond.new_empty(0)
        return bond, angle


def _offsets(graphs: Sequence[GraphGeometry]) -> tuple[list[int], list[int], list[int]]:
    atoms, bonds, angles = [0], [0], [0]
    for graph in graphs:
        atoms.append(atoms[-1] + int(graph.atom_categorical.size(0)))
        bonds.append(bonds[-1] + int(graph.bonds.size(1)))
        angles.append(angles[-1] + int(graph.angles.size(0)))
    return atoms, bonds, angles


def molecule_equal_family_mean(
    bond_values: Tensor,
    angle_values: Tensor,
    graphs: Sequence[GraphGeometry],
) -> Tensor:
    """Molecule-balanced 0.5/0.5 Bond/Angle aggregation."""
    _, bo, ao = _offsets(graphs)
    values = []
    for i in range(len(graphs)):
        bond = bond_values[bo[i] : bo[i + 1]]
        angle = angle_values[ao[i] : ao[i + 1]]
        eb = bond.mean() if bond.numel() else bond_values.new_zeros(())
        ea = angle.mean() if angle.numel() else angle_values.new_zeros(())
        values.append(0.5 * (eb + ea))
    return torch.stack(values).mean()


def belief_loss(
    method: str,
    prediction: Mapping[str, Tensor],
    reference_bond: Tensor,
    reference_angle: Tensor,
    graphs: Sequence[GraphGeometry],
    beta: float = 0.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """J0, beta-NLL J1, or exact faithful-normal J2 objective."""
    method = method.upper()
    bond_mu, angle_mu = prediction["bond_mu"], prediction["angle_mu"]
    bond_sigma, angle_sigma = prediction["bond_sigma"], prediction["angle_sigma"]

    def nll(y: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
        return 0.5 * ((y - mu) / sigma).square() + sigma.log()

    if method == "J0":
        bond = nll(reference_bond, bond_mu, bond_sigma)
        angle = nll(reference_angle, angle_mu, angle_sigma)
        total = molecule_equal_family_mean(bond, angle, graphs)
        return total, {"mean": total, "variance": total}
    if method == "J1":
        bond = nll(reference_bond, bond_mu, bond_sigma) * (
            bond_sigma.square().detach().pow(float(beta))
        )
        angle = nll(reference_angle, angle_mu, angle_sigma) * (
            angle_sigma.square().detach().pow(float(beta))
        )
        total = molecule_equal_family_mean(bond, angle, graphs)
        return total, {"mean": total, "variance": total}
    if method != "J2":
        raise ValueError(f"unknown belief method: {method}")

    # Faithful Heteroscedastic Normal: unit-variance Gaussian mean loss;
    # variance NLL sees stop-gradient mean and, via forward(...,
    # detach_sigma_features=True), cannot update the shared representation.
    bond_mean = 0.5 * (reference_bond - bond_mu).square()
    angle_mean = 0.5 * (reference_angle - angle_mu).square()
    mean_loss = molecule_equal_family_mean(bond_mean, angle_mean, graphs)
    bond_variance = nll(reference_bond, bond_mu.detach(), bond_sigma)
    angle_variance = nll(reference_angle, angle_mu.detach(), angle_sigma)
    variance_loss = molecule_equal_family_mean(bond_variance, angle_variance, graphs)
    return mean_loss + variance_loss, {"mean": mean_loss, "variance": variance_loss}


def _bond_descent(
    coordinates: Tensor,
    graph: GraphGeometry,
    coefficient: Tensor,
) -> Tensor:
    left, right = graph.bonds
    delta = coordinates[left] - coordinates[right]
    derivative = delta / torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-12)[:, None]
    result = coordinates.new_zeros(coordinates.shape)
    contribution = -coefficient[:, None] * derivative.detach()
    result.index_add_(0, left, contribution)
    result.index_add_(0, right, -contribution)
    return result


def _angle_descent(
    coordinates: Tensor,
    graph: GraphGeometry,
    coefficient: Tensor,
) -> Tensor:
    if not graph.angles.numel():
        return coordinates.new_zeros(coordinates.shape)
    left, center, right = graph.angles.t()
    u = coordinates[left] - coordinates[center]
    v = coordinates[right] - coordinates[center]
    nu = torch.linalg.vector_norm(u, dim=-1).clamp_min(1.0e-12)
    nv = torch.linalg.vector_norm(v, dim=-1).clamp_min(1.0e-12)
    raw = (u * v).sum(-1) / (nu * nv)
    active = ((raw > -1.0 + 1.0e-7) & (raw < 1.0 - 1.0e-7)).to(raw)
    cosine = raw.clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
    du = v / (nu * nv)[:, None] - cosine[:, None] * u / nu.square()[:, None]
    dv = u / (nu * nv)[:, None] - cosine[:, None] * v / nv.square()[:, None]
    scale = (-coefficient * active)[:, None]
    gl = scale * du.detach()
    gr = scale * dv.detach()
    result = coordinates.new_zeros(coordinates.shape)
    result.index_add_(0, left, gl)
    result.index_add_(0, right, gr)
    result.index_add_(0, center, -(gl + gr))
    return result


@dataclass
class ActionOutput:
    direction: Tensor
    proposal: Tensor
    uncapped_proposal: Tensor
    cap_active: Tensor
    graph_rms: Tensor
    bond_reliability: Tensor
    angle_reliability: Tensor


def action_proposal(
    source: Tensor,
    graphs: Sequence[GraphGeometry],
    prediction: Mapping[str, Tensor],
    *,
    tau: float,
    atom_cap: float,
    reliability_head: PrimitiveReliabilityHead | None,
) -> ActionOutput:
    """Build a first-order reliability-differentiable Cartesian action.

    Primitive geometry derivatives, source defects, mu and sigma are detached.
    Only the reliability gates (and their shared-backbone features) retain the
    action-loss graph.  No second-order Cartesian derivative is constructed.
    """
    atom_offsets, bond_offsets, angle_offsets = _offsets(graphs)
    bond_q, angle_q = geometry_values(source, _collated_geometry_view(graphs, source.device))
    if reliability_head is None:
        bond_r = prediction["bond_mu"].new_ones(prediction["bond_mu"].shape)
        angle_r = prediction["angle_mu"].new_ones(prediction["angle_mu"].shape)
    else:
        bond_r, angle_r = reliability_head(
            prediction["bond_features"],
            prediction["angle_features"],
            bond_q.detach().to(prediction["bond_mu"]),
            angle_q.detach().to(prediction["angle_mu"]),
            prediction["bond_mu"].detach(),
            prediction["angle_mu"].detach(),
            prediction["bond_sigma"].detach(),
            prediction["angle_sigma"].detach(),
        )

    directions = []
    for i, graph in enumerate(graphs):
        # The persisted GraphGeometry objects are CPU-resident.  Keep the
        # immutable offsets above, but move local index tensors alongside the
        # Cartesian coordinates for CUDA action construction.
        graph = graph.to(source.device)
        aa = slice(atom_offsets[i], atom_offsets[i + 1])
        bb = slice(bond_offsets[i], bond_offsets[i + 1])
        cc = slice(angle_offsets[i], angle_offsets[i + 1])
        coordinates = source[aa]
        nb = max(1, bond_offsets[i + 1] - bond_offsets[i])
        na = max(1, angle_offsets[i + 1] - angle_offsets[i])
        bond_coefficient = (
            0.5
            / nb
            * bond_r[bb].to(coordinates)
            * (bond_q[bb].detach() - prediction["bond_mu"][bb].detach().to(bond_q))
            / prediction["bond_sigma"][bb].detach().to(bond_q).square()
        )
        angle_coefficient = (
            0.5
            / na
            * angle_r[cc].to(coordinates)
            * (angle_q[cc].detach() - prediction["angle_mu"][cc].detach().to(angle_q))
            / prediction["angle_sigma"][cc].detach().to(angle_q).square()
        )
        vector = _bond_descent(coordinates, graph, bond_coefficient)
        vector = vector + _angle_descent(coordinates, graph, angle_coefficient)
        projected = remove_rigid_component(vector, coordinates.detach())
        rms = projected.square().sum(-1).mean().sqrt()
        safe_rms = rms.clamp_min(1.0e-14)
        direction = projected / safe_rms
        direction = torch.where(
            (torch.isfinite(projected).all() & (rms > 1.0e-14)),
            direction,
            torch.zeros_like(direction),
        )
        directions.append(direction)
    direction = torch.cat(directions)
    uncapped = source + float(tau) * direction
    delta = uncapped - source
    norms = torch.linalg.vector_norm(delta, dim=-1)
    scales = torch.clamp(
        delta.new_tensor(float(atom_cap)) / norms.clamp_min(1.0e-15), max=1.0
    )
    proposal = source + delta * scales[:, None]
    cap_values, rms_values = [], []
    for i in range(len(graphs)):
        aa = slice(atom_offsets[i], atom_offsets[i + 1])
        cap_values.append((scales[aa] < 1.0).any().to(source))
        local_delta = proposal[aa] - source[aa]
        rms_values.append(local_delta.square().sum(-1).mean().sqrt())
    return ActionOutput(
        direction=direction,
        proposal=proposal,
        uncapped_proposal=uncapped,
        cap_active=torch.stack(cap_values),
        graph_rms=torch.stack(rms_values),
        bond_reliability=bond_r,
        angle_reliability=angle_r,
    )


def action_loss(
    action: ActionOutput,
    reference: Tensor,
    graphs: Sequence[GraphGeometry],
    prediction: Mapping[str, Tensor],
) -> Tensor:
    view = _collated_geometry_view(graphs, action.proposal.device)
    proposal_bond, proposal_angle = geometry_values(action.proposal, view)
    reference_bond, reference_angle = geometry_values(reference, view)
    bond = (
        (proposal_bond - reference_bond)
        / prediction["bond_sigma"].detach().to(proposal_bond)
    ).square()
    angle = (
        (proposal_angle - reference_angle)
        / prediction["angle_sigma"].detach().to(proposal_angle)
    ).square()
    return molecule_equal_family_mean(bond, angle, graphs)


def _collated_geometry_view(
    graphs: Sequence[GraphGeometry], device: torch.device
) -> GraphGeometry:
    atoms, edges, edge_codes = [], [], []
    bonds, bond_codes, angles, angle_codes = [], [], [], []
    fixed_b, fixed_a, offset = [], [], 0
    for graph in graphs:
        graph = graph.to(device)
        atoms.append(graph.atom_categorical)
        edges.append(graph.edge_index + offset)
        edge_codes.append(graph.edge_categorical)
        bonds.append(graph.bonds + offset)
        bond_codes.append(graph.bond_categorical)
        angles.append(graph.angles + offset)
        angle_codes.append(graph.angle_edge_categorical)
        fixed_b.append(graph.bond_fixed)
        fixed_a.append(graph.angle_fixed)
        offset += graph.atom_categorical.size(0)
    return GraphGeometry(
        atom_categorical=torch.cat(atoms),
        edge_index=torch.cat(edges, 1),
        edge_categorical=torch.cat(edge_codes),
        bonds=torch.cat(bonds, 1),
        bond_categorical=torch.cat(bond_codes),
        angles=torch.cat(angles),
        angle_edge_categorical=torch.cat(angle_codes),
        rings=(),
        chirality=(),
        bond_fixed=torch.cat(fixed_b),
        angle_fixed=torch.cat(fixed_a),
        ring_fixed=None,
        backoff=None,
    )
