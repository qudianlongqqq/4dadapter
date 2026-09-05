"""Source-conditioned residual-geometry belief model for the SIXS-v2 prototype.

This module is intentionally separate from the frozen SIXS-v1 implementation.
It changes only the primitive prediction target and its source-coordinate
conditioning.  The existing Reliability, Adaptive-BA, first-order VJP action,
rigid projection, graph-RMS normalization, and unrestricted tau head are reused.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from .j1r1_full_joint_unrestricted import UnrestrictedFullJointModel
from .learned_geometry import GraphGeometry, LearnedGeometryObjective, geometry_values
from .musigma_reliability import (
    LOG_SIGMA_RATIO_LIMIT,
    molecule_equal_family_mean,
    primitive_features,
)


class SourceConditionedDeltaQBelief(nn.Module):
    """Predict signed primitive corrections and their residual scales.

    Each family head receives the unchanged v1 graph primitive representation
    concatenated with its current source primitive value.  Delta heads are
    zero-output initialized, which makes the initial effective target exactly
    the source geometry (a no-op residual prior).
    """

    def __init__(self, hidden_dim: int = 128, layers: int = 3) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.geometry = LearnedGeometryObjective(
            hidden_dim=hidden_dim, layers=layers, learned_sigma=False
        )
        edge_dim = 20
        self.geometry.bond_head = self._head(3 * hidden_dim + edge_dim + 1, hidden_dim)
        self.geometry.angle_head = self._head(4 * hidden_dim + 2 * edge_dim + 1, hidden_dim)
        self.bond_sigma_head = self._head(3 * hidden_dim + edge_dim + 1, hidden_dim)
        self.angle_sigma_head = self._head(4 * hidden_dim + 2 * edge_dim + 1, hidden_dim)
        for head in (
            self.geometry.bond_head,
            self.geometry.angle_head,
            self.bond_sigma_head,
            self.angle_sigma_head,
        ):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _head(input_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def backbone_parameters(self) -> list[nn.Parameter]:
        excluded = {id(parameter) for parameter in self.deltaq_head_parameters()}
        return [
            parameter
            for parameter in self.geometry.parameters()
            if id(parameter) not in excluded
        ]

    def deltaq_head_parameters(self) -> list[nn.Parameter]:
        return list(self.geometry.bond_head.parameters()) + list(
            self.geometry.angle_head.parameters()
        )

    # Compatibility with the v1 optimizer/group-reporting interface.
    def mu_head_parameters(self) -> list[nn.Parameter]:
        return self.deltaq_head_parameters()

    def sigma_head_parameters(self) -> list[nn.Parameter]:
        return list(self.bond_sigma_head.parameters()) + list(
            self.angle_sigma_head.parameters()
        )

    def _encode(self, graph: GraphGeometry) -> Tensor:
        geometry = self.geometry
        atoms = graph.atom_categorical
        node = geometry.atom_projection(
            torch.cat(
                (
                    geometry.element(atoms[:, 0]),
                    geometry.charge(atoms[:, 1]),
                    geometry.hybridization(atoms[:, 2]),
                    geometry.aromatic(atoms[:, 3]),
                    geometry.ring(atoms[:, 4]),
                    geometry.degree(atoms[:, 5]),
                ),
                dim=-1,
            )
        )
        directed_edge = geometry._edge(graph.edge_categorical)
        for layer in geometry.layers:
            node = layer(node, graph.edge_index, directed_edge)
        return node

    def forward(
        self,
        graph: GraphGeometry,
        source: Tensor,
        *,
        detach_sigma_features: bool = False,
    ) -> dict[str, Tensor]:
        node = self._encode(graph)
        bond_features, angle_features = primitive_features(self.geometry, graph, node)
        source_bond, source_angle = geometry_values(source, graph)
        source_bond_feature = source_bond.detach().to(bond_features)
        source_angle_feature = source_angle.detach().to(angle_features)
        conditioned_bond = torch.cat((bond_features, source_bond_feature[:, None]), dim=-1)
        conditioned_angle = torch.cat((angle_features, source_angle_feature[:, None]), dim=-1)

        bond_deltaq = self.geometry.bond_head(conditioned_bond).squeeze(-1)
        angle_deltaq = (
            self.geometry.angle_head(conditioned_angle).squeeze(-1)
            if conditioned_angle.numel()
            else bond_deltaq.new_empty(0)
        )
        sigma_bond_features = conditioned_bond.detach() if detach_sigma_features else conditioned_bond
        sigma_angle_features = conditioned_angle.detach() if detach_sigma_features else conditioned_angle
        bond_raw = self.bond_sigma_head(sigma_bond_features).squeeze(-1)
        angle_raw = (
            self.angle_sigma_head(sigma_angle_features).squeeze(-1)
            if sigma_angle_features.numel()
            else bond_raw.new_empty(0)
        )
        bond_log_ratio = LOG_SIGMA_RATIO_LIMIT * torch.tanh(
            bond_raw / LOG_SIGMA_RATIO_LIMIT
        )
        angle_log_ratio = LOG_SIGMA_RATIO_LIMIT * torch.tanh(
            angle_raw / LOG_SIGMA_RATIO_LIMIT
        )
        bond_stat = graph.bond_fixed[:, 1].to(bond_log_ratio)
        angle_stat = graph.angle_fixed[:, 1].to(angle_log_ratio)
        bond_sigma = bond_stat * torch.exp(bond_log_ratio)
        angle_sigma = angle_stat * torch.exp(angle_log_ratio)
        # Effective absolute means are exposed only for compatibility with the
        # unchanged v1 action and Reliability interfaces.
        bond_mu_eff = source_bond_feature + bond_deltaq
        angle_mu_eff = source_angle_feature + angle_deltaq
        return {
            "node_embedding": node,
            "bond_features": bond_features,
            "angle_features": angle_features,
            "bond_source_q": source_bond_feature,
            "angle_source_q": source_angle_feature,
            "bond_deltaq": bond_deltaq,
            "angle_deltaq": angle_deltaq,
            "bond_mu": bond_mu_eff,
            "angle_mu": angle_mu_eff,
            "bond_log_sigma_ratio": bond_log_ratio,
            "angle_log_sigma_ratio": angle_log_ratio,
            "bond_sigma": bond_sigma,
            "angle_sigma": angle_sigma,
        }


class DeltaQUnrestrictedFullJointModel(UnrestrictedFullJointModel):
    """Architecture-matched unrestricted Full Joint model with Delta-Q belief."""

    def __init__(self, hidden_dim: int = 128, layers: int = 3) -> None:
        super().__init__(hidden_dim, layers)
        self.belief = SourceConditionedDeltaQBelief(hidden_dim, layers)

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups = super().parameter_groups()
        groups["deltaq"] = groups.pop("mu")
        return groups


def deltaq_targets(
    source: Tensor, reference: Tensor, graph: GraphGeometry
) -> tuple[Tensor, Tensor]:
    """Return signed Bond-A and Angle-cosine reference-minus-source targets."""
    source_bond, source_angle = geometry_values(source, graph)
    reference_bond, reference_angle = geometry_values(reference, graph)
    return reference_bond - source_bond, reference_angle - source_angle


def deltaq_belief_loss(
    prediction: Mapping[str, Tensor],
    target_bond_deltaq: Tensor,
    target_angle_deltaq: Tensor,
    graphs: Sequence[GraphGeometry],
    beta: float = 0.5,
) -> tuple[Tensor, dict[str, Tensor]]:
    """The frozen v1 J1 beta-NLL applied to signed correction residuals.

    This preserves the exact v1 stop-gradient semantics:
    ``NLL * stop_gradient(sigma**2)**beta``.
    """

    def nll(target: Tensor, mean: Tensor, sigma: Tensor) -> Tensor:
        return 0.5 * ((target - mean) / sigma).square() + sigma.log()

    bond_sigma = prediction["bond_sigma"]
    angle_sigma = prediction["angle_sigma"]
    bond = nll(
        target_bond_deltaq.to(prediction["bond_deltaq"]),
        prediction["bond_deltaq"],
        bond_sigma,
    ) * bond_sigma.square().detach().pow(float(beta))
    angle = nll(
        target_angle_deltaq.to(prediction["angle_deltaq"]),
        prediction["angle_deltaq"],
        angle_sigma,
    ) * angle_sigma.square().detach().pow(float(beta))
    total = molecule_equal_family_mean(bond, angle, graphs)
    return total, {"bond": bond.mean(), "angle": angle.mean()}
