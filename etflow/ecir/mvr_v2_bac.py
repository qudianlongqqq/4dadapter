"""Unified Bond-Angle-Clash extension of the D1-B MCVR model."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from etflow.models.components.light_egnn_refiner import _mlp

from .audit import field
from .bac_constraints import (
    angle_equivariant_directions,
    sparse_clash_edges,
    stable_angle_cosine,
    standardized_interval_residual,
)
from .geometry import bond_angles, training_bond_index
from .mvr_model import MCVRModel, trust_clip_velocity


V2_A_BOND_ONLY = "V2_A_BOND_ONLY"
V2_B_BOND_ANGLE = "V2_B_BOND_ANGLE"
V2_C_BOND_CLASH = "V2_C_BOND_CLASH"
V2_D_BOND_ANGLE_CLASH = "V2_D_BOND_ANGLE_CLASH"
BAC_MODES = {
    V2_A_BOND_ONLY,
    V2_B_BOND_ANGLE,
    V2_C_BOND_CLASH,
    V2_D_BOND_ANGLE_CLASH,
}


def _scatter_constraint_vectors(
    atom_count: int,
    indices: Tensor,
    vectors: tuple[Tensor, ...],
    weights: Tensor,
    template: Tensor,
    count_mask: Tensor | None = None,
) -> Tensor:
    result = torch.zeros_like(template)
    counts = template.new_zeros(atom_count)
    count_values = (
        torch.ones_like(weights)
        if count_mask is None
        else torch.as_tensor(count_mask, device=weights.device, dtype=weights.dtype)
    )
    for column, direction in enumerate(vectors):
        atom_ids = indices[:, column]
        result.index_add_(0, atom_ids, weights[:, None] * direction)
        counts.index_add_(0, atom_ids, count_values)
    return result / counts.clamp_min(1.0)[:, None]


def _zero_last(module: nn.Sequential, *, bias: float = 0.0) -> None:
    nn.init.zeros_(module[-1].weight)
    nn.init.constant_(module[-1].bias, float(bias))


class MCVRBACModel(MCVRModel):
    """D1-B backbone plus sparse constraint fusion and one Cartesian output."""

    def __init__(
        self,
        *args: Any,
        bac_mode: str = V2_A_BOND_ONLY,
        bac_constraint_scale: float = 0.05,
        clash_cutoff: float = 2.0,
        clash_allowed_contact: float = 1.0,
        clash_exclude_topology_distance: int = 2,
        max_clash_edges_per_graph: int = 128,
        bac_active_constraint_normalization: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if bac_mode not in BAC_MODES:
            raise ValueError(f"unknown BAC mode: {bac_mode}")
        self.bac_mode = str(bac_mode)
        self.bac_constraint_scale = float(bac_constraint_scale)
        self.clash_cutoff = float(clash_cutoff)
        self.clash_allowed_contact = float(clash_allowed_contact)
        self.clash_exclude_topology_distance = int(
            clash_exclude_topology_distance
        )
        self.max_clash_edges_per_graph = int(max_clash_edges_per_graph)
        self.bac_active_constraint_normalization = bool(
            bac_active_constraint_normalization
        )
        self.angle_enabled = bac_mode in {
            V2_B_BOND_ANGLE,
            V2_D_BOND_ANGLE_CLASH,
        }
        self.clash_enabled = bac_mode in {
            V2_C_BOND_CLASH,
            V2_D_BOND_ANGLE_CLASH,
        }
        if not (self.angle_enabled or self.clash_enabled):
            return
        hidden_dim = int(self.backbone.atom_embedding.out_features)
        edge_hidden_dim = int(
            self.backbone.layers[0].message_mlp[0].out_features
        )
        if self.angle_enabled:
            # center, symmetric neighbors, neighbor difference, two lengths,
            # cosine, standardized residual, and severity.
            self.angle_constraint_encoder = _mlp(
                3 * hidden_dim + 5, edge_hidden_dim, hidden_dim, 0.0
            )
            self.angle_constraint_head = _mlp(
                hidden_dim, edge_hidden_dim, 3, 0.0
            )
            _zero_last(self.angle_constraint_head)
        if self.clash_enabled:
            self.clash_constraint_encoder = _mlp(
                2 * hidden_dim + 5, edge_hidden_dim, hidden_dim, 0.0
            )
            self.clash_constraint_head = _mlp(
                hidden_dim, edge_hidden_dim, 3, 0.0
            )
            _zero_last(self.clash_constraint_head)
        self.constraint_type_embedding = nn.Embedding(3, hidden_dim)
        nn.init.zeros_(self.constraint_type_embedding.weight)
        self.constraint_fusion = _mlp(
            hidden_dim + 4, hidden_dim, 2, 0.0
        )
        _zero_last(self.constraint_fusion, bias=-2.0)

    @property
    def has_bac_modules(self) -> bool:
        return self.angle_enabled or self.clash_enabled

    def _angle_branch(
        self, batch: Any, pos: Tensor, hidden: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        angles = field(batch, "active_angle_constraint_index")
        ranges = field(batch, "angle_allowed_range")
        if angles is None or ranges is None:
            return torch.zeros_like(pos), {
                "angle_strength": pos.new_empty(0),
                "angle_confidence": pos.new_empty(0),
                "angle_gate": pos.new_empty(0),
                "angle_standardized_residual": pos.new_empty(0),
            }
        angles = torch.as_tensor(angles, device=pos.device, dtype=torch.long)
        if angles.ndim == 2 and angles.size(0) == 3:
            angles = angles.t()
        angles = angles.reshape(-1, 3)
        ranges = torch.as_tensor(ranges, device=pos.device, dtype=pos.dtype)
        if not angles.numel():
            return torch.zeros_like(pos), {
                "angle_strength": pos.new_empty(0),
                "angle_confidence": pos.new_empty(0),
                "angle_gate": pos.new_empty(0),
                "angle_standardized_residual": pos.new_empty(0),
            }
        i, j, k = angles.unbind(-1)
        cosine = stable_angle_cosine(pos, angles)
        values = bond_angles(pos, angles)
        residual, severity = standardized_interval_residual(values, ranges)
        left_length = torch.linalg.vector_norm(pos[i] - pos[j], dim=-1)
        right_length = torch.linalg.vector_norm(pos[k] - pos[j], dim=-1)
        features = torch.cat(
            [
                hidden[j],
                hidden[i] + hidden[k],
                (hidden[i] - hidden[k]).abs(),
                left_length[:, None],
                right_length[:, None],
                cosine[:, None],
                residual[:, None],
                severity[:, None],
            ],
            dim=-1,
        )
        encoded = self.angle_constraint_encoder(features)
        raw = self.angle_constraint_head(encoded)
        strength = torch.tanh(raw[:, 0])
        confidence = torch.sigmoid(raw[:, 1])
        gate = torch.sigmoid(raw[:, 2])
        weights = (
            self.bac_constraint_scale
            * strength
            * confidence
            * gate
            * (severity > 0).to(pos.dtype)
        )
        directions = angle_equivariant_directions(pos, angles)
        correction = _scatter_constraint_vectors(
            pos.size(0),
            angles,
            directions,
            weights,
            pos,
            count_mask=(severity > 0)
            if self.bac_active_constraint_normalization
            else None,
        )
        return correction, {
            "angle_strength": strength,
            "angle_confidence": confidence,
            "angle_gate": gate,
            "angle_standardized_residual": residual,
        }

    def _clash_branch(
        self,
        batch: Any,
        pos: Tensor,
        hidden: Tensor,
        atom_batch: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        bonds = field(batch, "active_bond_constraint_index")
        if bonds is None:
            bonds = training_bond_index(batch, pos.device)
        clash = sparse_clash_edges(
            pos,
            bonds,
            atom_batch=atom_batch,
            cutoff=self.clash_cutoff,
            allowed_contact=self.clash_allowed_contact,
            exclude_topology_distance=self.clash_exclude_topology_distance,
            max_edges_per_graph=self.max_clash_edges_per_graph,
        )
        edges = clash["edge_index"]
        if not edges.numel():
            return torch.zeros_like(pos), {
                **clash,
                "clash_strength": pos.new_empty(0),
                "clash_confidence": pos.new_empty(0),
                "clash_gate": pos.new_empty(0),
            }
        left, right = edges
        topology = clash["topology_distance"].to(pos.dtype)
        features = torch.cat(
            [
                hidden[left] + hidden[right],
                (hidden[left] - hidden[right]).abs(),
                clash["distance"][:, None],
                clash["allowed_contact"][:, None],
                clash["penetration"][:, None],
                topology[:, None],
                clash["active_mask"].to(pos.dtype)[:, None],
            ],
            dim=-1,
        )
        encoded = self.clash_constraint_encoder(features)
        raw = self.clash_constraint_head(encoded)
        strength = torch.tanh(raw[:, 0])
        confidence = torch.sigmoid(raw[:, 1])
        gate = torch.sigmoid(raw[:, 2])
        weights = (
            self.bac_constraint_scale
            * strength
            * confidence
            * gate
            * clash["active_mask"].to(pos.dtype)
        )
        correction = torch.zeros_like(pos)
        counts = pos.new_zeros(pos.size(0))
        direction = clash["direction"]
        correction.index_add_(0, left, weights[:, None] * direction)
        correction.index_add_(0, right, -weights[:, None] * direction)
        count_values = (
            clash["active_mask"].to(pos.dtype)
            if self.bac_active_constraint_normalization
            else torch.ones_like(weights)
        )
        counts.index_add_(0, left, count_values)
        counts.index_add_(0, right, count_values)
        correction = correction / counts.clamp_min(1.0)[:, None]
        return correction, {
            **clash,
            "clash_strength": strength,
            "clash_confidence": confidence,
            "clash_gate": gate,
        }

    def forward(
        self,
        batch: Any,
        pos: Tensor,
        t: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        base = super().forward(batch, pos, t, **kwargs)
        if not self.has_bac_modules:
            return base
        hidden = base["node_embedding"]
        atom_batch = base["atom_batch"]
        angle_correction = torch.zeros_like(pos)
        clash_correction = torch.zeros_like(pos)
        diagnostics: dict[str, Tensor] = {}
        if self.angle_enabled:
            angle_correction, angle_diagnostics = self._angle_branch(
                batch, pos, hidden
            )
            diagnostics.update(angle_diagnostics)
        if self.clash_enabled:
            clash_correction, clash_diagnostics = self._clash_branch(
                batch, pos, hidden, atom_batch
            )
            diagnostics.update(clash_diagnostics)
        local = torch.stack(
            [
                torch.linalg.vector_norm(base["v_raw"], dim=-1),
                torch.linalg.vector_norm(angle_correction, dim=-1),
                torch.linalg.vector_norm(clash_correction, dim=-1),
                torch.linalg.vector_norm(
                    angle_correction + clash_correction, dim=-1
                ),
            ],
            dim=-1,
        )
        fusion_gate = torch.sigmoid(
            self.constraint_fusion(torch.cat([hidden, local], dim=-1))
        )
        angle_fused = fusion_gate[:, :1] * angle_correction
        clash_fused = fusion_gate[:, 1:2] * clash_correction
        unified_raw = base["v_raw"] + angle_fused + clash_fused
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
            **diagnostics,
            "v_angle_constraint": angle_correction,
            "v_clash_constraint": clash_correction,
            "v_angle_fused": angle_fused,
            "v_clash_fused": clash_fused,
            "constraint_fusion_gate": fusion_gate,
            "v_raw": unified_raw,
            "v_trust_clipped": unified_clipped,
            "v_final": unified_final,
            "velocity": unified_final,
            "unified_delta_count": pos.new_tensor(1, dtype=torch.long),
        }

    def load_d1b_state_dict(
        self, state_dict: dict[str, Tensor], *, strict: bool = True
    ) -> tuple[list[str], list[str]]:
        if self.bac_mode == V2_A_BOND_ONLY:
            incompatible = self.load_state_dict(state_dict, strict=strict)
            return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
        current = self.state_dict()
        current_keys = set(current)
        checkpoint_keys = set(state_dict)
        new_prefixes = (
            "angle_constraint_",
            "clash_constraint_",
            "constraint_type_embedding.",
            "constraint_fusion.",
        )
        missing_base = sorted(
            key
            for key in current_keys - checkpoint_keys
            if not key.startswith(new_prefixes)
        )
        unexpected = sorted(checkpoint_keys - current_keys)
        if strict and (missing_base or unexpected):
            raise RuntimeError(
                "D1-B base state mismatch: "
                f"missing={missing_base}, unexpected={unexpected}"
            )
        current.update(
            {key: value for key, value in state_dict.items() if key in current_keys}
        )
        self.load_state_dict(current, strict=True)
        return missing_base, unexpected
