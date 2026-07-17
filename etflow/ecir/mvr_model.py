"""Cartesian MCVR model with rigid/local and conservative flexible heads."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from etflow.models.components.light_egnn_refiner import LightEGNNRefinerBackbone, _mlp

from .model import ECIRErrorEncoder, _atom_batch, _field, _pool
from .bond_explicit import batched_bond_projection, bounded_bond_residual
from .geometry import unique_bonds


def _graph_tensor(batch: Any, name: str, graphs: int, width: int, pos: Tensor) -> Tensor:
    value = _field(batch, name)
    if value is None:
        return pos.new_zeros((graphs, width))
    return torch.as_tensor(value, device=pos.device, dtype=pos.dtype).reshape(graphs, width)


def _equivariant_head(
    h: Tensor,
    base_velocity: Tensor,
    pos: Tensor,
    edge_index: Tensor,
    atom_batch: Tensor,
    graph_context: Tensor,
    base_head: nn.Module,
    edge_head: nn.Module,
) -> Tensor:
    context = graph_context[atom_batch]
    velocity = base_head(torch.cat([h, context], dim=-1)) * base_velocity
    src, dst = edge_index
    relative = pos[src] - pos[dst]
    scalar = edge_head(torch.cat([
        h[src], h[dst], graph_context[atom_batch[src]], relative.square().sum(-1, keepdim=True)
    ], dim=-1))
    velocity = velocity.clone()
    velocity.index_add_(0, dst, scalar * relative)
    return velocity


def trust_clip_velocity(
    velocity: Tensor,
    atom_batch: Tensor,
    *,
    max_atom_norm: float,
    max_graph_rms: float,
) -> Tensor:
    """Differentiable norm clipping without removing Cartesian directions."""

    norms = torch.linalg.vector_norm(velocity, dim=-1)
    atom_scale = torch.clamp(float(max_atom_norm) / norms.clamp_min(1.0e-12), max=1.0)
    clipped = velocity * atom_scale[:, None]
    graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
    energy = clipped.new_zeros(graphs)
    energy.index_add_(0, atom_batch, clipped.square().sum(-1))
    counts = torch.bincount(atom_batch, minlength=graphs).clamp_min(1).to(clipped.dtype)
    rms = torch.sqrt(energy / counts + 1.0e-12)
    graph_scale = torch.clamp(float(max_graph_rms) / rms.clamp_min(1.0e-12), max=1.0)
    return clipped * graph_scale[atom_batch, None]


class MCVRModel(nn.Module):
    """Shared equivariant backbone with two Cartesian repair branches.

    The output is always ``[N,3]`` Cartesian velocity. No four-dimensional
    coefficient head or strict Global4D fusion is used.
    """

    def __init__(
        self,
        atom_feature_dim: int = 10,
        edge_attr_dim: int = 1,
        hidden_dim: int = 128,
        edge_hidden_dim: int = 128,
        time_embedding_dim: int = 64,
        num_layers: int = 6,
        encoder_num_layers: int = 4,
        dropout: float = 0.0,
        cutoff: float = 10.0,
        error_embedding_dim: int = 32,
        deterministic_feature_dim: int = 10,
        rigid_scale: float = 1.0,
        torsion_scale: float = 0.25,
        high_flex_torsion_scale: float = 0.125,
        torsion_gate_fixed_zero: bool = False,
        conservative_torsion_gate: bool = False,
        torsion_uncertainty_max: float = 1.0,
        max_velocity_atom_norm: float = 0.12,
        max_velocity_graph_rms: float = 0.06,
        metadata_dropout: float = 0.5,
        bond_head_enabled: bool = False,
        bond_explicit_alpha: float = 0.0,
        max_abs_bond_residual: float = 0.05,
        bond_projection_damping: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if not (
            0.0 <= high_flex_torsion_scale <= torsion_scale
            and torsion_scale < rigid_scale
        ):
            raise ValueError("require high_flex_torsion_scale <= torsion_scale < rigid_scale")
        self.rigid_scale = float(rigid_scale)
        self.torsion_scale = float(torsion_scale)
        self.high_flex_torsion_scale = float(high_flex_torsion_scale)
        self.torsion_gate_fixed_zero = bool(torsion_gate_fixed_zero)
        self.conservative_torsion_gate = bool(conservative_torsion_gate)
        self.torsion_uncertainty_max = float(torsion_uncertainty_max)
        self.max_velocity_atom_norm = float(max_velocity_atom_norm)
        self.max_velocity_graph_rms = float(max_velocity_graph_rms)
        self.bond_head_enabled = bool(bond_head_enabled)
        self.bond_explicit_alpha = float(bond_explicit_alpha)
        self.max_abs_bond_residual = float(max_abs_bond_residual)
        self.bond_projection_damping = float(bond_projection_damping)
        self.error_encoder = ECIRErrorEncoder(
            atom_feature_dim=atom_feature_dim,
            edge_attr_dim=edge_attr_dim,
            hidden_dim=hidden_dim,
            edge_hidden_dim=edge_hidden_dim,
            time_embedding_dim=time_embedding_dim,
            num_layers=encoder_num_layers,
            dropout=dropout,
            cutoff=cutoff,
            metadata_dropout=metadata_dropout,
            error_embedding_dim=error_embedding_dim,
        )
        self.backbone = LightEGNNRefinerBackbone(
            atom_feature_dim, edge_attr_dim, hidden_dim, edge_hidden_dim,
            time_embedding_dim, num_layers, dropout, cutoff,
        )
        self.deterministic_embedding = _mlp(
            deterministic_feature_dim, hidden_dim, error_embedding_dim, dropout
        )
        context_dim = hidden_dim + 2 * error_embedding_dim + 6
        head_input = 2 * hidden_dim + context_dim + 1
        base_input = hidden_dim + context_dim
        self.rigid_base = _mlp(base_input, hidden_dim, 1, dropout)
        self.rigid_edge = _mlp(head_input, edge_hidden_dim, 1, dropout)
        self.torsion_base = _mlp(base_input, hidden_dim, 1, dropout)
        self.torsion_edge = _mlp(head_input, edge_hidden_dim, 1, dropout)
        self.rigid_gate = _mlp(context_dim, hidden_dim, 1, dropout)
        self.torsion_gate = _mlp(context_dim, hidden_dim, 1, dropout)
        self.global_safety_gate = _mlp(context_dim, hidden_dim, 1, dropout)
        self.uncertainty_head = _mlp(context_dim, hidden_dim, 1, dropout)
        self.error_auxiliary_head = _mlp(context_dim, hidden_dim, 6, dropout)
        if self.bond_head_enabled:
            bond_input = 2 * hidden_dim + edge_attr_dim + 1 + time_embedding_dim
            self.bond_explicit_head = _mlp(bond_input, edge_hidden_dim, 3, dropout)
            nn.init.zeros_(self.bond_explicit_head[-1].weight)
            nn.init.zeros_(self.bond_explicit_head[-1].bias)
            with torch.no_grad():
                self.bond_explicit_head[-1].bias[1:] = -2.0
        for module in (self.rigid_base, self.rigid_edge, self.torsion_base, self.torsion_edge):
            nn.init.zeros_(module[-1].weight)
            nn.init.zeros_(module[-1].bias)
        nn.init.constant_(self.rigid_gate[-1].bias, -2.0)
        nn.init.constant_(self.torsion_gate[-1].bias, -4.0)
        nn.init.constant_(self.global_safety_gate[-1].bias, 2.0)

    def forward(
        self,
        batch: Any,
        pos: Tensor,
        t: Tensor,
        *,
        deterministic_features: Tensor | None = None,
        upstream_metadata: Tensor | None = None,
        torsion_trust_remaining: Tensor | None = None,
    ) -> dict[str, Tensor]:
        atom_batch = _atom_batch(batch, pos)
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        time = torch.as_tensor(t, device=pos.device, dtype=pos.dtype).reshape(-1)
        if time.numel() == 1:
            time = time.expand(graphs)
        encoded = self.error_encoder(
            batch, pos, time, upstream_metadata=upstream_metadata
        )
        h, base_velocity, atom_time_embedding = self.backbone.encode(
            _field(batch, "node_attr"), pos, _field(batch, "edge_index"),
            _field(batch, "edge_attr"), time[atom_batch],
        )
        pooled = _pool(h, atom_batch, graphs)
        deterministic = deterministic_features
        if deterministic is None:
            deterministic = _field(batch, "deterministic_error_features")
        if deterministic is None:
            deterministic = pos.new_zeros((graphs, 10))
        deterministic = torch.as_tensor(
            deterministic, device=pos.device, dtype=pos.dtype
        ).reshape(graphs, -1)
        deterministic_embedding = self.deterministic_embedding(deterministic)
        uncertainty_embedding = torch.cat([
            encoded["error_logvar"].mean(-1, keepdim=True),
            encoded["error_mean"].mean(-1, keepdim=True),
            encoded["repair_gate"],
            deterministic[:, 6:7],
            deterministic[:, 8:9],
            deterministic[:, 9:10],
        ], dim=-1)
        context = torch.cat([
            pooled, encoded["error_embedding"], deterministic_embedding,
            uncertainty_embedding,
        ], dim=-1)
        edge_index = torch.as_tensor(_field(batch, "edge_index"), device=pos.device)
        rigid_velocity = _equivariant_head(
            h, base_velocity, pos, edge_index, atom_batch, context,
            self.rigid_base, self.rigid_edge,
        )
        torsion_velocity = _equivariant_head(
            h, base_velocity, pos, edge_index, atom_batch, context,
            self.torsion_base, self.torsion_edge,
        )
        rigid_gate = torch.sigmoid(self.rigid_gate(context))
        torsion_gate = torch.sigmoid(self.torsion_gate(context))
        uncertainty = torch.nn.functional.softplus(self.uncertainty_head(context))
        # Torsion repair vanishes without deterministic torsion evidence.
        torsion_evidence = (1.0 - torch.exp(-deterministic[:, 6:7].clamp_min(0.0)))
        torsion_gate = torsion_gate * torsion_evidence
        if self.conservative_torsion_gate:
            active = _field(batch, "active_mode_mask")
            if active is None:
                torsion_active = deterministic[:, 6:7] > 0.0
                clean = torch.zeros_like(torsion_active)
            else:
                active = torch.as_tensor(
                    active, device=pos.device, dtype=pos.dtype
                ).reshape(graphs, 6)
                torsion_active = active[:, 4:5] > 0.0
                clean = active[:, 5:6] > 0.0
            no_safety_risk = deterministic[:, 2:6].abs().amax(-1, keepdim=True) <= 0.0
            confident = uncertainty < self.torsion_uncertainty_max
            if torsion_trust_remaining is None:
                trust_available = torch.ones_like(torsion_active)
            else:
                trust_available = torch.as_tensor(
                    torsion_trust_remaining, device=pos.device, dtype=pos.dtype
                ).reshape(graphs, 1) > 0.0
            conservative_mask = (
                torsion_active & ~clean & no_safety_risk & confident & trust_available
            )
            torsion_gate = torsion_gate * conservative_mask.to(torsion_gate.dtype)
        if self.torsion_gate_fixed_zero:
            torsion_gate = torch.zeros_like(torsion_gate)
        high_flex = deterministic[:, 8:9] >= 1.0 - 1.0e-6
        torsion_scale = torch.where(
            high_flex,
            torsion_gate.new_full(torsion_gate.shape, self.high_flex_torsion_scale),
            torsion_gate.new_full(torsion_gate.shape, self.torsion_scale),
        )
        safety_gate = torch.sigmoid(self.global_safety_gate(context)) * torch.exp(-uncertainty).clamp(0.0, 1.0)
        v_rigid = self.rigid_scale * rigid_gate[atom_batch] * rigid_velocity
        v_torsion = torsion_scale[atom_batch] * torsion_gate[atom_batch] * torsion_velocity
        cartesian_raw = v_rigid + v_torsion
        bonds = unique_bonds(edge_index).to(pos.device)
        predicted_bond_residual = pos.new_empty(0)
        raw_bond_residual = pos.new_empty(0)
        unattenuated_bond_residual = pos.new_empty(0)
        bond_confidence_logit = pos.new_empty(0)
        bond_confidence = pos.new_empty(0)
        bond_uncertainty = pos.new_empty(0)
        bond_correction = torch.zeros_like(pos)
        bond_solver_failure = pos.new_zeros(graphs)
        if self.bond_head_enabled and bonds.numel():
            left, right = bonds
            edge_keep = edge_index[0] < edge_index[1]
            edge_attr = _field(batch, "edge_attr")
            if edge_attr is None:
                edge_attr = pos.new_zeros((edge_index.size(1), self.backbone.edge_attr_dim))
            edge_attr = torch.as_tensor(edge_attr, device=pos.device, dtype=pos.dtype)
            if edge_attr.ndim == 1:
                edge_attr = edge_attr[:, None]
            bond_features = torch.cat([
                h[left] + h[right],
                (h[left] - h[right]).abs(),
                edge_attr[edge_keep],
                torch.linalg.vector_norm(pos[right] - pos[left], dim=-1, keepdim=True),
                atom_time_embedding[left],
            ], dim=-1)
            bond_output = self.bond_explicit_head(bond_features)
            raw_bond_residual = bond_output[:, 0]
            bond_confidence_logit = bond_output[:, 1]
            bond_confidence = torch.sigmoid(bond_confidence_logit)
            bond_uncertainty = torch.nn.functional.softplus(bond_output[:, 2])
            predicted_bond_residual = bounded_bond_residual(
                raw_bond_residual, bond_confidence_logit,
                max_abs_residual=self.max_abs_bond_residual,
            )
            unattenuated_bond_residual = (
                self.max_abs_bond_residual * torch.tanh(raw_bond_residual)
            )
            bond_correction, bond_solver_failure = batched_bond_projection(
                pos, bonds, predicted_bond_residual, atom_batch,
                damping=self.bond_projection_damping,
            )
        raw = cartesian_raw + self.bond_explicit_alpha * bond_correction
        clipped = trust_clip_velocity(
            raw, atom_batch,
            max_atom_norm=self.max_velocity_atom_norm,
            max_graph_rms=self.max_velocity_graph_rms,
        )
        final = safety_gate[atom_batch] * clipped
        return {
            **encoded,
            "rigid_velocity": rigid_velocity,
            "torsion_velocity": torsion_velocity,
            "rigid_gate": rigid_gate,
            "torsion_gate": torsion_gate,
            "torsion_gate_active": (torsion_gate > 1.0e-8).to(torsion_gate.dtype),
            "v_rigid_contribution": v_rigid,
            "v_torsion_contribution": v_torsion,
            "v_cartesian_raw": cartesian_raw,
            "bond_indices": bonds,
            "bond_predicted_residual": predicted_bond_residual,
            "bond_raw_residual": raw_bond_residual,
            "bond_unattenuated_residual": unattenuated_bond_residual,
            "bond_confidence_logit": bond_confidence_logit,
            "bond_confidence": bond_confidence,
            "bond_uncertainty": bond_uncertainty,
            "v_bond_correction": bond_correction,
            "bond_solver_failure": bond_solver_failure,
            "global_safety_gate": safety_gate,
            "uncertainty": uncertainty,
            "error_logits": self.error_auxiliary_head(context),
            "v_raw": raw,
            "v_trust_clipped": clipped,
            "v_final": final,
            "velocity": final,
        }
