"""Cartesian MCVR model with rigid/local and conservative flexible heads."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from etflow.models.components.light_egnn_refiner import LightEGNNRefinerBackbone, _mlp

from .model import ECIRErrorEncoder, _atom_batch, _field, _pool


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
        max_velocity_atom_norm: float = 0.12,
        max_velocity_graph_rms: float = 0.06,
        metadata_dropout: float = 0.5,
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
        self.max_velocity_atom_norm = float(max_velocity_atom_norm)
        self.max_velocity_graph_rms = float(max_velocity_graph_rms)
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
    ) -> dict[str, Tensor]:
        atom_batch = _atom_batch(batch, pos)
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        time = torch.as_tensor(t, device=pos.device, dtype=pos.dtype).reshape(-1)
        if time.numel() == 1:
            time = time.expand(graphs)
        encoded = self.error_encoder(
            batch, pos, time, upstream_metadata=upstream_metadata
        )
        h, base_velocity, _ = self.backbone.encode(
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
        # Torsion repair vanishes without deterministic torsion evidence.
        torsion_evidence = (1.0 - torch.exp(-deterministic[:, 6:7].clamp_min(0.0)))
        torsion_gate = torsion_gate * torsion_evidence
        high_flex = deterministic[:, 8:9] >= 1.0 - 1.0e-6
        torsion_scale = torch.where(
            high_flex,
            torsion_gate.new_full(torsion_gate.shape, self.high_flex_torsion_scale),
            torsion_gate.new_full(torsion_gate.shape, self.torsion_scale),
        )
        uncertainty = torch.nn.functional.softplus(self.uncertainty_head(context))
        safety_gate = torch.sigmoid(self.global_safety_gate(context)) * torch.exp(-uncertainty).clamp(0.0, 1.0)
        v_rigid = self.rigid_scale * rigid_gate[atom_batch] * rigid_velocity
        v_torsion = torsion_scale[atom_batch] * torsion_gate[atom_batch] * torsion_velocity
        raw = v_rigid + v_torsion
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
            "global_safety_gate": safety_gate,
            "uncertainty": uncertainty,
            "error_logits": self.error_auxiliary_head(context),
            "v_raw": raw,
            "v_final": final,
            "velocity": final,
        }
