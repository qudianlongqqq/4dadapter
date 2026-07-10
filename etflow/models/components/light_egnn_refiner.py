"""Small E(n)-equivariant backbone for conformer refinement."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("time embedding dimension must be at least 2.")
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        t = t.reshape(-1)
        half = self.dim // 2
        frequencies = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        phase = t[:, None] * frequencies[None, :] * (2 * math.pi)
        embedding = torch.cat([phase.sin(), phase.cos()], dim=-1)
        if embedding.size(-1) < self.dim:
            embedding = torch.cat([embedding, t[:, None]], dim=-1)
        return embedding


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class LightEGNNLayer(nn.Module):
    """Invariant message update plus an equivariant relative-vector readout."""

    def __init__(
        self,
        hidden_dim: int = 128,
        edge_hidden_dim: int = 128,
        edge_attr_dim: int = 1,
        time_embedding_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        message_input = 2 * hidden_dim + edge_attr_dim + 1 + time_embedding_dim
        self.message_mlp = _mlp(
            message_input, edge_hidden_dim, hidden_dim, dropout
        )
        self.node_mlp = _mlp(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.vector_gate = _mlp(hidden_dim, edge_hidden_dim, 1, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        atom_time: Tensor,
        cutoff: float,
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index
        relative = pos[src] - pos[dst]
        distance_sq = relative.square().sum(dim=-1, keepdim=True)
        distance = distance_sq.clamp_min(1.0e-12).sqrt()
        keep = distance.squeeze(-1) <= cutoff
        src, dst = src[keep], dst[keep]
        relative, distance_sq = relative[keep], distance_sq[keep]
        edge_attr = edge_attr[keep]
        message = self.message_mlp(
            torch.cat(
                [h[src], h[dst], distance_sq, edge_attr, atom_time[src]], dim=-1
            )
        )
        aggregate = h.new_zeros(h.shape)
        aggregate.index_add_(0, dst, message)
        h = self.norm(h + self.node_mlp(torch.cat([h, aggregate], dim=-1)))
        scalar = self.vector_gate(message)
        vector = pos.new_zeros(pos.shape)
        vector.index_add_(0, dst, scalar * relative)
        return h, vector


class LightEGNNRefinerBackbone(nn.Module):
    """Predict equivariant Cartesian velocity and invariant per-bond 4D q."""

    def __init__(
        self,
        atom_feature_dim: int = 10,
        edge_attr_dim: int = 1,
        hidden_dim: int = 128,
        edge_hidden_dim: int = 128,
        time_embedding_dim: int = 64,
        num_layers: int = 6,
        dropout: float = 0.0,
        cutoff: float = 10.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive.")
        self.cutoff = float(cutoff)
        self.edge_attr_dim = int(edge_attr_dim)
        self.atom_embedding = nn.Linear(atom_feature_dim, hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.layers = nn.ModuleList(
            [
                LightEGNNLayer(
                    hidden_dim,
                    edge_hidden_dim,
                    edge_attr_dim,
                    time_embedding_dim,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.cartesian_layer_weights = nn.Parameter(torch.zeros(num_layers))
        bond_input = 2 * hidden_dim + 1 + time_embedding_dim
        self.q_head = _mlp(bond_input, edge_hidden_dim, 4, dropout)

    def encode(
        self,
        node_attr: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        atom_time: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return invariant node states, Cartesian velocity, and time embeddings."""

        if edge_attr is None:
            edge_attr = pos.new_zeros((edge_index.size(1), self.edge_attr_dim))
        edge_attr = edge_attr.to(dtype=pos.dtype)
        if edge_attr.ndim == 1:
            edge_attr = edge_attr[:, None]
        time_emb = self.time_embedding(atom_time.to(dtype=pos.dtype))
        h = self.atom_embedding(node_attr.to(dtype=pos.dtype))
        vectors = []
        for layer in self.layers:
            h, vector = layer(h, pos, edge_index, edge_attr, time_emb, cutoff=self.cutoff)
            vectors.append(vector)
        weights = torch.softmax(self.cartesian_layer_weights, dim=0)
        v_cart = sum(weight * vector for weight, vector in zip(weights, vectors))
        return h, v_cart, time_emb

    def forward(
        self,
        node_attr: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        atom_time: Tensor,
        anchor_index: Tensor,
        moving_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        h, v_cart, time_emb = self.encode(
            node_attr, pos, edge_index, edge_attr, atom_time
        )

        if anchor_index.numel() == 0:
            return v_cart, pos.new_empty((0, 4))
        bond_vector = pos[moving_index] - pos[anchor_index]
        bond_distance_sq = bond_vector.square().sum(dim=-1, keepdim=True)
        bond_features = torch.cat(
            [
                h[anchor_index],
                h[moving_index],
                bond_distance_sq,
                time_emb[anchor_index],
            ],
            dim=-1,
        )
        q_b = self.q_head(bond_features)
        return v_cart, q_b
