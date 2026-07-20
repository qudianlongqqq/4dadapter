"""Invariant upstream-error and bounded-confidence heads for MCVR V8."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from etflow.models.components.light_egnn_refiner import _mlp

from .model import _pool


class V8ErrorStateHead(nn.Module):
    """Predict atom/graph correction magnitudes and bounded prior confidence."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        confidence_min: float = 0.25,
        confidence_max: float = 4.0,
        confidence_mode: str = "learned_bounded",
        fixed_confidence: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 < confidence_min < confidence_max < math.inf:
            raise ValueError("confidence bounds must be finite, positive, and ordered")
        if confidence_mode not in {"fixed", "learned_bounded"}:
            raise ValueError(f"unknown confidence mode: {confidence_mode}")
        if not confidence_min <= fixed_confidence <= confidence_max:
            raise ValueError("fixed confidence must lie within configured bounds")
        self.hidden_dim = int(hidden_dim)
        self.confidence_min = float(confidence_min)
        self.confidence_max = float(confidence_max)
        self.confidence_mode = str(confidence_mode)
        self.fixed_confidence = float(fixed_confidence)
        self.atom_head = _mlp(hidden_dim, hidden_dim, 2, 0.0)
        self.graph_head = _mlp(hidden_dim, hidden_dim, 3, 0.0)
        nn.init.zeros_(self.atom_head[-1].weight)
        nn.init.zeros_(self.atom_head[-1].bias)
        nn.init.zeros_(self.graph_head[-1].weight)
        nn.init.zeros_(self.graph_head[-1].bias)
        neutral_fraction = (1.0 - self.confidence_min) / (self.confidence_max - self.confidence_min)
        neutral_fraction = min(max(neutral_fraction, 1.0e-6), 1.0 - 1.0e-6)
        neutral_logit = math.log(neutral_fraction / (1.0 - neutral_fraction))
        nn.init.constant_(self.atom_head[-1].bias[1], neutral_logit)

    def bounded_confidence(self, logits: Tensor) -> Tensor:
        if self.confidence_mode == "fixed":
            return logits.new_full(logits.shape, self.fixed_confidence)
        return self.confidence_min + (self.confidence_max - self.confidence_min) * torch.sigmoid(
            logits
        )

    def forward(self, node_features: Tensor, atom_batch: Tensor) -> dict[str, Tensor]:
        graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
        atom = self.atom_head(node_features)
        pooled = _pool(node_features, atom_batch, graphs)
        graph = self.graph_head(pooled)
        confidence_logit = atom[:, 1]
        confidence = self.bounded_confidence(confidence_logit)
        return {
            "predicted_error_magnitude": torch.nn.functional.softplus(atom[:, 0]),
            "prior_confidence_logit": confidence_logit,
            "bounded_prior_confidence": confidence,
            "predicted_graph_correction_rms": torch.nn.functional.softplus(graph[:, 0]),
            "predicted_max_atom_correction": torch.nn.functional.softplus(graph[:, 1]),
            "upstream_error_embedding": graph[:, 2:3],
            "graph_features": pooled,
        }


def confidence_regularization(
    confidence: Tensor,
    *,
    confidence_min: float,
    confidence_max: float,
    saturation_margin_fraction: float = 0.05,
    minimum_batch_std: float = 0.01,
) -> Tensor:
    """Discourage global bound saturation while retaining useful variation."""

    if not confidence.numel():
        return confidence.new_zeros(())
    span = float(confidence_max - confidence_min)
    margin = float(saturation_margin_fraction) * span
    lower = torch.relu(confidence.new_tensor(confidence_min + margin) - confidence)
    upper = torch.relu(confidence - confidence.new_tensor(confidence_max - margin))
    variation = torch.relu(
        confidence.new_tensor(minimum_batch_std) - confidence.std(unbiased=False)
    )
    return lower.square().mean() + upper.square().mean() + variation.square()
