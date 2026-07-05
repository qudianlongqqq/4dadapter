"""Small tensor-only utilities for stable post-generation refinement."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def clip_atom_displacement(
    displacement: Tensor, max_displacement: Optional[float]
) -> tuple[Tensor, Tensor]:
    """Clip each atom update independently and return the clipped-atom mask."""

    if max_displacement is None:
        return displacement, torch.zeros(
            displacement.size(0), dtype=torch.bool, device=displacement.device
        )
    limit = float(max_displacement)
    if limit <= 0:
        raise ValueError("max_displacement must be positive when provided.")
    norm = torch.linalg.norm(displacement, dim=-1, keepdim=True)
    clipped = norm.squeeze(-1) > limit
    scale = (limit / norm.clamp_min(1.0e-12)).clamp_max(1.0)
    return displacement * scale, clipped
