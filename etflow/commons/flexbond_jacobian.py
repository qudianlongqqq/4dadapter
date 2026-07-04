"""Public bond-local Jacobian API for the generator-agnostic adapter.

This module wraps the geometry already validated by the ETFlow Jacobian
prototype.  It has no dependency on the ETFlow generator or its backbone.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .jacobian_4d_selection import select_jacobian_4d_bonds
from .jacobian_4d_velocity import (
    apply_jacobian_4d_correction,
    build_bond_frames,
    solve_q_targets,
)


def identify_target_bonds(
    rotatable_bond_index: Tensor,
    atom_bond_influence_index: Tensor,
    batch: Optional[Tensor] = None,
    *,
    min_affected_atoms: int = 2,
    max_bonds_per_mol: int = 16,
) -> dict[str, Tensor]:
    """Select oriented rotatable bonds and their affected-side atoms."""

    selected = select_jacobian_4d_bonds(
        rotatable_bond_index,
        atom_bond_influence_index,
        batch,
        min_affected_atoms=min_affected_atoms,
        max_bonds_per_mol=max_bonds_per_mol,
    )
    num_bonds = rotatable_bond_index.size(1)
    full_counts = torch.bincount(
        atom_bond_influence_index[1], minlength=num_bonds
    )
    selected["num_input_bonds"] = torch.tensor(
        num_bonds, device=rotatable_bond_index.device
    )
    selected["num_skipped_too_small"] = (
        full_counts < min_affected_atoms
    ).sum()
    selected["num_skipped_by_cap"] = torch.tensor(
        max(
            int((full_counts >= min_affected_atoms).sum().item())
            - int(selected["anchor_index"].numel()),
            0,
        ),
        device=rotatable_bond_index.device,
    )
    return selected


def build_bond_local_frame(
    x: Tensor,
    anchor_index: Tensor,
    moving_index: Tensor,
    affected_atom_index: Tensor,
    affected_bond_index: Tensor,
    *,
    eps: float = 1.0e-8,
) -> tuple[Tensor, Tensor]:
    """Build SO(3)-consistent frames from bond and affected-side geometry."""

    return build_bond_frames(
        x,
        anchor_index,
        moving_index,
        affected_atom_index,
        affected_bond_index,
        eps=eps,
    )


def apply_bond_jacobian(
    x: Tensor,
    q_b: Tensor,
    target_bonds: dict[str, Tensor],
    *,
    eps: float = 1.0e-8,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Map local scalar coefficients ``[s, w1, w2, w3]`` to atom velocities."""

    correction, counts, valid = apply_jacobian_4d_correction(
        x,
        q_b,
        target_bonds["anchor_index"],
        target_bonds["moving_index"],
        target_bonds["affected_atom_index"],
        target_bonds["affected_bond_index"],
        eps=eps,
    )
    return correction, {
        "atom_contribution_count": counts,
        "valid_geometry_mask": valid,
    }


def aggregate_bond_corrections(
    atom_contributions: Tensor,
    atom_index: Tensor,
    num_atoms: int,
) -> tuple[Tensor, Tensor]:
    """Average arbitrary per-contribution velocities at their target atoms."""

    if atom_contributions.ndim != 2 or atom_contributions.size(1) != 3:
        raise ValueError("atom_contributions must have shape [K, 3].")
    if atom_index.ndim != 1 or atom_index.numel() != atom_contributions.size(0):
        raise ValueError("atom_index must have shape [K].")
    output = atom_contributions.new_zeros((num_atoms, 3))
    counts = atom_contributions.new_zeros((num_atoms, 1))
    output.index_add_(0, atom_index, atom_contributions)
    counts.index_add_(
        0, atom_index, atom_contributions.new_ones((atom_index.numel(), 1))
    )
    return output / counts.clamp_min(1), counts


@torch.no_grad()
def solve_q_star_least_squares(
    x: Tensor,
    residual_velocity: Tensor,
    target_bonds: dict[str, Tensor],
    *,
    ridge_eps: float = 1.0e-5,
    max_q_norm: float = 10.0,
    max_condition: float = 1.0e6,
) -> tuple[Tensor, Tensor, dict[str, Tensor | int]]:
    """Construct robust training-only pseudo-labels for the 4D head.

    ``q_b_star`` is a training-time pseudo-label.  It depends on the true
    residual velocity and therefore must never be used during inference.
    The no-grad decorator and detached residual enforce that boundary.
    """

    residual = residual_velocity.detach()
    q_star, valid, conditions = solve_q_targets(
        x.detach(),
        residual,
        target_bonds["anchor_index"],
        target_bonds["moving_index"],
        target_bonds["affected_atom_index"],
        target_bonds["affected_bond_index"],
        ridge_eps=ridge_eps,
        max_q_norm=max_q_norm,
        max_condition=max_condition,
    )
    num_bonds = q_star.size(0)
    affected_count = torch.bincount(
        target_bonds["affected_bond_index"], minlength=num_bonds
    )
    finite = torch.isfinite(q_star).all(dim=-1)
    stats: dict[str, Tensor | int] = {
        "num_valid_bonds": int(valid.sum().item()),
        "num_skipped_too_small": int(
            target_bonds.get("num_skipped_too_small", (affected_count == 0).sum()).item()
        ),
        "num_skipped_rank_deficient": int(
            ((~valid) & (affected_count > 0)).sum().item()
        ),
        "q_star_nan_count": int((~finite).sum().item()),
        "condition_numbers": conditions,
    }
    q_star = torch.where(finite[:, None], q_star, torch.zeros_like(q_star))
    return q_star, valid, stats


@torch.no_grad()
def jacobian_sanity_check(
    x: Tensor,
    q_b: Tensor,
    target_bonds: dict[str, Tensor],
) -> dict[str, object]:
    """Check Jacobian shapes, finiteness, and valid-bond counts."""

    correction, diagnostics = apply_bond_jacobian(x, q_b, target_bonds)
    valid = diagnostics["valid_geometry_mask"]
    expected_q_shape = (target_bonds["anchor_index"].numel(), 4)
    return {
        "q_shape_ok": tuple(q_b.shape) == expected_q_shape,
        "correction_shape_ok": correction.shape == x.shape,
        "finite": bool(torch.isfinite(correction).all()),
        "num_target_bonds": expected_q_shape[0],
        "num_valid_bonds": int(valid.sum().item()),
    }
