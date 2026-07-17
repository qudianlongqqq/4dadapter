"""Local conflict-aware Cartesian/Bond fusion diagnostics for MCVR Stage H0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor


VARIANTS = {
    "H0_BASE_D1B": ("base", 0.0, False),
    "H0_SIGN_SAFE_ONLY": ("sign_safe", 0.0, False),
    "H0_PAIRWISE_L025": ("pairwise", 0.25, False),
    "H0_PAIRWISE_L050": ("pairwise", 0.50, False),
    "H0_PAIRWISE_L100": ("pairwise", 1.00, False),
    "H0_MINNORM_L025": ("minnorm", 0.25, False),
    "H0_MINNORM_L050": ("minnorm", 0.50, False),
    "H0_MINNORM_L100": ("minnorm", 1.00, False),
    "H0_MINNORM_NONRING_L025": ("minnorm", 0.25, True),
    "H0_MINNORM_NONRING_L050": ("minnorm", 0.50, True),
    "H0_MINNORM_NONRING_L100": ("minnorm", 1.00, True),
}


@dataclass(frozen=True)
class ProjectionInfo:
    rank: int = 0
    condition: float = 0.0
    fallback: str = "none"
    conflicts: int = 0


def bond_axes(coordinates: Tensor, bonds: Tensor, *, eps: float = 1.0e-12) -> Tensor:
    displacement = coordinates[bonds[1]] - coordinates[bonds[0]]
    return displacement / torch.linalg.vector_norm(displacement, dim=-1, keepdim=True).clamp_min(eps)


def axial_change(velocity: Tensor, bonds: Tensor, axes: Tensor) -> Tensor:
    return ((velocity[bonds[1]] - velocity[bonds[0]]) * axes).sum(dim=-1)


def local_conflict_mask(
    coordinates: Tensor,
    v_cart: Tensor,
    v_bond_safe: Tensor,
    bonds: Tensor,
    sign_safe_mask: Tensor,
    *,
    conflict_eps: float = 1.0e-10,
    eligible_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    axes = bond_axes(coordinates, bonds)
    cart_axial = axial_change(v_cart, bonds, axes)
    bond_axial = axial_change(v_bond_safe, bonds, axes)
    safe = sign_safe_mask.to(device=coordinates.device, dtype=torch.bool)
    if eligible_mask is not None:
        safe = safe & eligible_mask.to(device=coordinates.device, dtype=torch.bool)
    conflict = safe & (bond_axial.abs() > float(conflict_eps)) & (
        cart_axial * bond_axial < -float(conflict_eps)
    )
    return conflict, axes, cart_axial, bond_axial


def pairwise_conflict_removal(
    v_cart: Tensor,
    bonds: Tensor,
    axes: Tensor,
    cart_axial: Tensor,
    conflict_mask: Tensor,
    *,
    lambda_conflict: float,
) -> Tensor:
    if not bool(conflict_mask.any()) or float(lambda_conflict) == 0.0:
        return v_cart
    keep = torch.nonzero(conflict_mask, as_tuple=False).reshape(-1)
    selected_bonds = bonds[:, keep]
    correction = (
        0.5 * float(lambda_conflict) * cart_axial[keep, None] * axes[keep]
    )
    delta = torch.zeros_like(v_cart)
    delta.index_add_(0, selected_bonds[0], correction)
    delta.index_add_(0, selected_bonds[1], -correction)
    result = v_cart + delta
    if not torch.isfinite(result).all():
        raise FloatingPointError("pairwise conflict removal produced non-finite output")
    return result


def minimum_norm_conflict_projection(
    v_cart: Tensor,
    bonds: Tensor,
    axes: Tensor,
    conflict_mask: Tensor,
    *,
    lambda_conflict: float,
    ridge: float = 1.0e-10,
) -> tuple[Tensor, ProjectionInfo]:
    if not bool(conflict_mask.any()) or float(lambda_conflict) == 0.0:
        return v_cart, ProjectionInfo(conflicts=int(conflict_mask.sum()))
    keep = torch.nonzero(conflict_mask, as_tuple=False).reshape(-1)
    local_bonds = bonds[:, keep]
    local_axes = axes[keep].to(torch.float64)
    rows = len(keep)
    matrix = torch.zeros(
        (rows, v_cart.shape[0] * 3), device=v_cart.device, dtype=torch.float64
    )
    row_index = torch.arange(rows, device=v_cart.device)[:, None].expand(-1, 3)
    xyz = torch.arange(3, device=v_cart.device)[None, :].expand(rows, -1)
    matrix[row_index, local_bonds[0, :, None] * 3 + xyz] = -local_axes
    matrix[row_index, local_bonds[1, :, None] * 3 + xyz] = local_axes
    velocity = v_cart.to(torch.float64).reshape(-1)
    y = matrix @ velocity
    gram = matrix @ matrix.T
    system = gram + float(ridge) * torch.eye(rows, device=gram.device, dtype=gram.dtype)
    rank = int(torch.linalg.matrix_rank(matrix))
    condition = float(torch.linalg.cond(system))
    fallback = "none"
    try:
        factor = torch.linalg.cholesky(system)
        weights = torch.cholesky_solve(y[:, None], factor).squeeze(-1)
    except RuntimeError:
        fallback = "lstsq"
        weights = torch.linalg.lstsq(system, y[:, None]).solution.squeeze(-1)
    delta = -(matrix.T @ weights).reshape_as(v_cart)
    result = v_cart + float(lambda_conflict) * delta.to(v_cart.dtype)
    if not torch.isfinite(result).all() or not torch.isfinite(delta).all():
        raise FloatingPointError("minimum-norm conflict projection produced non-finite output")
    return result, ProjectionInfo(
        rank=rank, condition=condition, fallback=fallback, conflicts=rows
    )


def fuse_conflict_aware(
    coordinates: Tensor,
    v_cart: Tensor,
    v_bond_safe: Tensor,
    bonds: Tensor,
    sign_safe_mask: Tensor,
    *,
    operator: str,
    lambda_conflict: float,
    conflict_eps: float,
    ridge: float,
    ring_mask: Tensor | None = None,
    nonring_only: bool = False,
) -> tuple[Tensor, dict[str, Any]]:
    if not all(torch.isfinite(value).all() for value in (coordinates, v_cart, v_bond_safe)):
        raise FloatingPointError("conflict-aware fusion received non-finite input")
    eligible = ~ring_mask.to(torch.bool) if nonring_only and ring_mask is not None else None
    conflict, axes, before, bond_axial = local_conflict_mask(
        coordinates, v_cart, v_bond_safe, bonds, sign_safe_mask,
        conflict_eps=conflict_eps, eligible_mask=eligible,
    )
    if operator == "pairwise":
        safe, info = pairwise_conflict_removal(
            v_cart, bonds, axes, before, conflict,
            lambda_conflict=lambda_conflict,
        ), ProjectionInfo(conflicts=int(conflict.sum()))
    elif operator == "minnorm":
        safe, info = minimum_norm_conflict_projection(
            v_cart, bonds, axes, conflict,
            lambda_conflict=lambda_conflict, ridge=ridge,
        )
    elif operator in {"base", "sign_safe"}:
        safe, info = v_cart, ProjectionInfo(conflicts=int(conflict.sum()))
    else:
        raise ValueError(f"unknown conflict operator: {operator}")
    after = axial_change(safe, bonds, axes)
    conflict_energy_before = float(before[conflict].square().sum())
    conflict_energy_after = float(after[conflict].square().sum())
    return safe, {
        "conflict_mask": conflict,
        "cart_axial_before": before,
        "cart_axial_after": after,
        "bond_axial": bond_axial,
        "total_active_bonds": int(sign_safe_mask.to(torch.bool).sum()),
        "total_conflict_bonds": int(conflict.sum()),
        "cartesian_axial_energy_before": float(before.square().sum()),
        "cartesian_axial_conflict_energy_before": conflict_energy_before,
        "cartesian_axial_conflict_energy_after": conflict_energy_after,
        "removed_conflict_energy_fraction": (
            1.0 - conflict_energy_after / max(conflict_energy_before, 1.0e-30)
            if conflict_energy_before else 0.0
        ),
        "bond_axial_energy": float(bond_axial.square().sum()),
        "final_axial_energy": float((after + bond_axial).square().sum()),
        "branch_dot_product_before": float((before * bond_axial).sum()),
        "branch_dot_product_after": float((after * bond_axial).sum()),
        "cancellation_energy_before": float(torch.relu(-(before * bond_axial)).sum()),
        "cancellation_energy_after": float(torch.relu(-(after * bond_axial)).sum()),
        "projection_rank": info.rank,
        "condition_number": info.condition,
        "solver_fallback": info.fallback,
        "non_finite_count": int((~torch.isfinite(safe)).sum()),
        "empty_conflict": int(not bool(conflict.any())),
    }


def stage_h0_decision(
    strong: Mapping[str, bool], weak: Mapping[str, bool], *, invalid: bool
) -> str:
    if invalid:
        return "STAGE_H0_INVALID"
    if any(strong.values()):
        return "STAGE_H0_STRONG_HEADROOM"
    if any(weak.values()):
        return "STAGE_H0_WEAK_HEADROOM"
    return "STAGE_H0_NO_HEADROOM"
