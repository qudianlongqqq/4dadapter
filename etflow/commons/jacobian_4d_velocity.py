"""Geometry and online pseudo-label utilities for the 4D Jacobian prototype."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def build_local_frame(
    axis: Tensor,
    reference: Optional[Tensor] = None,
    eps: float = 1.0e-8,
) -> Tensor:
    """Build right-handed frames with columns ``[e0, e1, e2]``."""

    if axis.ndim < 1 or axis.size(-1) != 3:
        raise ValueError(f"axis must end in dimension 3, got {tuple(axis.shape)}.")
    e0 = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(eps)
    if reference is None:
        # Generic fallback used by standalone callers. The model path supplies
        # an affected-side reference so its frame rotates with the molecule.
        reference_index = e0.detach().abs().argmin(dim=-1)
        reference = F.one_hot(reference_index, num_classes=3).to(
            device=axis.device, dtype=axis.dtype
        )
        e1 = torch.cross(e0, reference, dim=-1)
    else:
        if reference.shape != axis.shape:
            raise ValueError(
                "reference must have the same shape as axis, got "
                f"{tuple(reference.shape)} and {tuple(axis.shape)}."
            )
        reference = reference.to(device=axis.device, dtype=axis.dtype)
        e1 = reference - (reference * e0).sum(dim=-1, keepdim=True) * e0
    e1 = e1 / torch.linalg.norm(e1, dim=-1, keepdim=True).clamp_min(eps)
    e2 = torch.cross(e0, e1, dim=-1)
    return torch.stack([e0, e1, e2], dim=-1)


def build_bond_frames(
    pos: Tensor,
    anchor_index: Tensor,
    moving_index: Tensor,
    affected_atom_index: Tensor,
    affected_bond_index: Tensor,
    *,
    eps: float = 1.0e-8,
) -> Tuple[Tensor, Tensor]:
    """Build rotation-covariant frames from bond axes and affected-side atoms.

    For each bond, the affected atom with the largest perpendicular lever arm
    supplies ``e1``. Bonds without a finite non-collinear reference are marked
    invalid instead of falling back to a laboratory-frame Cartesian axis.
    """

    num_bonds = anchor_index.numel()
    if moving_index.shape != anchor_index.shape:
        raise ValueError("anchor_index and moving_index must have matching shapes.")
    if affected_atom_index.shape != affected_bond_index.shape:
        raise ValueError(
            "affected atom and bond index arrays must have matching shapes."
        )
    if num_bonds == 0:
        return pos.new_empty((0, 3, 3)), torch.empty(
            0, dtype=torch.bool, device=pos.device
        )

    bond_vector = pos[moving_index] - pos[anchor_index]
    bond_norm = torch.linalg.norm(bond_vector, dim=-1)
    e0 = bond_vector / bond_norm.unsqueeze(-1).clamp_min(eps)
    reference = torch.zeros_like(bond_vector)
    has_reference = torch.zeros(num_bonds, dtype=torch.bool, device=pos.device)

    if affected_atom_index.numel():
        lever = (
            pos[affected_atom_index]
            - pos[anchor_index[affected_bond_index]]
        )
        perpendicular = lever - (
            lever * e0[affected_bond_index]
        ).sum(dim=-1, keepdim=True) * e0[affected_bond_index]
        perpendicular_norm_sq = perpendicular.square().sum(dim=-1)
        max_norm_sq = pos.new_zeros((num_bonds,))
        max_norm_sq.scatter_reduce_(
            0,
            affected_bond_index,
            perpendicular_norm_sq,
            reduce="amax",
            include_self=True,
        )
        contribution_index = torch.arange(
            affected_atom_index.numel(),
            dtype=torch.long,
            device=pos.device,
        )
        sentinel = contribution_index.new_full(
            contribution_index.shape, affected_atom_index.numel()
        )
        candidate_index = torch.where(
            perpendicular_norm_sq == max_norm_sq[affected_bond_index],
            contribution_index,
            sentinel,
        )
        chosen_index = torch.full(
            (num_bonds,),
            affected_atom_index.numel(),
            dtype=torch.long,
            device=pos.device,
        )
        chosen_index.scatter_reduce_(
            0,
            affected_bond_index,
            candidate_index,
            reduce="amin",
            include_self=True,
        )
        has_reference = chosen_index < affected_atom_index.numel()
        safe_index = chosen_index.clamp_max(affected_atom_index.numel() - 1)
        reference = torch.where(
            has_reference[:, None], perpendicular[safe_index], reference
        )

    reference_norm = torch.linalg.norm(reference, dim=-1)
    valid = (
        torch.isfinite(bond_vector).all(dim=-1)
        & torch.isfinite(reference).all(dim=-1)
        & (bond_norm > eps)
        & has_reference
        & (reference_norm > eps)
    )
    return build_local_frame(bond_vector, reference=reference, eps=eps), valid


def skew_matrix(vector: Tensor) -> Tensor:
    """Return the matrix ``[vector]_x`` for batched 3-vectors."""

    if vector.ndim < 1 or vector.size(-1) != 3:
        raise ValueError(
            f"vector must end in dimension 3, got {tuple(vector.shape)}."
        )
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)


def build_atom_jacobian(e0: Tensor, frame: Tensor, lever: Tensor) -> Tensor:
    """Construct ``J_a = [e0, -skew(lever) @ frame]``."""

    if e0.shape != lever.shape or e0.size(-1) != 3:
        raise ValueError(
            "e0 and lever must have the same [..., 3] shape, got "
            f"{tuple(e0.shape)} and {tuple(lever.shape)}."
        )
    if frame.shape != (*e0.shape[:-1], 3, 3):
        raise ValueError(
            "frame must have shape [..., 3, 3] matching e0, got "
            f"{tuple(frame.shape)}."
        )
    rotation = -torch.matmul(skew_matrix(lever), frame)
    return torch.cat([e0.unsqueeze(-1), rotation], dim=-1)


def apply_jacobian_4d_correction(
    pos: Tensor,
    q_pred: Tensor,
    anchor_index: Tensor,
    moving_index: Tensor,
    affected_atom_index: Tensor,
    affected_bond_index: Tensor,
    *,
    eps: float = 1.0e-8,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Map per-bond 4D predictions to averaged atom-wise corrections.

    Returns ``(v_corr, atom_contribution_count, valid_geometry_mask)``.
    """

    if pos.ndim != 2 or pos.size(1) != 3:
        raise ValueError(f"pos must have shape [N, 3], got {tuple(pos.shape)}.")
    num_bonds = anchor_index.numel()
    if q_pred.shape != (num_bonds, 4):
        raise ValueError(
            f"q_pred must have shape [{num_bonds}, 4], got {tuple(q_pred.shape)}."
        )
    if moving_index.shape != anchor_index.shape:
        raise ValueError("anchor_index and moving_index must have matching shapes.")
    if affected_atom_index.shape != affected_bond_index.shape:
        raise ValueError(
            "affected atom and bond index arrays must have matching shapes."
        )

    v_corr = torch.zeros_like(pos)
    counts = pos.new_zeros((pos.size(0), 1))
    if num_bonds == 0:
        return v_corr, counts, torch.empty(
            0, dtype=torch.bool, device=pos.device
        )

    frame, valid_geometry = build_bond_frames(
        pos,
        anchor_index,
        moving_index,
        affected_atom_index,
        affected_bond_index,
        eps=eps,
    )
    e0 = frame[:, :, 0]
    omega_global = torch.matmul(
        frame, q_pred[:, 1:].unsqueeze(-1)
    ).squeeze(-1)

    if affected_atom_index.numel():
        bond_for_atom = affected_bond_index
        lever = (
            pos[affected_atom_index]
            - pos[anchor_index[bond_for_atom]]
        )
        stretch = q_pred[bond_for_atom, :1] * e0[bond_for_atom]
        rotation = torch.cross(
            omega_global[bond_for_atom], lever, dim=-1
        )
        contribution_valid = valid_geometry[bond_for_atom]
        contribution = (stretch + rotation) * contribution_valid.unsqueeze(-1)
        v_corr.index_add_(0, affected_atom_index, contribution)
        count_values = contribution_valid.to(dtype=pos.dtype).unsqueeze(-1)
        counts.index_add_(0, affected_atom_index, count_values)
        v_corr = v_corr / counts.clamp_min(1.0)

    return v_corr, counts, valid_geometry


def combine_jacobian_4d_velocity(
    v_atom: Tensor,
    v_corr: Tensor,
    correction_scale,
    warmup_scale=1.0,
    *,
    enabled: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Combine the original velocity and correction without Python-float casts."""

    if v_atom.shape != v_corr.shape:
        raise ValueError(
            f"v_atom and v_corr shapes differ: {v_atom.shape} vs {v_corr.shape}."
        )
    if not enabled:
        return v_atom, torch.zeros_like(v_corr)
    scale = torch.as_tensor(
        correction_scale, device=v_atom.device, dtype=v_atom.dtype
    )
    warmup = torch.as_tensor(
        warmup_scale, device=v_atom.device, dtype=v_atom.dtype
    )
    scaled_correction = scale * warmup.clamp(0.0, 1.0) * v_corr
    return v_atom + scaled_correction, scaled_correction


@torch.no_grad()
def solve_q_targets(
    pos: Tensor,
    residual: Tensor,
    anchor_index: Tensor,
    moving_index: Tensor,
    affected_atom_index: Tensor,
    affected_bond_index: Tensor,
    *,
    ridge_eps: float = 1.0e-4,
    max_q_norm: float = 10.0,
    max_condition: float = 1.0e6,
    eps: float = 1.0e-8,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Solve online ridge-LS pseudo-labels from ``(u_target-v_atom).detach()``."""

    if pos.shape != residual.shape or pos.ndim != 2 or pos.size(1) != 3:
        raise ValueError(
            "pos and residual must both have shape [N, 3], got "
            f"{tuple(pos.shape)} and {tuple(residual.shape)}."
        )
    num_bonds = anchor_index.numel()
    targets = pos.new_zeros((num_bonds, 4))
    valid = torch.zeros(num_bonds, dtype=torch.bool, device=pos.device)
    conditions = pos.new_full((num_bonds,), float("inf"))
    if num_bonds == 0:
        return targets, valid, conditions

    solve_dtype = torch.float64 if pos.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=pos.device.type, enabled=False):
        pos_solve = pos.to(dtype=solve_dtype)
        residual_solve = residual.to(dtype=solve_dtype)
        frame, geometry_valid = build_bond_frames(
            pos_solve,
            anchor_index,
            moving_index,
            affected_atom_index,
            affected_bond_index,
            eps=eps,
        )
        e0 = frame[:, :, 0]

        atoms = affected_atom_index
        atom_bonds = affected_bond_index
        lever = pos_solve[atoms] - pos_solve[anchor_index[atom_bonds]]
        atom_jacobian = build_atom_jacobian(
            e0[atom_bonds], frame[atom_bonds], lever
        )
        atom_residual = residual_solve[atoms]
        normal_contribution = torch.matmul(
            atom_jacobian.transpose(1, 2), atom_jacobian
        )
        rhs_contribution = torch.matmul(
            atom_jacobian.transpose(1, 2), atom_residual.unsqueeze(-1)
        ).squeeze(-1)

        normal = torch.zeros(
            (num_bonds, 4, 4), device=pos.device, dtype=solve_dtype
        )
        rhs = torch.zeros((num_bonds, 4), device=pos.device, dtype=solve_dtype)
        normal.index_add_(0, atom_bonds, normal_contribution)
        rhs.index_add_(0, atom_bonds, rhs_contribution)
        identity = torch.eye(4, device=pos.device, dtype=solve_dtype)
        normal = normal + float(ridge_eps) * identity.unsqueeze(0)

        affected_count = torch.bincount(atom_bonds, minlength=num_bonds)
        finite_system = torch.isfinite(normal).flatten(1).all(
            dim=1
        ) & torch.isfinite(rhs).all(dim=-1)
        safe_normal = torch.where(
            finite_system[:, None, None], normal, identity.unsqueeze(0)
        )
        condition = torch.linalg.cond(safe_normal)
        conditions = torch.where(
            finite_system,
            condition.to(dtype=pos.dtype),
            conditions,
        )
        pre_solve_valid = (
            geometry_valid
            & (affected_count > 0)
            & finite_system
            & torch.isfinite(condition)
            & (condition <= max_condition)
        )
        safe_rhs = torch.where(pre_solve_valid[:, None], rhs, torch.zeros_like(rhs))
        solve_normal = torch.where(
            pre_solve_valid[:, None, None], safe_normal, identity.unsqueeze(0)
        )
        q_target = torch.linalg.solve(solve_normal, safe_rhs.unsqueeze(-1)).squeeze(-1)
        q_norm = torch.linalg.norm(q_target, dim=-1)
        valid = (
            pre_solve_valid
            & torch.isfinite(q_target).all(dim=-1)
            & torch.isfinite(q_norm)
            & (q_norm <= max_q_norm)
        )
        targets = torch.where(
            valid[:, None], q_target.to(dtype=pos.dtype), targets
        )

    return targets, valid, conditions
