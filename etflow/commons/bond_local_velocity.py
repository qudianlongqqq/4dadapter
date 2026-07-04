"""Bond-local projections for atom-wise Cartesian velocity supervision."""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


def _validate_pos(pos: Tensor) -> None:
    if pos.ndim != 2 or pos.size(-1) != 3:
        raise ValueError(f"pos must have shape [N, 3], got {tuple(pos.shape)}.")


def _validate_bond_index(bond_index: Tensor, num_atoms: int) -> None:
    if bond_index.ndim != 2 or bond_index.size(0) != 2:
        raise ValueError(
            f"bond_index must have shape [2, B], got {tuple(bond_index.shape)}."
        )
    if bond_index.numel() == 0:
        return
    if bond_index.min() < 0 or bond_index.max() >= num_atoms:
        raise IndexError("bond_index contains an invalid atom index.")


def build_bond_frame(
    pos: Tensor,
    bond_index: Tensor,
    eps: float = 1.0e-8,
) -> Tuple[Tensor, Tensor]:
    """Build right-handed local frames for bonds.

    Returns frames with shape ``[B, 3, 3]`` whose columns are
    ``[parallel, perp1, perp2]``, plus a boolean mask for non-degenerate bonds.
    """

    _validate_pos(pos)
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}.")

    bond_index = bond_index.to(device=pos.device, dtype=torch.long)
    _validate_bond_index(bond_index, pos.size(0))
    num_bonds = bond_index.size(1)
    if num_bonds == 0:
        return pos.new_empty((0, 3, 3)), torch.zeros(
            0, dtype=torch.bool, device=pos.device
        )

    atom_i, atom_j = bond_index
    bond_vector = pos[atom_j] - pos[atom_i]
    bond_length = torch.linalg.norm(bond_vector, dim=-1)
    valid_mask = bond_length > eps
    e_parallel = bond_vector / bond_length.unsqueeze(-1).clamp_min(eps)

    z_axis = pos.new_tensor([0.0, 0.0, 1.0]).expand(num_bonds, -1)
    y_axis = pos.new_tensor([0.0, 1.0, 0.0]).expand(num_bonds, -1)
    use_z_axis = e_parallel[:, 2].abs() < 0.9
    reference_axis = torch.where(use_z_axis.unsqueeze(-1), z_axis, y_axis)

    e_perp1 = torch.cross(e_parallel, reference_axis, dim=-1)
    e_perp1 = e_perp1 / torch.linalg.norm(
        e_perp1, dim=-1, keepdim=True
    ).clamp_min(eps)
    e_perp2 = torch.cross(e_parallel, e_perp1, dim=-1)
    frame = torch.stack((e_parallel, e_perp1, e_perp2), dim=-1)
    return frame, valid_mask


def compute_bond_local_velocity(
    pos: Tensor,
    velocity: Tensor,
    bond_index: Tensor,
    eps: float = 1.0e-8,
) -> Tuple[Tensor, Tensor]:
    """Project relative atom velocities into each bond's local frame."""

    _validate_pos(pos)
    if velocity.shape != pos.shape:
        raise ValueError(
            "velocity must have the same shape as pos, got "
            f"{tuple(velocity.shape)} and {tuple(pos.shape)}."
        )

    velocity = velocity.to(device=pos.device, dtype=pos.dtype)
    bond_index = bond_index.to(device=pos.device, dtype=torch.long)
    frame, valid_mask = build_bond_frame(pos, bond_index, eps=eps)
    if bond_index.size(1) == 0:
        return pos.new_empty((0, 3)), valid_mask

    atom_i, atom_j = bond_index
    relative_velocity = velocity[atom_j] - velocity[atom_i]
    local_velocity = torch.bmm(
        frame.transpose(1, 2), relative_velocity.unsqueeze(-1)
    ).squeeze(-1)
    return local_velocity, valid_mask


def bond_local_velocity_loss(
    pos: Tensor,
    pred_velocity: Tensor,
    target_velocity: Tensor,
    bond_index: Tensor,
    bond_mask: Optional[Tensor] = None,
    eps: float = 1.0e-8,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Return MSE between predicted and target bond-local relative velocities."""

    q_pred, valid_mask = compute_bond_local_velocity(
        pos, pred_velocity, bond_index, eps=eps
    )
    q_target, target_valid_mask = compute_bond_local_velocity(
        pos, target_velocity, bond_index, eps=eps
    )
    valid_mask = valid_mask & target_valid_mask

    if bond_mask is not None:
        if bond_mask.ndim != 1 or bond_mask.numel() != valid_mask.numel():
            raise ValueError(
                "bond_mask must have shape [B], got "
                f"{tuple(bond_mask.shape)} for B={valid_mask.numel()}."
            )
        valid_mask = valid_mask & bond_mask.to(device=pos.device, dtype=torch.bool)

    if valid_mask.any():
        q_pred_valid = q_pred[valid_mask]
        q_target_valid = q_target[valid_mask]
        squared_error = (q_pred_valid - q_target_valid).square()
        loss = squared_error.mean()
        parallel_loss = squared_error[:, 0].mean()
        perp_loss = squared_error[:, 1:].mean()
        q_pred_abs_mean = q_pred_valid.detach().abs().mean()
        q_target_abs_mean = q_target_valid.detach().abs().mean()
    else:
        loss = pred_velocity.sum() * 0.0
        parallel_loss = loss.detach()
        perp_loss = loss.detach()
        q_pred_abs_mean = loss.detach()
        q_target_abs_mean = loss.detach()

    stats = {
        "loss": loss.detach(),
        "q_pred_abs_mean": q_pred_abs_mean,
        "q_target_abs_mean": q_target_abs_mean,
        "valid_bonds": valid_mask.sum().to(dtype=pos.dtype),
        "parallel_loss": parallel_loss.detach(),
        "perp_loss": perp_loss.detach(),
    }
    return loss, stats


__all__ = [
    "bond_local_velocity_loss",
    "build_bond_frame",
    "compute_bond_local_velocity",
]
