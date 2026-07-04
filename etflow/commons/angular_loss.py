"""Utilities for supervising the scalar velocity of rotatable bonds."""

from typing import Optional, Tuple

import torch
from torch import Tensor


def _validate_index(name: str, index: Tensor) -> None:
    if index.ndim != 2 or index.size(0) != 2:
        raise ValueError(f"{name} must have shape [2, M], got {tuple(index.shape)}.")


def compute_target_dot_tau(
    pos: Tensor,
    target_velocity: Tensor,
    rotatable_bond_index: Tensor,
    atom_bond_influence_index: Tensor,
    batch: Optional[Tensor] = None,
    eps: float = 1.0e-8,
) -> Tuple[Tensor, Tensor]:
    """Project atom velocities onto each bond's scalar rotational basis.

    A bond is valid when it has at least one influenced atom and the summed
    squared norm of its rotational bases is larger than eps. Invalid bond
    targets are zero and must be excluded from the auxiliary loss using the
    returned mask.
    """

    if pos.ndim != 2 or pos.size(-1) != 3:
        raise ValueError(f"pos must have shape [N, 3], got {tuple(pos.shape)}.")
    if target_velocity.shape != pos.shape:
        raise ValueError(
            "target_velocity must have the same shape as pos, got "
            f"{tuple(target_velocity.shape)} and {tuple(pos.shape)}."
        )
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}.")

    _validate_index("rotatable_bond_index", rotatable_bond_index)
    _validate_index("atom_bond_influence_index", atom_bond_influence_index)

    device = pos.device
    dtype = pos.dtype
    target_velocity = target_velocity.to(device=device, dtype=dtype)
    rotatable_bond_index = rotatable_bond_index.to(device=device, dtype=torch.long)
    atom_bond_influence_index = atom_bond_influence_index.to(
        device=device, dtype=torch.long
    )

    num_atoms = pos.size(0)
    num_bonds = rotatable_bond_index.size(1)
    dot_tau_target = pos.new_zeros((num_bonds,))
    valid_bond_mask = torch.zeros(num_bonds, dtype=torch.bool, device=device)

    if batch is not None:
        if batch.ndim != 1 or batch.numel() != num_atoms:
            raise ValueError(
                f"batch must have shape [N], got {tuple(batch.shape)} for N={num_atoms}."
            )
        batch = batch.to(device=device, dtype=torch.long)

    if num_bonds == 0:
        if atom_bond_influence_index.size(1) != 0:
            raise ValueError(
                "atom_bond_influence_index must be empty when there are no "
                "rotatable bonds."
            )
        return dot_tau_target, valid_bond_mask

    if rotatable_bond_index.min() < 0 or rotatable_bond_index.max() >= num_atoms:
        raise IndexError("rotatable_bond_index contains an invalid atom index.")

    fixed_atom = rotatable_bond_index[0]
    rotating_atom = rotatable_bond_index[1]
    if batch is not None and not torch.equal(batch[fixed_atom], batch[rotating_atom]):
        raise ValueError("A rotatable bond cannot connect atoms from different graphs.")

    num_influences = atom_bond_influence_index.size(1)
    if num_influences == 0:
        return dot_tau_target, valid_bond_mask

    atom_index = atom_bond_influence_index[0]
    bond_index = atom_bond_influence_index[1]
    if atom_index.min() < 0 or atom_index.max() >= num_atoms:
        raise IndexError("atom_bond_influence_index contains an invalid atom index.")
    if bond_index.min() < 0 or bond_index.max() >= num_bonds:
        raise IndexError("atom_bond_influence_index contains an invalid bond index.")
    if batch is not None and not torch.equal(
        batch[atom_index],
        batch[fixed_atom[bond_index]],
    ):
        raise ValueError("An influenced atom and its rotatable bond must share a graph.")

    bond_vector = pos[rotating_atom] - pos[fixed_atom]
    bond_axis = bond_vector / torch.linalg.norm(
        bond_vector,
        dim=-1,
        keepdim=True,
    ).clamp_min(eps)
    bond_center = 0.5 * (pos[fixed_atom] + pos[rotating_atom])

    lever = pos[atom_index] - bond_center[bond_index]
    basis = torch.cross(bond_axis[bond_index], lever, dim=-1)
    numerator_per_atom = (target_velocity[atom_index] * basis).sum(dim=-1)
    denominator_per_atom = basis.square().sum(dim=-1)

    numerator = pos.new_zeros((num_bonds,))
    denominator = pos.new_zeros((num_bonds,))
    numerator.index_add_(0, bond_index, numerator_per_atom)
    denominator.index_add_(0, bond_index, denominator_per_atom)

    influence_count = torch.zeros(num_bonds, dtype=torch.long, device=device)
    influence_count.index_add_(
        0,
        bond_index,
        torch.ones_like(bond_index, dtype=torch.long),
    )
    valid_bond_mask = (influence_count > 0) & (denominator > eps)
    dot_tau_target[valid_bond_mask] = numerator[valid_bond_mask] / (
        denominator[valid_bond_mask] + eps
    )
    return dot_tau_target, valid_bond_mask


__all__ = ["compute_target_dot_tau"]
