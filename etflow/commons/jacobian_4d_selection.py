"""Tensor-only selection for the experimental 4D Jacobian branch.

The dataset already stores rotatable bonds oriented from the fixed endpoint to
the endpoint on the deterministically selected smaller component.  This module
only filters and caps those cached tensors; it never performs RDKit work in the
training step.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor


def _validate_index_tensor(name: str, value: Tensor) -> None:
    if value.ndim != 2 or value.size(0) != 2:
        raise ValueError(f"{name} must have shape [2, K], got {tuple(value.shape)}.")
    if value.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long, got {value.dtype}.")


def select_jacobian_4d_bonds(
    rotatable_bond_index: Tensor,
    atom_bond_influence_index: Tensor,
    batch: Optional[Tensor],
    *,
    min_affected_atoms: int = 2,
    max_bonds_per_mol: int = 16,
) -> Dict[str, Tensor]:
    """Filter cached oriented rotatable bonds for the 4D correction.

    Returns selected anchor/moving atom indices and a remapped sparse influence
    tensor whose bond row is local to the selected bond array.
    """

    _validate_index_tensor("rotatable_bond_index", rotatable_bond_index)
    _validate_index_tensor(
        "atom_bond_influence_index", atom_bond_influence_index
    )
    if min_affected_atoms < 1:
        raise ValueError(
            "min_affected_atoms must be positive, got "
            f"{min_affected_atoms}."
        )
    if max_bonds_per_mol < 1:
        raise ValueError(
            "max_bonds_per_mol must be positive, got "
            f"{max_bonds_per_mol}."
        )

    device = rotatable_bond_index.device
    num_bonds = rotatable_bond_index.size(1)
    if batch is None:
        maximum_indices = []
        if rotatable_bond_index.numel():
            maximum_indices.append(rotatable_bond_index.max())
        if atom_bond_influence_index.size(1):
            maximum_indices.append(atom_bond_influence_index[0].max())
        num_atoms = (
            int(torch.stack(maximum_indices).max().item()) + 1
            if maximum_indices
            else 0
        )
        atom_batch = torch.zeros(num_atoms, dtype=torch.long, device=device)
    else:
        atom_batch = batch.to(device=device, dtype=torch.long).reshape(-1)
        num_atoms = atom_batch.numel()

    if num_bonds:
        if rotatable_bond_index.min() < 0 or rotatable_bond_index.max() >= num_atoms:
            raise IndexError("rotatable_bond_index contains an invalid atom index.")
    if atom_bond_influence_index.size(1):
        influence_atoms = atom_bond_influence_index[0]
        influence_bonds = atom_bond_influence_index[1]
        if influence_atoms.min() < 0 or influence_atoms.max() >= num_atoms:
            raise IndexError(
                "atom_bond_influence_index contains an invalid atom index."
            )
        if influence_bonds.min() < 0 or influence_bonds.max() >= num_bonds:
            raise IndexError(
                "atom_bond_influence_index contains an invalid bond index."
            )
    elif num_bonds:
        influence_atoms = atom_bond_influence_index.new_empty((0,))
        influence_bonds = atom_bond_influence_index.new_empty((0,))
    else:
        influence_atoms = atom_bond_influence_index.new_empty((0,))
        influence_bonds = atom_bond_influence_index.new_empty((0,))

    affected_counts = torch.bincount(influence_bonds, minlength=num_bonds)
    eligible = torch.nonzero(
        affected_counts >= min_affected_atoms, as_tuple=False
    ).reshape(-1)

    if eligible.numel():
        bond_batch_all = atom_batch[rotatable_bond_index[0]]
        kept = []
        for graph_index in torch.unique(bond_batch_all[eligible], sorted=True):
            graph_bonds = eligible[bond_batch_all[eligible] == graph_index]
            kept.append(graph_bonds[:max_bonds_per_mol])
        selected_original = torch.cat(kept) if kept else eligible[:0]
    else:
        bond_batch_all = torch.empty(0, dtype=torch.long, device=device)
        selected_original = eligible

    original_to_selected = torch.full(
        (num_bonds,), -1, dtype=torch.long, device=device
    )
    original_to_selected[selected_original] = torch.arange(
        selected_original.numel(), dtype=torch.long, device=device
    )
    if influence_bonds.numel():
        remapped_bonds = original_to_selected[influence_bonds]
        keep_influence = remapped_bonds >= 0
        selected_influence = torch.stack(
            [
                influence_atoms[keep_influence],
                remapped_bonds[keep_influence],
            ],
            dim=0,
        )
    else:
        selected_influence = torch.empty(
            (2, 0), dtype=torch.long, device=device
        )

    selected_bond_index = rotatable_bond_index[:, selected_original]
    selected_batch = (
        atom_batch[selected_bond_index[0]]
        if selected_original.numel()
        else torch.empty(0, dtype=torch.long, device=device)
    )
    return {
        "anchor_index": selected_bond_index[0],
        "moving_index": selected_bond_index[1],
        "affected_atom_index": selected_influence[0],
        "affected_bond_index": selected_influence[1],
        "affected_count": affected_counts[selected_original],
        "bond_batch": selected_batch,
        "original_bond_index": selected_original,
    }
