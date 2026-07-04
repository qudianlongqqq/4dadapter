from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional

import torch
from rdkit import Chem
from rdkit.Chem import Lipinski, rdMolDescriptors
from rdkit.Chem.rdchem import BondType, Mol


def _mol_from_smiles(smiles: str) -> Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    return mol


def _neighbors_without_bond(mol: Mol, atom_idx: int, cut_a: int, cut_b: int) -> Iterable[int]:
    atom = mol.GetAtomWithIdx(int(atom_idx))
    for nbr in atom.GetNeighbors():
        nbr_idx = int(nbr.GetIdx())
        if {int(atom_idx), nbr_idx} == {int(cut_a), int(cut_b)}:
            continue
        yield nbr_idx


def _component_after_cut(mol: Mol, start_idx: int, cut_a: int, cut_b: int) -> List[int]:
    visited = {int(start_idx)}
    queue: deque[int] = deque([int(start_idx)])

    while queue:
        atom_idx = queue.popleft()
        for nbr_idx in _neighbors_without_bond(mol, atom_idx, cut_a, cut_b):
            if nbr_idx in visited:
                continue
            visited.add(nbr_idx)
            queue.append(nbr_idx)

    return sorted(visited)


def rotatable_bond_sides(
    smiles: Optional[str] = None,
    mol: Optional[Mol] = None,
    expected_num_atoms: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Return atom sets on each side of every acyclic rotatable bond."""

    if mol is None:
        if smiles is None:
            raise ValueError("Either smiles or mol must be provided.")
        mol = _mol_from_smiles(smiles)

    num_atoms = mol.GetNumAtoms()
    if expected_num_atoms is not None and num_atoms != expected_num_atoms:
        raise ValueError(
            "Rotatable bond atom count mismatch: "
            f"mol has {num_atoms} atoms, but positions have {expected_num_atoms} atoms. "
            "Pass the dataset RDKit mol when possible to preserve atom ordering."
        )

    rows: List[Dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for match in mol.GetSubstructMatches(Lipinski.RotatableBondSmarts):
        atom_a = int(match[0])
        atom_b = int(match[1])
        key = tuple(sorted((atom_a, atom_b)))
        if key in seen:
            continue
        seen.add(key)

        bond = mol.GetBondBetweenAtoms(atom_a, atom_b)
        if bond is None:
            continue
        if bond.IsInRing() or bond.GetBondType() != BondType.SINGLE:
            continue

        side_a = _component_after_cut(mol, atom_a, atom_a, atom_b)
        side_b = _component_after_cut(mol, atom_b, atom_a, atom_b)
        if set(side_a).intersection(side_b):
            continue
        if len(side_a) + len(side_b) != num_atoms:
            continue

        rows.append(
            {
                "bond_index": int(bond.GetIdx()),
                "bond_atom_a": atom_a,
                "bond_atom_b": atom_b,
                "side_a_atoms": side_a,
                "side_b_atoms": side_b,
            }
        )

    rows.sort(key=lambda row: int(row["bond_index"]))
    return rows


def _cross_matrix_for_omega_cross_r(r: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(r[:, 0])
    rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
    return torch.stack(
        [
            torch.stack([zeros, rz, -ry], dim=1),
            torch.stack([-rz, zeros, rx], dim=1),
            torch.stack([ry, -rx, zeros], dim=1),
        ],
        dim=1,
    ).reshape(-1, 3)


def fit_rigid_motion(
    x: torch.Tensor,
    v: torch.Tensor,
    atom_indices: List[int],
    min_atoms: int = 3,
    eps: float = 1.0e-8,
) -> Dict[str, object]:
    """Fit translation + angular velocity for one atom set."""

    if len(atom_indices) < min_atoms:
        zero = torch.zeros(3, dtype=x.dtype, device=x.device)
        return {
            "status": "too_small",
            "rank": 0,
            "center": zero,
            "translation": zero,
            "omega": zero,
            "residual_ratio": float("nan"),
            "rigid_explain_ratio": float("nan"),
        }

    index = torch.tensor(atom_indices, dtype=torch.long, device=x.device)
    original_dtype = x.dtype
    x_side = x.index_select(0, index).detach().to(dtype=torch.float64)
    v_side = v.index_select(0, index).detach().to(dtype=torch.float64)

    center = x_side.mean(dim=0)
    translation = v_side.mean(dim=0)
    r = x_side - center
    v_rel = v_side - translation
    a = _cross_matrix_for_omega_cross_r(r)
    b = v_rel.reshape(-1)
    rank = int(torch.linalg.matrix_rank(a).item())

    try:
        omega = torch.linalg.lstsq(a, b).solution
    except RuntimeError:
        omega = torch.linalg.pinv(a) @ b

    rotation_velocity = torch.cross(omega.expand_as(r), r, dim=1)
    rigid_velocity = translation.expand_as(v_side) + rotation_velocity
    residual_velocity = v_side - rigid_velocity

    velocity_norm = torch.linalg.norm(v_side)
    residual_norm = torch.linalg.norm(residual_velocity)
    residual_ratio = residual_norm / (velocity_norm + eps)
    rigid_explain_ratio = 1.0 - residual_norm.pow(2) / (velocity_norm.pow(2) + eps)

    status = "ok" if rank >= 3 else "rank_deficient"
    return {
        "status": status,
        "rank": rank,
        "center": center.to(dtype=original_dtype),
        "translation": translation.to(dtype=original_dtype),
        "omega": omega.to(dtype=original_dtype),
        "residual_ratio": float(residual_ratio.item()),
        "rigid_explain_ratio": float(rigid_explain_ratio.item()),
    }


def _nan_row_values() -> Dict[str, float]:
    return {
        "delta_omega_x": float("nan"),
        "delta_omega_y": float("nan"),
        "delta_omega_z": float("nan"),
        "relative_rotation_norm": float("nan"),
        "bond_axis_x": float("nan"),
        "bond_axis_y": float("nan"),
        "bond_axis_z": float("nan"),
        "torsion_velocity": float("nan"),
        "abs_torsion_velocity": float("nan"),
    }


def decompose_rotatable_bond_motion(
    x: torch.Tensor,
    v: torch.Tensor,
    bond_sides: List[Dict[str, object]],
    min_side_atoms: int = 3,
    eps: float = 1.0e-8,
) -> List[Dict[str, object]]:
    """Fit both sides of each rotatable bond and compute relative angular motion."""

    if x.ndim != 2 or x.size(-1) != 3:
        raise ValueError(f"x must have shape [num_atoms, 3], got {tuple(x.shape)}")
    if v.shape != x.shape:
        raise ValueError(f"v must have the same shape as x, got {tuple(v.shape)}")

    rows: List[Dict[str, object]] = []
    x_fit = x.detach().to(dtype=torch.float64)

    for bond in bond_sides:
        atom_a = int(bond["bond_atom_a"])
        atom_b = int(bond["bond_atom_b"])
        side_a_atoms = list(bond["side_a_atoms"])
        side_b_atoms = list(bond["side_b_atoms"])
        fit_a = fit_rigid_motion(x, v, side_a_atoms, min_atoms=min_side_atoms, eps=eps)
        fit_b = fit_rigid_motion(x, v, side_b_atoms, min_atoms=min_side_atoms, eps=eps)

        row: Dict[str, object] = {
            "bond_index": int(bond["bond_index"]),
            "bond_atom_a": atom_a,
            "bond_atom_b": atom_b,
            "side_a_size": len(side_a_atoms),
            "side_b_size": len(side_b_atoms),
            "fit_status_a": fit_a["status"],
            "fit_status_b": fit_b["status"],
            "fit_rank_a": int(fit_a["rank"]),
            "fit_rank_b": int(fit_b["rank"]),
        }

        omega_a = fit_a["omega"]
        omega_b = fit_b["omega"]
        row.update(
            {
                "omega_a_x": float(omega_a[0].item()),
                "omega_a_y": float(omega_a[1].item()),
                "omega_a_z": float(omega_a[2].item()),
                "omega_b_x": float(omega_b[0].item()),
                "omega_b_y": float(omega_b[1].item()),
                "omega_b_z": float(omega_b[2].item()),
                "side_a_residual_ratio": float(fit_a["residual_ratio"]),
                "side_b_residual_ratio": float(fit_b["residual_ratio"]),
                "side_a_rigid_explain_ratio": float(fit_a["rigid_explain_ratio"]),
                "side_b_rigid_explain_ratio": float(fit_b["rigid_explain_ratio"]),
            }
        )

        if fit_a["status"] != "ok" or fit_b["status"] != "ok":
            row.update(_nan_row_values())
            rows.append(row)
            continue

        axis = x_fit[atom_b] - x_fit[atom_a]
        axis_norm = torch.linalg.norm(axis)
        if axis_norm <= eps:
            row["fit_status_a"] = "invalid_axis"
            row["fit_status_b"] = "invalid_axis"
            row.update(_nan_row_values())
            rows.append(row)
            continue

        bond_axis = axis / axis_norm
        delta_omega = omega_b.to(dtype=torch.float64) - omega_a.to(dtype=torch.float64)
        relative_rotation_norm = torch.linalg.norm(delta_omega)
        torsion_velocity = torch.dot(delta_omega, bond_axis)
        row.update(
            {
                "delta_omega_x": float(delta_omega[0].item()),
                "delta_omega_y": float(delta_omega[1].item()),
                "delta_omega_z": float(delta_omega[2].item()),
                "relative_rotation_norm": float(relative_rotation_norm.item()),
                "bond_axis_x": float(bond_axis[0].item()),
                "bond_axis_y": float(bond_axis[1].item()),
                "bond_axis_z": float(bond_axis[2].item()),
                "torsion_velocity": float(torsion_velocity.item()),
                "abs_torsion_velocity": float(abs(torsion_velocity.item())),
            }
        )
        rows.append(row)

    return rows


def count_rotatable_bonds(smiles: Optional[str] = None, mol: Optional[Mol] = None) -> int:
    if mol is None:
        if smiles is None:
            raise ValueError("Either smiles or mol must be provided.")
        mol = _mol_from_smiles(smiles)
    return int(rdMolDescriptors.CalcNumRotatableBonds(mol))
