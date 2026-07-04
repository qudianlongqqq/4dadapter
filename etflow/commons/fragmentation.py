from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from rdkit import Chem
from rdkit.Chem import Lipinski, rdMolDescriptors
from rdkit.Chem.rdchem import Mol


FRAGMENT_TYPE_ORDER = ("aromatic_ring", "ring", "rotatable_region", "other")


@dataclass
class FragmentationResult:
    atom_to_fragment_id: torch.Tensor
    fragment_types: Dict[int, str]
    fragment_type_per_atom: List[str]
    ring_atom_mask: torch.Tensor
    aromatic_atom_mask: torch.Tensor
    rotatable_bond_atom_mask: torch.Tensor
    num_rotatable_bonds: int


def _mol_from_smiles(smiles: str) -> Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    return mol


def _neighbors(mol: Mol, atom_indices: Iterable[int]) -> set[int]:
    out = set(atom_indices)
    for atom_idx in atom_indices:
        atom = mol.GetAtomWithIdx(int(atom_idx))
        out.update(nbr.GetIdx() for nbr in atom.GetNeighbors())
    return out


def _rotatable_bond_atoms(mol: Mol, include_neighbors: bool = True) -> set[int]:
    atoms: set[int] = set()
    for match in mol.GetSubstructMatches(Lipinski.RotatableBondSmarts):
        atoms.update(int(idx) for idx in match[:2])

    if include_neighbors and atoms:
        atoms = _neighbors(mol, atoms)

    return atoms


def _connected_components(mol: Mol, atom_indices: set[int]) -> List[List[int]]:
    components = []
    visited: set[int] = set()

    for start_idx in sorted(atom_indices):
        if start_idx in visited:
            continue

        stack = [start_idx]
        visited.add(start_idx)
        component = []

        while stack:
            atom_idx = stack.pop()
            component.append(atom_idx)
            atom = mol.GetAtomWithIdx(int(atom_idx))

            for nbr in atom.GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx in atom_indices and nbr_idx not in visited:
                    visited.add(nbr_idx)
                    stack.append(nbr_idx)

        components.append(sorted(component))

    return components


def fragment_molecule(
    smiles: Optional[str] = None,
    mol: Optional[Mol] = None,
    expected_num_atoms: Optional[int] = None,
    include_rotatable_neighbors: bool = True,
) -> FragmentationResult:
    """Assign atoms to simple chemistry-driven diagnostic fragments.

    The first-pass priority is aromatic ring > non-aromatic ring >
    rotatable-bond region > other. Each type is split into graph-connected
    components so the rigid fit is local enough for diagnostics.
    """

    if mol is None:
        if smiles is None:
            raise ValueError("Either smiles or mol must be provided.")
        mol = _mol_from_smiles(smiles)

    num_atoms = mol.GetNumAtoms()
    if expected_num_atoms is not None and num_atoms != expected_num_atoms:
        raise ValueError(
            "Fragmentation atom count mismatch: "
            f"mol has {num_atoms} atoms, but positions have {expected_num_atoms} atoms. "
            "Pass the dataset RDKit mol when possible to preserve atom ordering."
        )

    ring_atoms = {atom.GetIdx() for atom in mol.GetAtoms() if atom.IsInRing()}
    aromatic_atoms = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.IsInRing() and atom.GetIsAromatic()
    }
    rotatable_atoms = _rotatable_bond_atoms(
        mol, include_neighbors=include_rotatable_neighbors
    )

    atom_type = []
    for atom_idx in range(num_atoms):
        if atom_idx in aromatic_atoms:
            atom_type.append("aromatic_ring")
        elif atom_idx in ring_atoms:
            atom_type.append("ring")
        elif atom_idx in rotatable_atoms:
            atom_type.append("rotatable_region")
        else:
            atom_type.append("other")

    atom_to_fragment_id = torch.full((num_atoms,), -1, dtype=torch.long)
    fragment_types: Dict[int, str] = {}
    fragment_id = 0

    for fragment_type in FRAGMENT_TYPE_ORDER:
        type_atoms = {
            atom_idx for atom_idx, current_type in enumerate(atom_type) if current_type == fragment_type
        }
        for component in _connected_components(mol, type_atoms):
            for atom_idx in component:
                atom_to_fragment_id[atom_idx] = fragment_id
            fragment_types[fragment_id] = fragment_type
            fragment_id += 1

    if (atom_to_fragment_id < 0).any():
        missing = torch.where(atom_to_fragment_id < 0)[0].tolist()
        raise RuntimeError(f"Fragment assignment failed for atoms: {missing}")

    ring_atom_mask = torch.tensor(
        [atom_idx in ring_atoms for atom_idx in range(num_atoms)], dtype=torch.bool
    )
    aromatic_atom_mask = torch.tensor(
        [atom_idx in aromatic_atoms for atom_idx in range(num_atoms)], dtype=torch.bool
    )
    rotatable_bond_atom_mask = torch.tensor(
        [atom_idx in rotatable_atoms for atom_idx in range(num_atoms)], dtype=torch.bool
    )

    return FragmentationResult(
        atom_to_fragment_id=atom_to_fragment_id,
        fragment_types=fragment_types,
        fragment_type_per_atom=atom_type,
        ring_atom_mask=ring_atom_mask,
        aromatic_atom_mask=aromatic_atom_mask,
        rotatable_bond_atom_mask=rotatable_bond_atom_mask,
        num_rotatable_bonds=int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
    )
