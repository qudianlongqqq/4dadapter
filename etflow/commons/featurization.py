# allowable multiple choice node and edge features
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Optional, Tuple

import datamol as dm
import torch
from datamol.types import Mol
from rdkit import Chem
from torch_geometric.data import Data

from .covmat import build_conformer
from .rotatable_motion import rotatable_bond_sides
from .utils import atom_to_feature_vector, compute_edge_index, get_chiral_tensors


MOL_OBJECT_FIELDS = ("mol", "rdmol", "rdkit_mol", "rd_mol")
MOL_BLOCK_FIELDS = ("mol_block", "molblock", "molBlock", "molfile_block")
SMILES_FIELDS = ("smiles", "canonical_smiles", "smi")


class MolRecoveryResult(NamedTuple):
    mol: Mol
    source: str


class MoleculeData(Data):
    """PyG data with separate batching domains for atoms and rotatable bonds."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "atom_bond_influence_index":
            num_rotatable_bonds = int(self.rotatable_bond_index.size(1))
            return torch.tensor(
                [[self.num_nodes], [num_rotatable_bonds]],
                dtype=value.dtype,
                device=value.device,
            )
        return super().__inc__(key, value, *args, **kwargs)


def _require_mol(mol: Optional[Mol], context: str) -> Mol:
    if mol is None:
        raise ValueError(f"RDKit Mol is None while {context}.")
    if not hasattr(mol, "GetAtoms") or not hasattr(mol, "GetNumAtoms"):
        raise TypeError(
            f"Expected an RDKit Mol while {context}, got {type(mol).__name__}."
        )
    if mol.GetNumAtoms() == 0:
        raise ValueError(f"RDKit Mol has zero atoms while {context}.")
    return mol


def get_mol_from_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"SMILES must be a non-empty string, got {smiles!r}.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    return _require_mol(mol, f"adding hydrogens to SMILES {smiles!r}")


def get_sample_field(sample: Any, key: str) -> Any:
    if isinstance(sample, Mapping):
        return sample.get(key, None)
    getter = getattr(sample, "get", None)
    if callable(getter):
        return getter(key, None)
    return getattr(sample, key, None)


def get_sample_field_names(sample: Any) -> list[str]:
    if isinstance(sample, Mapping):
        return sorted(str(key) for key in sample.keys())
    keys = getattr(sample, "keys", None)
    if callable(keys):
        return sorted(str(key) for key in keys())
    try:
        return sorted(str(key) for key in vars(sample))
    except TypeError:
        return []


def _iter_values(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield f"[{index}]", item
    else:
        yield "", value


def _parse_smiles_candidates(smiles: str) -> list[tuple[str, Mol]]:
    candidates = []
    try:
        ordered_mol = dm.to_mol(smiles, remove_hs=False, ordered=True)
    except Exception:
        ordered_mol = None
    if ordered_mol is not None:
        candidates.append(("ordered_smiles", ordered_mol))

    try:
        rdkit_mol = Chem.MolFromSmiles(smiles)
    except Exception:
        rdkit_mol = None
    if rdkit_mol is not None:
        candidates.append(("rdkit_smiles", rdkit_mol))
    return candidates


def _candidate_mols(value: Any, field_name: str) -> list[tuple[str, Mol]]:
    candidates = []
    for suffix, item in _iter_values(value):
        source = f"{field_name}{suffix}"
        if item is None:
            continue
        if hasattr(item, "GetAtoms") and hasattr(item, "GetNumAtoms"):
            try:
                candidates.append((source, Chem.Mol(item)))
            except Exception:
                pass
            continue
        if isinstance(item, bytes):
            item = item.decode("utf-8", errors="replace")
        if not isinstance(item, str) or not item.strip():
            continue

        if field_name in MOL_BLOCK_FIELDS or "M  END" in item:
            try:
                mol = Chem.MolFromMolBlock(
                    item,
                    sanitize=True,
                    removeHs=False,
                    strictParsing=True,
                )
            except Exception:
                mol = None
            if mol is not None:
                candidates.append((source, mol))
        else:
            candidates.extend(
                (f"{source}:{parser}", mol)
                for parser, mol in _parse_smiles_candidates(item)
            )
    return candidates


def _validate_recovered_mol(
    mol: Mol,
    expected_atomic_numbers: Optional[torch.Tensor],
    source: str,
) -> Mol:
    mol = _require_mol(Chem.Mol(mol), f"validating recovered molecule from {source}")
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise ValueError(f"RDKit sanitization failed for {source}: {exc}") from exc

    if expected_atomic_numbers is not None:
        expected = torch.as_tensor(expected_atomic_numbers).view(-1).tolist()
        actual = [int(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
        if actual != [int(value) for value in expected]:
            raise ValueError(
                f"atomic-number mismatch for {source}: expected {expected}, got {actual}"
            )
    return mol


def recover_mol_from_sample(
    sample: Any,
    expected_atomic_numbers: Optional[torch.Tensor] = None,
) -> MolRecoveryResult:
    """Recover and validate a molecule from legacy or current processed samples."""

    attempts = []
    fields = MOL_OBJECT_FIELDS + MOL_BLOCK_FIELDS + SMILES_FIELDS
    for field_name in fields:
        value = get_sample_field(sample, field_name)
        if value is None:
            attempts.append(f"{field_name}=None/missing")
            continue

        candidates = _candidate_mols(value, field_name)
        if not candidates:
            attempts.append(f"{field_name}=present but unsupported/unparseable")
            continue

        for source, candidate in candidates:
            variants = [(source, candidate)]
            try:
                variants.append((f"{source}+AddHs", Chem.AddHs(Chem.Mol(candidate))))
            except Exception:
                pass

            for variant_source, variant in variants:
                try:
                    validated = _validate_recovered_mol(
                        variant,
                        expected_atomic_numbers=expected_atomic_numbers,
                        source=variant_source,
                    )
                    return MolRecoveryResult(validated, variant_source)
                except Exception as exc:
                    attempts.append(f"{variant_source}: {exc}")

    available = get_sample_field_names(sample)
    attempt_text = "; ".join(attempts)
    raise ValueError(
        "Could not recover a valid RDKit Mol from processed sample. "
        f"Available fields: {available}. Attempts: {attempt_text}"
    )


def mol_to_ordered_smiles(mol: Mol) -> str:
    mol = _require_mol(mol, "serializing molecule to SMILES")
    smiles = dm.to_smiles(
        mol,
        canonical=False,
        explicit_hs=True,
        with_atom_indices=True,
        isomeric=True,
    )
    if not isinstance(smiles, str) or not smiles:
        raise ValueError("Could not serialize recovered RDKit Mol to a valid SMILES.")
    return smiles


def cache_decorator(func: Callable):
    """Decorator to handle caching logic."""

    def wrapper(self, smiles: str, *args, **kwargs):
        cache_key = func.__name__
        if smiles in self.cache and cache_key in self.cache[smiles]:
            return self.cache[smiles][cache_key]
        result = func(self, smiles, *args, **kwargs)
        self.cache[smiles][cache_key] = result
        return result

    return wrapper


class MoleculeFeaturizer:
    """A Featurizer Class for Molecules.
    - Give smiles, get mol objects, atom features, bond features, etc.
    - Smiles-based Caching to avoid recomputation.
    """

    def __init__(self):
        # smiles based cache
        self.cache = defaultdict(dict)

    @cache_decorator
    def get_mol(self, smiles: str) -> Mol:
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError(f"SMILES must be a non-empty string, got {smiles!r}.")
        errors = []
        for source, mol in _parse_smiles_candidates(smiles):
            try:
                return _validate_recovered_mol(mol, None, source)
            except Exception as exc:
                errors.append(str(exc))
        raise ValueError(
            f"Could not parse SMILES into a valid RDKit Mol: {smiles!r}. "
            f"Errors: {errors}"
        )

    def cache_recovered_mol(self, smiles: str, mol: Mol) -> None:
        mol = Chem.Mol(_require_mol(mol, "caching a recovered molecule"))
        mol.RemoveAllConformers()
        self.cache[smiles]["get_mol"] = mol

    @cache_decorator
    def get_atom_features(self, smiles: str, use_ogb_feat: bool = True) -> torch.Tensor:
        # compute atom features
        mol = self.get_mol(smiles)
        atom_features = self.get_atom_features_from_mol(mol, use_ogb_feat=use_ogb_feat)
        return atom_features

    @cache_decorator
    def get_atomic_numbers(self, smiles: str) -> torch.Tensor:
        # compute atomic numbers
        mol = self.get_mol(smiles)
        atomic_numbers = self.get_atomic_numbers_from_mol(mol)
        return atomic_numbers

    def get_atomic_numbers_from_mol(self, mol: Mol) -> torch.Tensor:
        mol = _require_mol(mol, "computing atomic numbers")
        atomic_numbers = torch.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()],
            dtype=torch.int32,
        )
        return atomic_numbers

    def get_atom_features_from_mol(
        self, mol: Mol, use_ogb_feat: bool = True
    ) -> torch.Tensor:
        mol = _require_mol(mol, "computing atom features")
        if use_ogb_feat:
            atom_features = torch.tensor(
                [atom_to_feature_vector(atom) for atom in mol.GetAtoms()],
                dtype=torch.float32,
            )
        else:
            atom_features = torch.tensor(
                [atom.GetFormalCharge() for atom in mol.GetAtoms()],
                dtype=torch.float32,
            ).view(-1, 1)
        return atom_features

    @cache_decorator
    def get_chiral_centers(self, smiles: str) -> torch.Tensor:
        # compute chiral centers
        mol = self.get_mol(smiles)
        chiral_index, chiral_nbr_index, chiral_tag = self.get_chiral_centers_from_mol(
            mol
        )

        self.cache[smiles]["chiral_centers"] = (
            chiral_index,
            chiral_nbr_index,
            chiral_tag,
        )
        return chiral_index, chiral_nbr_index, chiral_tag

    def get_chiral_centers_from_mol(self, mol: Mol) -> torch.Tensor:
        mol = _require_mol(mol, "computing chiral centers")
        chiral_index, chiral_nbr_index, chiral_tag = get_chiral_tensors(mol)
        return chiral_index, chiral_nbr_index, chiral_tag

    def get_mol_with_conformer(self, smiles: str, positions: torch.Tensor) -> Mol:
        mol = self.get_mol(smiles)
        return self.get_mol_with_conformer_from_mol(mol, positions)

    def get_mol_with_conformer_from_mol(
        self, mol: Mol, positions: torch.Tensor
    ) -> Mol:
        mol = Chem.Mol(_require_mol(mol, "adding a conformer"))
        mol.RemoveAllConformers()
        if positions.ndim != 2 or tuple(positions.shape) != (mol.GetNumAtoms(), 3):
            raise ValueError(
                "Conformer positions must have shape "
                f"[{mol.GetNumAtoms()}, 3], got {tuple(positions.shape)}."
            )
        mol.AddConformer(build_conformer(positions))
        return mol

    @cache_decorator
    def get_edge_index(
        self, smiles: str, use_edge_feat: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns edge index and edge attributes for a given smiles."""
        # compute edge index
        mol = self.get_mol(smiles)
        edge_index, edge_attr = self.get_edge_index_from_mol(
            mol, use_edge_feat=use_edge_feat
        )

        self.cache[smiles]["edge_index"] = edge_index
        self.cache[smiles]["edge_attr"] = edge_attr
        return edge_index, edge_attr

    def get_edge_index_from_mol(
        self, mol: Mol, use_edge_feat: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns edge index and edge attributes for a given mol object."""
        mol = _require_mol(mol, "computing edge indices")
        edge_index, edge_attr = compute_edge_index(mol, with_edge_attr=use_edge_feat)
        return edge_index, edge_attr

    @cache_decorator
    def get_rotatable_bond_features(
        self, smiles: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mol = self.get_mol(smiles)
        return self.get_rotatable_bond_features_from_mol(mol)

    def get_rotatable_bond_features_from_mol(
        self, mol: Mol
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return oriented rotatable bonds and sparse atom-bond influences.

        Each bond is oriented from the fixed endpoint to the endpoint in the
        deterministically selected smaller component.  The selected component
        supplies the atoms influenced by that bond.
        """

        mol = _require_mol(mol, "computing rotatable-bond features")
        bond_columns = []
        influence_columns = []
        bond_sides = rotatable_bond_sides(
            mol=mol,
            expected_num_atoms=mol.GetNumAtoms(),
        )
        for local_bond_idx, bond in enumerate(bond_sides):
            atom_a = int(bond["bond_atom_a"])
            atom_b = int(bond["bond_atom_b"])
            side_a = list(bond["side_a_atoms"])
            side_b = list(bond["side_b_atoms"])
            key_a = (len(side_a), tuple(side_a))
            key_b = (len(side_b), tuple(side_b))

            if key_a <= key_b:
                fixed_atom, rotating_atom = atom_b, atom_a
                affected_atoms = side_a
            else:
                fixed_atom, rotating_atom = atom_a, atom_b
                affected_atoms = side_b

            bond_columns.append((fixed_atom, rotating_atom))
            influence_columns.extend(
                (int(atom_idx), local_bond_idx) for atom_idx in affected_atoms
            )

        if bond_columns:
            rotatable_bond_index = torch.tensor(bond_columns, dtype=torch.long).t()
        else:
            rotatable_bond_index = torch.empty((2, 0), dtype=torch.long)

        if influence_columns:
            atom_bond_influence_index = torch.tensor(
                influence_columns,
                dtype=torch.long,
            ).t()
        else:
            atom_bond_influence_index = torch.empty((2, 0), dtype=torch.long)

        return rotatable_bond_index, atom_bond_influence_index

    def get_data_from_smiles(self, smiles: str) -> Data:
        mol = get_mol_from_smiles(smiles)  # added hs
        smiles_changed = mol_to_ordered_smiles(mol)
        node_attr = self.get_atom_features_from_mol(mol, True)
        chiral_index, chiral_nbr_index, chiral_tag = self.get_chiral_centers_from_mol(
            mol
        )
        edge_index, edge_attr = self.get_edge_index_from_mol(mol, False)
        rotatable_bond_index, atom_bond_influence_index = (
            self.get_rotatable_bond_features_from_mol(mol)
        )
        atomic_numbers = self.get_atomic_numbers_from_mol(mol)

        graph = MoleculeData(
            num_nodes=int(atomic_numbers.size(0)),
            atomic_numbers=atomic_numbers,
            smiles=smiles_changed,
            edge_index=edge_index,
            chiral_index=chiral_index,
            chiral_nbr_index=chiral_nbr_index,
            chiral_tag=chiral_tag,
            node_attr=node_attr,
            edge_attr=edge_attr,
            rotatable_bond_index=rotatable_bond_index,
            atom_bond_influence_index=atom_bond_influence_index,
        )
        return graph
