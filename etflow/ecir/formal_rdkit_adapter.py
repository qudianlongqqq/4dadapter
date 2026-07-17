"""Strict formal-cache atom-order adaptation for RDKit consumers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Mapping

import networkx as nx
import torch
from rdkit import Chem

from etflow.data.flexbond_cache_schema import validate_graph_record


FORMAL_ADAPTER_SCHEMA = "ecir-mvr-formal-rdkit-adapter-v1"
_BOND_NAMES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")


def _tensor(record: Mapping[str, Any], key: str) -> torch.Tensor:
    if key not in record or record[key] is None:
        raise ValueError(f"formal cache record is missing {key}")
    return torch.as_tensor(record[key], dtype=torch.long).view(-1)


def _ordered_topology_signature(mol: Chem.Mol) -> str:
    atoms = [int(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    bonds = []
    for bond in mol.GetBonds():
        atom_a, atom_b = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        bonds.append(
            (
                int(atom_a),
                int(atom_b),
                str(bond.GetBondType()),
                bool(bond.GetIsAromatic()),
                bool(bond.IsInRing()),
            )
        )
    payload = json.dumps(
        {"atoms": atoms, "bonds": sorted(bonds)}, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_maps(record: Mapping[str, Any], num_atoms: int) -> tuple[int, ...] | None:
    atom_maps = record.get("atom_map_ids")
    input_maps = record.get("x_init_atom_map_ids")
    if atom_maps is None and input_maps is None:
        return None
    if atom_maps is None or input_maps is None:
        raise ValueError("formal cache requires both atom_map_ids and x_init_atom_map_ids")
    first = tuple(int(value) for value in torch.as_tensor(atom_maps).view(-1).tolist())
    second = tuple(int(value) for value in torch.as_tensor(input_maps).view(-1).tolist())
    if len(first) != num_atoms or first != second:
        raise ValueError("formal cache atom-map identity does not match x_init atom order")
    if any(value <= 0 for value in first) or len(set(first)) != num_atoms:
        raise ValueError("formal cache atom-map ids must be complete, positive, and unique")
    return first


def _cache_bonds(record: Mapping[str, Any]) -> dict[tuple[int, int], int]:
    edge_index = torch.as_tensor(record["edge_index"], dtype=torch.long)
    bond_type = torch.as_tensor(record["bond_type"], dtype=torch.long).view(-1)
    bonds: dict[tuple[int, int], int] = {}
    for position, (left, right) in enumerate(edge_index.t().tolist()):
        pair = tuple(sorted((int(left), int(right))))
        value = int(bond_type[position])
        if value < 0 or value >= len(_BOND_NAMES):
            raise ValueError(f"unsupported formal cache bond type: {value}")
        if pair in bonds and bonds[pair] != value:
            raise ValueError(f"inconsistent directed bond type for {pair}")
        bonds[pair] = value
    return bonds


def _rdkit_bond_type(bond: Chem.Bond) -> int:
    name = str(bond.GetBondType())
    if name not in _BOND_NAMES:
        raise ValueError(f"unsupported RDKit bond type: {name}")
    return _BOND_NAMES.index(name)


def _validate_mapping(
    mol: Chem.Mol,
    cache_to_rdkit: Mapping[int, int],
    atomic_numbers: tuple[int, ...],
    cache_bonds: Mapping[tuple[int, int], int],
) -> None:
    if set(cache_to_rdkit) != set(range(len(atomic_numbers))):
        raise ValueError("RDKit mapping does not cover every cache atom")
    if len(set(cache_to_rdkit.values())) != len(atomic_numbers):
        raise ValueError("RDKit mapping is not one-to-one")
    actual = tuple(
        int(mol.GetAtomWithIdx(cache_to_rdkit[index]).GetAtomicNum())
        for index in range(len(atomic_numbers))
    )
    if actual != atomic_numbers:
        raise ValueError("mapped RDKit atomic-number sequence differs from cache")
    mapped_bonds = {}
    inverse = {rdkit: cache for cache, rdkit in cache_to_rdkit.items()}
    for bond in mol.GetBonds():
        pair = tuple(
            sorted(
                (
                    inverse[bond.GetBeginAtomIdx()],
                    inverse[bond.GetEndAtomIdx()],
                )
            )
        )
        mapped_bonds[pair] = _rdkit_bond_type(bond)
    if mapped_bonds != dict(cache_bonds):
        raise ValueError("mapped RDKit bond endpoints or bond types differ from cache")


def _candidate_mapping(
    mol: Chem.Mol,
    atomic_numbers: tuple[int, ...],
    cache_maps: tuple[int, ...] | None,
    cache_bonds: Mapping[tuple[int, int], int],
) -> dict[int, int]:
    if mol.GetNumAtoms() != len(atomic_numbers):
        raise ValueError("RDKit/cache atom counts differ")
    actual_numbers = tuple(int(atom.GetAtomicNum()) for atom in mol.GetAtoms())
    if Counter(actual_numbers) != Counter(atomic_numbers):
        raise ValueError("RDKit/cache elemental compositions differ")

    rdkit_maps = tuple(int(atom.GetAtomMapNum()) for atom in mol.GetAtoms())
    if cache_maps is not None and any(rdkit_maps):
        if any(value <= 0 for value in rdkit_maps) or len(set(rdkit_maps)) != len(
            rdkit_maps
        ):
            raise ValueError("RDKit atom-map ids are incomplete or ambiguous")
        if set(rdkit_maps) != set(cache_maps):
            raise ValueError("RDKit/cache atom-map identity sets differ")
        by_map = {value: index for index, value in enumerate(rdkit_maps)}
        mapping = {
            cache_index: by_map[map_id]
            for cache_index, map_id in enumerate(cache_maps)
        }
        _validate_mapping(mol, mapping, atomic_numbers, cache_bonds)
        return mapping

    cache_graph = nx.Graph()
    cache_graph.add_nodes_from(
        (index, {"z": atomic_number})
        for index, atomic_number in enumerate(atomic_numbers)
    )
    cache_graph.add_edges_from(
        (left, right, {"bond_type": bond_type})
        for (left, right), bond_type in cache_bonds.items()
    )
    rdkit_graph = nx.Graph()
    rdkit_graph.add_nodes_from(
        (atom.GetIdx(), {"z": int(atom.GetAtomicNum())}) for atom in mol.GetAtoms()
    )
    rdkit_graph.add_edges_from(
        (
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            {"bond_type": _rdkit_bond_type(bond)},
        )
        for bond in mol.GetBonds()
    )
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        cache_graph,
        rdkit_graph,
        node_match=lambda left, right: left["z"] == right["z"],
        edge_match=lambda left, right: left["bond_type"] == right["bond_type"],
    )
    iterator = matcher.isomorphisms_iter()
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("RDKit/cache typed molecular graphs are not isomorphic") from exc
    try:
        next(iterator)
    except StopIteration:
        mapping = {int(cache): int(rdkit) for cache, rdkit in first.items()}
        _validate_mapping(mol, mapping, atomic_numbers, cache_bonds)
        return mapping
    raise ValueError("RDKit/cache atom mapping is not unique")


def _parse_candidates(smiles: str) -> list[tuple[str, Chem.Mol]]:
    base = Chem.MolFromSmiles(smiles)
    if base is None:
        raise ValueError("RDKit could not parse formal cache SMILES")
    candidates = [("MolFromSmiles", base)]
    try:
        candidates.append(("MolFromSmiles+AddHs", Chem.AddHs(Chem.Mol(base))))
    except Exception:
        pass
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    explicit = Chem.MolFromSmiles(smiles, parameters)
    if explicit is not None:
        candidates.append(("MolFromSmiles(removeHs=False)", explicit))
    return candidates


def _chiral_quads(mol: Chem.Mol) -> tuple[tuple[int, int, int, int], ...]:
    result = []
    for center, _ in Chem.FindMolChiralCenters(
        mol, includeUnassigned=False, useLegacyImplementation=False
    ):
        neighbors = sorted(
            atom.GetIdx() for atom in mol.GetAtomWithIdx(center).GetNeighbors()
        )
        if len(neighbors) >= 3:
            result.append((center, neighbors[0], neighbors[1], neighbors[2]))
    return tuple(result)


def adapt_formal_cache_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a runtime-only record whose RDKit molecule exactly matches x_init."""

    if record.get("_formal_rdkit_adapter_schema") == FORMAL_ADAPTER_SCHEMA:
        return dict(record)
    validate_graph_record(record)
    atomic = _tensor(record, "atomic_numbers")
    x_init_atomic = _tensor(record, "x_init_atomic_numbers")
    if not torch.equal(atomic, x_init_atomic):
        raise ValueError("formal cache atomic_numbers differ from x_init_atomic_numbers")
    atomic_numbers = tuple(int(value) for value in atomic.tolist())
    cache_maps = _cache_maps(record, len(atomic_numbers))
    cache_bonds = _cache_bonds(record)
    topology_signature = str(record.get("topology_signature", ""))
    if not topology_signature:
        raise ValueError("formal cache topology_signature is required")
    smiles = str(record.get("smiles", ""))
    if not smiles:
        raise ValueError("formal cache SMILES is required")

    successes = []
    errors = []
    for source, candidate in _parse_candidates(smiles):
        try:
            mapping = _candidate_mapping(
                candidate, atomic_numbers, cache_maps, cache_bonds
            )
            order = [mapping[index] for index in range(len(atomic_numbers))]
            renumbered = Chem.RenumberAtoms(candidate, order)
            if tuple(atom.GetAtomicNum() for atom in renumbered.GetAtoms()) != atomic_numbers:
                raise ValueError("renumbered RDKit atomic-number sequence differs")
            if cache_maps is not None:
                actual_maps = tuple(
                    int(atom.GetAtomMapNum()) for atom in renumbered.GetAtoms()
                )
                if actual_maps != cache_maps:
                    raise ValueError("renumbered RDKit atom-map identity differs")
            if _ordered_topology_signature(renumbered) != topology_signature:
                raise ValueError("renumbered RDKit topology signature differs")
            successes.append((source, renumbered, tuple(order)))
        except ValueError as error:
            errors.append(f"{source}: {error}")
    unique = {}
    for source, mol, order in successes:
        identity = (
            tuple(atom.GetAtomicNum() for atom in mol.GetAtoms()),
            tuple(atom.GetAtomMapNum() for atom in mol.GetAtoms()),
            _ordered_topology_signature(mol),
        )
        unique.setdefault(identity, (source, mol, order))
    if len(unique) != 1:
        detail = "; ".join(errors) or "multiple incompatible mappings"
        raise ValueError(f"formal RDKit/cache mapping is not uniquely proven: {detail}")
    source, mol, order = next(iter(unique.values()))
    adapted = dict(record)
    adapted.update(
        {
            "_formal_rdkit_adapter_schema": FORMAL_ADAPTER_SCHEMA,
            "_formal_rdkit_mol": mol,
            "_formal_cache_to_rdkit": tuple(range(len(atomic_numbers))),
            "_formal_rdkit_original_order": order,
            "_formal_rdkit_mapping_source": source,
            "_formal_chiral_center_quads": _chiral_quads(mol),
        }
    )
    return adapted

