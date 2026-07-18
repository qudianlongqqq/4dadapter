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


def _cache_maps(
    record: Mapping[str, Any], num_atoms: int
) -> tuple[tuple[int, ...] | None, str]:
    atom_maps = record.get("atom_map_ids")
    input_maps = record.get("x_init_atom_map_ids")
    reference_maps = record.get("x_ref_atom_map_ids")
    if atom_maps is None and input_maps is None and reference_maps is None:
        return None, "absent"
    if atom_maps is None or input_maps is None or reference_maps is None:
        raise ValueError(
            "formal cache requires atom_map_ids, x_init_atom_map_ids, and x_ref_atom_map_ids"
        )
    first = tuple(int(value) for value in torch.as_tensor(atom_maps).view(-1).tolist())
    second = tuple(int(value) for value in torch.as_tensor(input_maps).view(-1).tolist())
    third = tuple(
        int(value) for value in torch.as_tensor(reference_maps).view(-1).tolist()
    )
    if len(first) != num_atoms or first != second or first != third:
        raise ValueError(
            "formal cache positional identity differs across cache, x_init, and x_ref"
        )
    if first == tuple(range(num_atoms)):
        return first, "zero_based_cache_position"
    if any(value <= 0 for value in first) or len(set(first)) != num_atoms:
        raise ValueError(
            "formal cache atom ids must be zero-based positions or complete semantic maps"
        )
    return first, "semantic_atom_map"


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


def _typed_graph(
    atomic_numbers: tuple[int, ...],
    bonds: Mapping[tuple[int, int], int],
    *,
    heavy_only: bool,
) -> nx.Graph:
    graph = nx.Graph()
    indices = [
        index
        for index, atomic_number in enumerate(atomic_numbers)
        if not heavy_only or atomic_number != 1
    ]
    graph.add_nodes_from(
        (index, {"z": atomic_numbers[index]}) for index in indices
    )
    included = set(indices)
    graph.add_edges_from(
        (left, right, {"bond_type": bond_type})
        for (left, right), bond_type in bonds.items()
        if left in included and right in included
    )
    return graph


def _rdkit_graph(mol: Chem.Mol, *, heavy_only: bool) -> nx.Graph:
    graph = nx.Graph()
    indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if not heavy_only or atom.GetAtomicNum() != 1
    ]
    graph.add_nodes_from(
        (index, {"z": int(mol.GetAtomWithIdx(index).GetAtomicNum())})
        for index in indices
    )
    included = set(indices)
    graph.add_edges_from(
        (
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            {"bond_type": _rdkit_bond_type(bond)},
        )
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in included and bond.GetEndAtomIdx() in included
    )
    return graph


def _complete_hydrogen_mapping(
    mol: Chem.Mol,
    heavy_mapping: Mapping[int, int],
    atomic_numbers: tuple[int, ...],
    cache_bonds: Mapping[tuple[int, int], int],
) -> dict[int, int]:
    mapping = {int(cache): int(rdkit) for cache, rdkit in heavy_mapping.items()}
    cache_neighbors: dict[int, list[int]] = {cache: [] for cache in mapping}
    for left, right in cache_bonds:
        if atomic_numbers[left] == 1 and atomic_numbers[right] != 1:
            cache_neighbors[right].append(left)
        elif atomic_numbers[right] == 1 and atomic_numbers[left] != 1:
            cache_neighbors[left].append(right)
        elif atomic_numbers[left] == 1 or atomic_numbers[right] == 1:
            raise ValueError("formal cache hydrogen must be singly attached to a heavy atom")
    for cache_heavy, rdkit_heavy in heavy_mapping.items():
        cache_hydrogens = sorted(cache_neighbors[int(cache_heavy)])
        rdkit_hydrogens = sorted(
            atom.GetIdx()
            for atom in mol.GetAtomWithIdx(int(rdkit_heavy)).GetNeighbors()
            if atom.GetAtomicNum() == 1
        )
        if len(cache_hydrogens) != len(rdkit_hydrogens):
            raise ValueError("explicit hydrogen counts differ for a mapped heavy atom")
        mapping.update(zip(cache_hydrogens, rdkit_hydrogens, strict=True))
    if len(mapping) != len(atomic_numbers):
        raise ValueError("explicit hydrogen mapping does not cover every cache atom")
    return mapping


def _local_hydrogen_classes(
    atomic_numbers: tuple[int, ...],
    cache_bonds: Mapping[tuple[int, int], int],
) -> tuple[tuple[int, ...], ...]:
    attached: dict[int, list[int]] = {}
    for left, right in cache_bonds:
        if atomic_numbers[left] == 1 and atomic_numbers[right] != 1:
            attached.setdefault(right, []).append(left)
        elif atomic_numbers[right] == 1 and atomic_numbers[left] != 1:
            attached.setdefault(left, []).append(right)
    classes = []
    for hydrogens in attached.values():
        group = tuple(sorted(hydrogens))
        if len(group) < 2:
            continue
        representative = group[0]
        for alternative in group[1:]:
            permutation = {index: index for index in range(len(atomic_numbers))}
            permutation[representative] = alternative
            permutation[alternative] = representative
            if not _is_typed_automorphism(
                permutation, atomic_numbers, cache_bonds
            ):
                raise ValueError(
                    "hydrogens attached to one heavy atom are not topology-equivalent"
                )
        classes.append(group)
    return tuple(sorted(classes, key=lambda values: values[0]))


def _is_typed_automorphism(
    permutation: Mapping[int, int],
    atomic_numbers: tuple[int, ...],
    cache_bonds: Mapping[tuple[int, int], int],
) -> bool:
    if set(permutation) != set(range(len(atomic_numbers))):
        return False
    if len(set(permutation.values())) != len(atomic_numbers):
        return False
    if any(
        atomic_numbers[source] != atomic_numbers[target]
        for source, target in permutation.items()
    ):
        return False
    transformed = {
        tuple(sorted((permutation[left], permutation[right]))): bond_type
        for (left, right), bond_type in cache_bonds.items()
    }
    return transformed == dict(cache_bonds)


def _choose_equivalent_mapping(
    mappings: list[dict[int, int]],
    atomic_numbers: tuple[int, ...],
    cache_bonds: Mapping[tuple[int, int], int],
) -> tuple[dict[int, int], tuple[tuple[int, ...], ...]]:
    ordered = sorted(
        mappings,
        key=lambda mapping: tuple(mapping[index] for index in range(len(atomic_numbers))),
    )
    selected = ordered[0]
    inverse = {rdkit: cache for cache, rdkit in selected.items()}
    equivalence_sets = [{index} for index in range(len(atomic_numbers))]

    def merge(source: int, target: int) -> None:
        merged = equivalence_sets[source] | equivalence_sets[target]
        for index in merged:
            equivalence_sets[index] = merged

    for group in _local_hydrogen_classes(atomic_numbers, cache_bonds):
        for index in group[1:]:
            merge(group[0], index)
    for alternative in ordered[1:]:
        permutation = {
            cache: inverse[alternative[cache]] for cache in range(len(atomic_numbers))
        }
        if not _is_typed_automorphism(permutation, atomic_numbers, cache_bonds):
            raise ValueError("candidate mappings vary outside typed topology classes")
        for source, target in permutation.items():
            merge(source, target)
    classes = tuple(
        sorted(
            {tuple(sorted(values)) for values in equivalence_sets},
            key=lambda values: values[0],
        )
    )
    return selected, classes


def _candidate_mapping(
    mol: Chem.Mol,
    atomic_numbers: tuple[int, ...],
    cache_maps: tuple[int, ...] | None,
    cache_map_kind: str,
    cache_bonds: Mapping[tuple[int, int], int],
) -> tuple[dict[int, int], tuple[tuple[int, ...], ...]]:
    if mol.GetNumAtoms() != len(atomic_numbers):
        raise ValueError("RDKit/cache atom counts differ")
    actual_numbers = tuple(int(atom.GetAtomicNum()) for atom in mol.GetAtoms())
    if Counter(actual_numbers) != Counter(atomic_numbers):
        raise ValueError("RDKit/cache elemental compositions differ")

    rdkit_maps = tuple(int(atom.GetAtomMapNum()) for atom in mol.GetAtoms())
    if cache_map_kind == "semantic_atom_map" and cache_maps is not None:
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
        return mapping, tuple((index,) for index in range(len(atomic_numbers)))

    cache_graph = _typed_graph(atomic_numbers, cache_bonds, heavy_only=True)
    rdkit_graph = _rdkit_graph(mol, heavy_only=True)
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        cache_graph,
        rdkit_graph,
        node_match=lambda left, right: left["z"] == right["z"],
        edge_match=lambda left, right: left["bond_type"] == right["bond_type"],
    )
    mappings = []
    for heavy_mapping in matcher.isomorphisms_iter():
        mapping = _complete_hydrogen_mapping(
            mol, heavy_mapping, atomic_numbers, cache_bonds
        )
        _validate_mapping(mol, mapping, atomic_numbers, cache_bonds)
        mappings.append(mapping)
    if not mappings:
        raise ValueError("RDKit/cache typed molecular graphs are not isomorphic")
    return _choose_equivalent_mapping(mappings, atomic_numbers, cache_bonds)


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
    cache_maps, cache_map_kind = _cache_maps(record, len(atomic_numbers))
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
            mapping, equivalence_classes = _candidate_mapping(
                candidate,
                atomic_numbers,
                cache_maps,
                cache_map_kind,
                cache_bonds,
            )
            order = [mapping[index] for index in range(len(atomic_numbers))]
            renumbered = Chem.RenumberAtoms(candidate, order)
            if tuple(atom.GetAtomicNum() for atom in renumbered.GetAtoms()) != atomic_numbers:
                raise ValueError("renumbered RDKit atomic-number sequence differs")
            if cache_map_kind == "semantic_atom_map" and cache_maps is not None:
                actual_maps = tuple(
                    int(atom.GetAtomMapNum()) for atom in renumbered.GetAtoms()
                )
                if actual_maps != cache_maps:
                    raise ValueError("renumbered RDKit atom-map identity differs")
            else:
                for atom in renumbered.GetAtoms():
                    atom.SetAtomMapNum(0)
            if _ordered_topology_signature(renumbered) != topology_signature:
                raise ValueError("renumbered RDKit topology signature differs")
            successes.append(
                (source, renumbered, tuple(order), equivalence_classes)
            )
        except ValueError as error:
            errors.append(f"{source}: {error}")
    unique = {}
    for source, mol, order, equivalence_classes in successes:
        identity = (
            tuple(atom.GetAtomicNum() for atom in mol.GetAtoms()),
            tuple(atom.GetAtomMapNum() for atom in mol.GetAtoms()),
            _ordered_topology_signature(mol),
        )
        unique.setdefault(identity, (source, mol, order, equivalence_classes))
    if len(unique) != 1:
        detail = "; ".join(errors) or "multiple incompatible mappings"
        raise ValueError(f"formal RDKit/cache mapping is not uniquely proven: {detail}")
    source, mol, order, equivalence_classes = next(iter(unique.values()))
    adapted = dict(record)
    adapted.update(
        {
            "_formal_rdkit_adapter_schema": FORMAL_ADAPTER_SCHEMA,
            "_formal_rdkit_mol": mol,
            "_formal_cache_to_rdkit": tuple(range(len(atomic_numbers))),
            "_formal_rdkit_original_order": order,
            "_formal_rdkit_mapping_source": source,
            "_formal_cache_identity_kind": cache_map_kind,
            "_formal_topology_equivalence_classes": equivalence_classes,
            "_formal_chiral_center_quads": _chiral_quads(mol),
        }
    )
    return adapted
