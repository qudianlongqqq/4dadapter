"""Strict cache-to-RDKit graph mapping and stereocenter lookup."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

import networkx as nx
import torch
from rdkit import Chem

from .geometry import unique_bonds


def _field(record: Any, name: str, default=None):
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


@lru_cache(maxsize=8192)
def _cached_stereocenters(
    smiles: str,
    atomic_numbers: tuple[int, ...],
    bonds: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int, int, int], ...]:
    base = Chem.MolFromSmiles(smiles)
    if base is None:
        return ()
    mol = Chem.AddHs(base)
    if mol.GetNumAtoms() != len(atomic_numbers):
        return ()
    cache_graph = nx.Graph()
    cache_graph.add_nodes_from((i, {"z": z}) for i, z in enumerate(atomic_numbers))
    cache_graph.add_edges_from(bonds)
    rdkit_graph = nx.Graph()
    rdkit_graph.add_nodes_from(
        (atom.GetIdx(), {"z": atom.GetAtomicNum()}) for atom in mol.GetAtoms()
    )
    rdkit_graph.add_edges_from(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()
    )
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        cache_graph,
        rdkit_graph,
        node_match=lambda left, right: left["z"] == right["z"],
    )
    try:
        cache_to_rdkit = next(matcher.isomorphisms_iter())
    except StopIteration:
        return ()
    rdkit_to_cache = {rdkit: cache for cache, rdkit in cache_to_rdkit.items()}
    result = []
    for center, _ in Chem.FindMolChiralCenters(
        mol, includeUnassigned=False, useLegacyImplementation=False
    ):
        neighbors = sorted(atom.GetIdx() for atom in mol.GetAtomWithIdx(center).GetNeighbors())
        if len(neighbors) < 3:
            continue
        result.append(
            (
                rdkit_to_cache[center],
                rdkit_to_cache[neighbors[0]],
                rdkit_to_cache[neighbors[1]],
                rdkit_to_cache[neighbors[2]],
            )
        )
    return tuple(result)


def chiral_center_quads(record: Any) -> tuple[tuple[int, int, int, int], ...]:
    formal = _field(record, "_formal_chiral_center_quads")
    if formal is not None:
        return tuple(tuple(int(value) for value in row) for row in formal)
    smiles = str(_field(record, "smiles", ""))
    if not smiles:
        return ()
    atomic_numbers = tuple(
        int(value) for value in torch.as_tensor(_field(record, "atomic_numbers")).tolist()
    )
    bonds = tuple(
        tuple(int(value) for value in pair)
        for pair in unique_bonds(torch.as_tensor(_field(record, "edge_index"))).t().tolist()
    )
    return _cached_stereocenters(smiles, atomic_numbers, bonds)
