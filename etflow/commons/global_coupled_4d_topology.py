"""Topology construction and caching for Global Coupled 4D joints.

The coordinate-independent fragment tree is shared with the validated global
torsion implementation.  This module adds an explicit cache and fail-closed
metadata without changing the legacy implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor

from .molecular_kinematics import (
    MolecularKinematicTopology,
    build_molecular_kinematic_topology,
)


def topology_cache_key(
    num_atoms: int, edge_index: Tensor, rotatable_bond_index: Tensor
) -> Tuple[object, ...]:
    """Return a deterministic, coordinate-free topology key."""

    edges = tuple(int(value) for value in edge_index.detach().cpu().reshape(-1))
    joints = tuple(
        int(value) for value in rotatable_bond_index.detach().cpu().reshape(-1)
    )
    return int(num_atoms), tuple(edge_index.shape), edges, tuple(rotatable_bond_index.shape), joints


@dataclass
class TopologyCacheStats:
    hits: int = 0
    misses: int = 0
    fallbacks: int = 0

    @property
    def fallback_rate(self) -> float:
        total = self.hits + self.misses
        return self.fallbacks / total if total else 0.0


class GlobalCoupled4DTopologyCache:
    """Cache CPU topology templates and move them to the caller's device.

    Invalid rings, disconnected inputs, and non-tree fragment graphs retain the
    fail-closed behavior of ``build_molecular_kinematic_topology``: they expose
    no joints and therefore make the model residual-only.
    """

    def __init__(self) -> None:
        self._cache: Dict[Tuple[object, ...], MolecularKinematicTopology] = {}
        self.stats = TopologyCacheStats()

    def get(
        self,
        num_atoms: int,
        edge_index: Tensor,
        rotatable_bond_index: Tensor,
    ) -> MolecularKinematicTopology:
        key = topology_cache_key(num_atoms, edge_index, rotatable_bond_index)
        if key in self._cache:
            self.stats.hits += 1
            topology = self._cache[key]
        else:
            self.stats.misses += 1
            topology = build_molecular_kinematic_topology(
                num_atoms,
                edge_index.detach().cpu(),
                rotatable_bond_index.detach().cpu(),
            )
            self._cache[key] = topology
        if not topology.valid:
            self.stats.fallbacks += 1
        return topology.to(edge_index.device)

    def clear(self) -> None:
        self._cache.clear()
        self.stats = TopologyCacheStats()


def build_global_coupled_4d_topology(
    num_atoms: int,
    edge_index: Tensor,
    rotatable_bond_index: Tensor,
) -> MolecularKinematicTopology:
    """Public uncached constructor used by tests and standalone diagnostics."""

    return build_molecular_kinematic_topology(
        num_atoms, edge_index, rotatable_bond_index
    )

