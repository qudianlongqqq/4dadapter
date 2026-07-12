"""Topology construction and caching for Global Coupled 4D joints.

The coordinate-independent fragment tree is shared with the validated global
torsion implementation.  This module adds an explicit cache and fail-closed
metadata without changing the legacy implementation.
"""

from __future__ import annotations

import time
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


@dataclass
class PreparedGlobalCoupled4DTopology:
    """Topology plus every coordinate-independent rollout tensor."""

    topology: MolecularKinematicTopology
    fragment_atom_index: Tensor
    fragment_index: Tensor
    fragment_counts: Tensor
    downstream_mask: Tensor
    ancestor_mask: Tensor
    joint_to_atom_incidence: Tensor
    jacobian_flat_index: Tensor

    @property
    def device(self) -> torch.device:
        return self.topology.parent_atom.device


class GlobalCoupled4DTopologyCache:
    """Cache CPU topology templates and move them to the caller's device.

    Invalid rings, disconnected inputs, and non-tree fragment graphs retain the
    fail-closed behavior of ``build_molecular_kinematic_topology``: they expose
    no joints and therefore make the model residual-only.
    """

    def __init__(self) -> None:
        self._cache: Dict[Tuple[object, ...], MolecularKinematicTopology] = {}
        self._device_cache: Dict[
            Tuple[Tuple[object, ...], str], MolecularKinematicTopology
        ] = {}
        self._prepared_cache: Dict[
            Tuple[Tuple[object, ...], str], PreparedGlobalCoupled4DTopology
        ] = {}
        self.stats = TopologyCacheStats()
        self.last_prepare_timing: dict[str, float | bool] = {}

    def _get_cpu(
        self,
        key: Tuple[object, ...],
        num_atoms: int,
        edge_index: Tensor,
        rotatable_bond_index: Tensor,
    ) -> tuple[MolecularKinematicTopology, bool, float]:
        started = time.perf_counter()
        hit = key in self._cache
        if hit:
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
        return topology, hit, time.perf_counter() - started

    def get(
        self,
        num_atoms: int,
        edge_index: Tensor,
        rotatable_bond_index: Tensor,
    ) -> MolecularKinematicTopology:
        key = topology_cache_key(num_atoms, edge_index, rotatable_bond_index)
        topology, _, _ = self._get_cpu(
            key, num_atoms, edge_index, rotatable_bond_index
        )
        device_key = (key, str(edge_index.device))
        if device_key not in self._device_cache:
            self._device_cache[device_key] = topology.to(edge_index.device)
        return self._device_cache[device_key]

    def get_prepared(
        self,
        num_atoms: int,
        edge_index: Tensor,
        rotatable_bond_index: Tensor,
    ) -> PreparedGlobalCoupled4DTopology:
        """Build masks/indices once and retain them on the rollout device."""

        key_started = time.perf_counter()
        key = topology_cache_key(num_atoms, edge_index, rotatable_bond_index)
        key_time = time.perf_counter() - key_started
        device_key = (key, str(edge_index.device))
        if device_key in self._prepared_cache:
            self.stats.hits += 1
            self.last_prepare_timing = {
                "cache_hit": True,
                "cache_key_time": key_time,
                "topology_construction_time": 0.0,
                "mask_construction_time": 0.0,
                "device_transfer_time": 0.0,
            }
            return self._prepared_cache[device_key]

        topology, _, topology_time = self._get_cpu(
            key, num_atoms, edge_index, rotatable_bond_index
        )
        transfer_started = time.perf_counter()
        device_topology = topology.to(edge_index.device)
        transfer_time = time.perf_counter() - transfer_started
        mask_started = time.perf_counter()
        num_joints = device_topology.num_joints
        fragment_atoms = [
            atom for fragment in device_topology.fragments for atom in fragment
        ]
        fragment_ids = [
            fragment_id
            for fragment_id, fragment in enumerate(device_topology.fragments)
            for _ in fragment
        ]
        fragment_atom_index = torch.tensor(
            fragment_atoms, dtype=torch.long, device=edge_index.device
        )
        fragment_index = torch.tensor(
            fragment_ids, dtype=torch.long, device=edge_index.device
        )
        fragment_counts = torch.bincount(
            fragment_index, minlength=len(device_topology.fragments)
        ).clamp_min(1)
        downstream_mask = torch.zeros(
            (num_joints, num_atoms), dtype=torch.bool, device=edge_index.device
        )
        if device_topology.affected_atom_index.numel():
            downstream_mask[
                device_topology.affected_joint_index,
                device_topology.affected_atom_index,
            ] = True
        ancestor_mask = (
            downstream_mask[:, device_topology.parent_atom]
            if num_joints
            else downstream_mask.new_zeros((0, 0))
        )
        jacobian_flat_index = (
            device_topology.affected_atom_index * num_joints
            + device_topology.affected_joint_index
        )
        prepared = PreparedGlobalCoupled4DTopology(
            topology=device_topology,
            fragment_atom_index=fragment_atom_index,
            fragment_index=fragment_index,
            fragment_counts=fragment_counts,
            downstream_mask=downstream_mask,
            ancestor_mask=ancestor_mask,
            joint_to_atom_incidence=downstream_mask,
            jacobian_flat_index=jacobian_flat_index,
        )
        self._prepared_cache[device_key] = prepared
        self.last_prepare_timing = {
            "cache_hit": False,
            "cache_key_time": key_time,
            "topology_construction_time": topology_time,
            "mask_construction_time": time.perf_counter() - mask_started,
            "device_transfer_time": transfer_time,
        }
        return prepared

    def clear(self) -> None:
        self._cache.clear()
        self._device_cache.clear()
        self._prepared_cache.clear()
        self.stats = TopologyCacheStats()
        self.last_prepare_timing = {}


def build_global_coupled_4d_topology(
    num_atoms: int,
    edge_index: Tensor,
    rotatable_bond_index: Tensor,
) -> MolecularKinematicTopology:
    """Public uncached constructor used by tests and standalone diagnostics."""

    return build_molecular_kinematic_topology(
        num_atoms, edge_index, rotatable_bond_index
    )
