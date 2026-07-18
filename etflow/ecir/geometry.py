"""Rigid-invariant geometry diagnostics and internal-mode velocity operators."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor


ERROR_MODES = ("bond", "angle", "torsion", "ring", "clash", "chirality")


def _field(record: Any, name: str, default=None):
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def unique_bonds(edge_index: Tensor) -> Tensor:
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if edge_index.numel() == 0:
        return edge_index.new_empty((2, 0))
    keep = edge_index[0] < edge_index[1]
    return edge_index[:, keep]


def _precomputed_index(
    record: Any, name: str, width: int, device: torch.device
) -> Tensor | None:
    value = _field(record, name)
    if value is None:
        return None
    tensor = torch.as_tensor(value, device=device, dtype=torch.long)
    if tensor.numel() == 0:
        return tensor.reshape(width, 0).t() if width > 2 else tensor.reshape(width, 0)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a rank-two index tensor")
    if tensor.size(0) == width:
        return tensor if width == 2 else tensor.t()
    if tensor.size(1) == width:
        return tensor.t() if width == 2 else tensor
    raise ValueError(f"{name} width differs from {width}")


def training_bond_index(record: Any, device: torch.device) -> Tensor:
    precomputed = _precomputed_index(record, "canonical_bond_index", 2, device)
    if precomputed is not None:
        return precomputed
    edge_index = torch.as_tensor(_field(record, "edge_index"), device=device)
    return unique_bonds(edge_index).to(device)


def training_topology_indices(
    record: Any, num_atoms: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    bonds = training_bond_index(record, device)
    angles = _precomputed_index(record, "canonical_angle_index", 3, device)
    torsions = _precomputed_index(record, "canonical_torsion_index", 4, device)
    if angles is not None and torsions is not None:
        return bonds, angles, torsions
    if (angles is None) != (torsions is None):
        raise ValueError("precomputed angle and torsion indices must be provided together")
    edge_index = torch.as_tensor(_field(record, "edge_index"), device=device)
    rotatable = torch.as_tensor(
        _field(record, "rotatable_bond_index", torch.empty(2, 0)),
        device=device,
    )
    return (
        bonds,
        angle_triplets(edge_index.cpu(), num_atoms).to(device),
        torsion_quads(edge_index.cpu(), rotatable.cpu(), num_atoms).to(device),
    )


def training_ring_bond_index(record: Any, device: torch.device) -> Tensor:
    precomputed = _precomputed_index(
        record, "canonical_ring_bond_index", 2, device
    )
    if precomputed is not None:
        return precomputed
    edge_index = torch.as_tensor(_field(record, "edge_index"), device=device)
    flags = torch.as_tensor(
        _field(record, "bond_is_in_ring", torch.zeros(edge_index.size(1))),
        device=device,
        dtype=torch.bool,
    )
    keep = (edge_index[0] < edge_index[1]) & flags
    return edge_index[:, keep]


def angle_triplets(edge_index: Tensor, num_atoms: int) -> Tensor:
    bonds = unique_bonds(edge_index).t().tolist()
    neighbors: list[list[int]] = [[] for _ in range(num_atoms)]
    for a, b in bonds:
        neighbors[a].append(b)
        neighbors[b].append(a)
    triples = []
    for center, adjacent in enumerate(neighbors):
        for i, left in enumerate(adjacent):
            for right in adjacent[i + 1 :]:
                triples.append((left, center, right))
    return torch.tensor(triples, dtype=torch.long).reshape(-1, 3)


def torsion_quads(edge_index: Tensor, rotatable_bond_index: Tensor, num_atoms: int) -> Tensor:
    bonds = unique_bonds(edge_index).t().tolist()
    neighbors: list[list[int]] = [[] for _ in range(num_atoms)]
    for a, b in bonds:
        neighbors[a].append(b)
        neighbors[b].append(a)
    quads = []
    for a, b in torch.as_tensor(rotatable_bond_index, dtype=torch.long).t().tolist():
        left = next((n for n in neighbors[a] if n != b), None)
        right = next((n for n in neighbors[b] if n != a), None)
        if left is not None and right is not None:
            quads.append((left, a, b, right))
    return torch.tensor(quads, dtype=torch.long).reshape(-1, 4)


def bond_lengths(pos: Tensor, bonds: Tensor) -> Tensor:
    bonds = torch.as_tensor(bonds, device=pos.device, dtype=torch.long)
    if bonds.numel() == 0:
        return pos.new_empty(0)
    return torch.linalg.vector_norm(pos[bonds[0]] - pos[bonds[1]], dim=-1)


def bond_angles(pos: Tensor, triplets: Tensor) -> Tensor:
    triplets = torch.as_tensor(triplets, device=pos.device, dtype=torch.long)
    if triplets.numel() == 0:
        return pos.new_empty(0)
    left = pos[triplets[:, 0]] - pos[triplets[:, 1]]
    right = pos[triplets[:, 2]] - pos[triplets[:, 1]]
    cosine = (left * right).sum(-1) / (
        torch.linalg.vector_norm(left, dim=-1)
        * torch.linalg.vector_norm(right, dim=-1)
    ).clamp_min(1.0e-8)
    return torch.acos(cosine.clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7))


def dihedral_angles(pos: Tensor, quads: Tensor) -> Tensor:
    quads = torch.as_tensor(quads, device=pos.device, dtype=torch.long)
    if quads.numel() == 0:
        return pos.new_empty(0)
    p0, p1, p2, p3 = (pos[quads[:, index]] for index in range(4))
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    axis = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True).clamp_min(1.0e-8)
    v = b0 - (b0 * axis).sum(-1, keepdim=True) * axis
    w = b2 - (b2 * axis).sum(-1, keepdim=True) * axis
    v_norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    w_norm = torch.linalg.vector_norm(w, dim=-1, keepdim=True)
    valid = (v_norm > 1.0e-7) & (w_norm > 1.0e-7)
    v_unit = torch.where(valid, v / v_norm.clamp_min(1.0e-7), torch.zeros_like(v))
    w_unit = torch.where(valid, w / w_norm.clamp_min(1.0e-7), torch.zeros_like(w))
    x = (v_unit * w_unit).sum(-1)
    y = (torch.cross(axis, v_unit, dim=-1) * w_unit).sum(-1)
    # atan2(0, 0) has undefined gradients. Degenerate torsions contribute a
    # constant zero angle and are ignored by the directional loss.
    x = torch.where(valid.squeeze(-1), x, torch.ones_like(x))
    y = torch.where(valid.squeeze(-1), y, torch.zeros_like(y))
    return torch.atan2(y, x)


def circular_difference(a: Tensor, b: Tensor) -> Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def circular_difference_degrees(a: float | Tensor, b: float | Tensor) -> Tensor:
    a = torch.as_tensor(a, dtype=torch.float64) * (math.pi / 180.0)
    b = torch.as_tensor(b, dtype=torch.float64) * (math.pi / 180.0)
    return circular_difference(a, b).abs() * (180.0 / math.pi)


def clash_score(pos: Tensor, edge_index: Tensor, threshold: float = 1.0) -> Tensor:
    if pos.size(0) < 2:
        return pos.new_zeros(())
    excluded = torch.eye(pos.size(0), dtype=torch.bool, device=pos.device)
    edge_index = torch.as_tensor(edge_index, device=pos.device, dtype=torch.long)
    if edge_index.numel():
        excluded[edge_index[0], edge_index[1]] = True
    distances = torch.cdist(pos, pos)
    valid = torch.triu(~excluded, diagonal=1)
    if not bool(valid.any()):
        return pos.new_zeros(())
    penetration = (float(threshold) - distances[valid]).clamp_min(0.0)
    return penetration.mean()


def severe_clash(pos: Tensor, edge_index: Tensor, threshold: float = 0.6) -> Tensor:
    if pos.size(0) < 2:
        return torch.tensor(False, device=pos.device)
    excluded = torch.eye(pos.size(0), dtype=torch.bool, device=pos.device)
    edge_index = torch.as_tensor(edge_index, device=pos.device, dtype=torch.long)
    if edge_index.numel():
        excluded[edge_index[0], edge_index[1]] = True
    return ((torch.cdist(pos, pos) < threshold) & torch.triu(~excluded, 1)).any()


def chirality_mismatch_fraction(current: Tensor, target: Tensor, record: Any) -> Tensor:
    from .rdkit_utils import chiral_center_quads

    mismatches = []
    for center, first, second, third in chiral_center_quads(record):
        ids = (first, second, third)
        c_volume = torch.linalg.det(
            torch.stack([current[i] - current[center] for i in ids])
        )
        t_volume = torch.linalg.det(
            torch.stack([target[i] - target[center] for i in ids])
        )
        valid = (c_volume.abs() > 1.0e-5) & (t_volume.abs() > 1.0e-5)
        if bool(valid):
            mismatches.append((torch.sign(c_volume) != torch.sign(t_volume)).to(current.dtype))
    return torch.stack(mismatches).mean() if mismatches else current.new_zeros(())


def geometry_error_vector(current: Tensor, target: Tensor, record: Any) -> Tensor:
    edge_index = torch.as_tensor(_field(record, "edge_index"), device=current.device)
    bonds, angles, torsions = training_topology_indices(
        record, current.size(0), current.device
    )
    ring_bonds = training_ring_bond_index(record, current.device)

    def mean_abs(left: Tensor, right: Tensor, *, circular: bool = False) -> Tensor:
        if left.numel() == 0:
            return current.new_zeros(())
        delta = circular_difference(left, right) if circular else left - right
        return delta.abs().mean()

    return torch.stack(
        [
            mean_abs(bond_lengths(current, bonds), bond_lengths(target, bonds)),
            mean_abs(bond_angles(current, angles), bond_angles(target, angles)),
            mean_abs(dihedral_angles(current, torsions), dihedral_angles(target, torsions), circular=True),
            mean_abs(bond_lengths(current, ring_bonds), bond_lengths(target, ring_bonds)),
            clash_score(current, edge_index),
            chirality_mismatch_fraction(current, target, record),
        ]
    )


def internal_mode_velocities(pos: Tensor, velocity: Tensor, record: Any, eps: float = 1.0e-3) -> dict[str, Tensor]:
    """Return directional geometry derivatives ``B_mode(pos) velocity``."""

    bonds, angles, torsions = training_topology_indices(
        record, pos.size(0), pos.device
    )
    moved = pos + float(eps) * velocity
    return {
        "bond": (bond_lengths(moved, bonds) - bond_lengths(pos, bonds)) / float(eps),
        "angle": circular_difference(bond_angles(moved, angles), bond_angles(pos, angles)) / float(eps),
        "torsion": circular_difference(dihedral_angles(moved, torsions), dihedral_angles(pos, torsions)) / float(eps),
    }
