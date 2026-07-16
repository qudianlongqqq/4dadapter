"""Controlled geometry corruptions used by ECIR synthetic-error training."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .geometry import clash_score, geometry_error_vector, unique_bonds


CORRUPTION_MODES = (
    "torsion",
    "multi_torsion",
    "bond_angle",
    "bond_length",
    "clash",
    "ring",
    "mixed",
    "zero",
)


def _field(record: Any, name: str, default=None):
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _randint(high: int, generator: torch.Generator | None) -> int:
    if high < 1:
        raise ValueError("cannot sample from an empty collection")
    return int(torch.randint(high, (1,), generator=generator).item())


def _affected_atoms(record: Any, bond_index: int) -> Tensor:
    influence = torch.as_tensor(
        _field(record, "atom_bond_influence_index", torch.empty(2, 0)),
        dtype=torch.long,
    )
    if influence.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    return influence[0, influence[1] == int(bond_index)].unique()


def _rotate(points: Tensor, pivot: Tensor, axis: Tensor, radians: float) -> Tensor:
    axis = axis / torch.linalg.vector_norm(axis).clamp_min(1.0e-8)
    shifted = points - pivot
    cosine = math.cos(float(radians))
    sine = math.sin(float(radians))
    return (
        shifted * cosine
        + torch.cross(axis.expand_as(shifted), shifted, dim=-1) * sine
        + axis * (shifted * axis).sum(-1, keepdim=True) * (1.0 - cosine)
        + pivot
    )


def _torsion(
    pos: Tensor,
    record: Any,
    generator: torch.Generator | None,
    amplitude: float,
    count: int,
) -> tuple[Tensor, list[int], list[int]]:
    rotatable = torch.as_tensor(
        _field(record, "rotatable_bond_index", torch.empty(2, 0)), dtype=torch.long
    )
    if rotatable.size(1) == 0:
        return pos.clone(), [], []
    order = torch.randperm(rotatable.size(1), generator=generator)[:count].tolist()
    result = pos.clone()
    affected_all: set[int] = set()
    for joint in order:
        left, right = rotatable[:, joint].tolist()
        affected = _affected_atoms(record, joint)
        if affected.numel() == 0:
            continue
        sign = -1.0 if _randint(2, generator) == 0 else 1.0
        result[affected] = _rotate(
            result[affected], result[left], result[right] - result[left], sign * amplitude
        )
        affected_all.update(affected.tolist())
    return result, order, sorted(affected_all)


def _bond_angle(
    pos: Tensor, record: Any, generator: torch.Generator | None, amplitude: float
) -> tuple[Tensor, list[int], list[int]]:
    rotatable = torch.as_tensor(
        _field(record, "rotatable_bond_index", torch.empty(2, 0)), dtype=torch.long
    )
    if rotatable.size(1) == 0:
        return pos.clone(), [], []
    joint = _randint(rotatable.size(1), generator)
    left, right = rotatable[:, joint].tolist()
    affected = _affected_atoms(record, joint)
    if affected.numel() == 0:
        return pos.clone(), [], []
    bond_axis = pos[right] - pos[left]
    trial = pos.new_tensor((1.0, 0.0, 0.0))
    if torch.abs(torch.dot(bond_axis / bond_axis.norm().clamp_min(1e-8), trial)) > 0.9:
        trial = pos.new_tensor((0.0, 1.0, 0.0))
    axis = torch.cross(bond_axis, trial, dim=0)
    result = pos.clone()
    result[affected] = _rotate(result[affected], result[left], axis, amplitude)
    return result, [joint], affected.tolist()


def _bond_length(
    pos: Tensor, record: Any, generator: torch.Generator | None, amplitude: float
) -> tuple[Tensor, list[int], list[int]]:
    rotatable = torch.as_tensor(
        _field(record, "rotatable_bond_index", torch.empty(2, 0)), dtype=torch.long
    )
    if rotatable.size(1) == 0:
        bonds = unique_bonds(torch.as_tensor(_field(record, "edge_index")))
        if bonds.size(1) == 0:
            return pos.clone(), [], []
        left, right = bonds[:, _randint(bonds.size(1), generator)].tolist()
        affected = torch.tensor([right])
        joint_ids: list[int] = []
    else:
        joint = _randint(rotatable.size(1), generator)
        left, right = rotatable[:, joint].tolist()
        affected = _affected_atoms(record, joint)
        joint_ids = [joint]
    axis = pos[right] - pos[left]
    axis = axis / torch.linalg.vector_norm(axis).clamp_min(1.0e-8)
    result = pos.clone()
    result[affected] += amplitude * axis
    return result, joint_ids, affected.tolist()


def _clash(
    pos: Tensor, record: Any, generator: torch.Generator | None, target_distance: float
) -> tuple[Tensor, list[int], list[int]]:
    edge_index = torch.as_tensor(_field(record, "edge_index"), dtype=torch.long)
    bonded = {tuple(pair) for pair in edge_index.t().tolist()}
    candidates = [
        (i, j)
        for i in range(pos.size(0))
        for j in range(i + 1, pos.size(0))
        if (i, j) not in bonded and (j, i) not in bonded
    ]
    if not candidates:
        return pos.clone(), [], []
    left, right = candidates[_randint(len(candidates), generator)]
    direction = pos[right] - pos[left]
    if torch.linalg.vector_norm(direction) < 1.0e-6:
        direction = pos.new_tensor((1.0, 0.0, 0.0))
    direction = direction / torch.linalg.vector_norm(direction)
    result = pos.clone()
    result[right] = result[left] + float(target_distance) * direction
    return result, [], [left, right]


def _ring(
    pos: Tensor, record: Any, generator: torch.Generator | None, amplitude: float
) -> tuple[Tensor, list[int], list[int]]:
    edge_index = torch.as_tensor(_field(record, "edge_index"), dtype=torch.long)
    flags = torch.as_tensor(
        _field(record, "bond_is_in_ring", torch.zeros(edge_index.size(1))), dtype=torch.bool
    )
    atoms = edge_index[:, flags].unique() if flags.numel() else torch.empty(0, dtype=torch.long)
    if atoms.numel() == 0:
        return pos.clone(), [], []
    center = pos[atoms].mean(0)
    radial = pos[atoms] - center
    normal = torch.cross(radial[0], radial[min(1, atoms.numel() - 1)], dim=0)
    if torch.linalg.vector_norm(normal) < 1.0e-6:
        normal = pos.new_tensor((0.0, 0.0, 1.0))
    normal = normal / torch.linalg.vector_norm(normal)
    signs = torch.where(
        torch.arange(atoms.numel(), device=pos.device) % 2 == 0, 1.0, -1.0
    ).to(pos.dtype)
    result = pos.clone()
    result[atoms] += float(amplitude) * signs[:, None] * normal
    return result, [], atoms.tolist()


def corrupt_conformer(
    record: Any,
    *,
    mode: str,
    coordinates: Tensor | None = None,
    generator: torch.Generator | None = None,
    torsion_amplitude_degrees: float = 30.0,
    angle_amplitude_degrees: float = 10.0,
    bond_strain: float = 0.08,
    clash_distance: float = 0.35,
    ring_amplitude: float = 0.08,
) -> tuple[Tensor, dict[str, Any]]:
    if mode not in CORRUPTION_MODES:
        raise ValueError(f"unknown corruption mode: {mode}")
    original = torch.as_tensor(
        coordinates if coordinates is not None else _field(record, "x_ref_aligned", _field(record, "x_init")),
        dtype=torch.float32,
    )
    result = original.clone()
    affected_bonds: list[int] = []
    affected_atoms: list[int] = []
    amplitudes: list[float] = []

    sequence: Sequence[str]
    if mode == "mixed":
        sequence = ("torsion", "bond_length", "clash")
    else:
        sequence = (mode,)
    for operation in sequence:
        if operation == "torsion":
            amplitude = math.radians(torsion_amplitude_degrees)
            result, bonds, atoms = _torsion(result, record, generator, amplitude, 1)
        elif operation == "multi_torsion":
            amplitude = math.radians(torsion_amplitude_degrees)
            result, bonds, atoms = _torsion(result, record, generator, amplitude, 3)
        elif operation == "bond_angle":
            amplitude = math.radians(angle_amplitude_degrees)
            result, bonds, atoms = _bond_angle(result, record, generator, amplitude)
        elif operation == "bond_length":
            amplitude = float(bond_strain)
            result, bonds, atoms = _bond_length(result, record, generator, amplitude)
        elif operation == "clash":
            amplitude = float(clash_distance)
            result, bonds, atoms = _clash(result, record, generator, amplitude)
        elif operation == "ring":
            amplitude = float(ring_amplitude)
            result, bonds, atoms = _ring(result, record, generator, amplitude)
        elif operation == "zero":
            amplitude, bonds, atoms = 0.0, [], []
        else:
            raise AssertionError(operation)
        affected_bonds.extend(bonds)
        affected_atoms.extend(atoms)
        amplitudes.append(amplitude)

    error = geometry_error_vector(result, original, record)
    metadata = {
        "mode": mode,
        "affected_bonds": sorted(set(affected_bonds)),
        "affected_atoms": sorted(set(affected_atoms)),
        "amplitude": amplitudes,
        "correction_direction": (original - result),
        "pre_internal_metrics": torch.zeros_like(error),
        "post_internal_metrics": error,
        "pre_clash_score": float(clash_score(original, _field(record, "edge_index"))),
        "post_clash_score": float(clash_score(result, _field(record, "edge_index"))),
        "effective": not torch.equal(result, original),
    }
    return result, metadata
