"""Local-geometry diagnostics for Cartesian conformer interpolation paths."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def unique_undirected_edges(edge_index: Tensor) -> Tensor:
    pairs = {
        tuple(sorted((int(atom_a), int(atom_b))))
        for atom_a, atom_b in edge_index.t().tolist()
        if int(atom_a) != int(atom_b)
    }
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    return torch.tensor(sorted(pairs), dtype=torch.long, device=edge_index.device).t()


def angle_triplets(edge_index: Tensor, num_atoms: int) -> Tensor:
    neighbors = [set() for _ in range(num_atoms)]
    for atom_a, atom_b in unique_undirected_edges(edge_index).t().tolist():
        neighbors[atom_a].add(atom_b)
        neighbors[atom_b].add(atom_a)
    triplets = []
    for center, adjacent in enumerate(neighbors):
        ordered = sorted(adjacent)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                triplets.append((left, center, right))
    if not triplets:
        return torch.empty((3, 0), dtype=torch.long, device=edge_index.device)
    return torch.tensor(triplets, dtype=torch.long, device=edge_index.device).t()


def rotatable_dihedral_quads(
    edge_index: Tensor, rotatable_bond_index: Tensor, num_atoms: int
) -> Tensor:
    neighbors = [set() for _ in range(num_atoms)]
    for atom_a, atom_b in unique_undirected_edges(edge_index).t().tolist():
        neighbors[atom_a].add(atom_b)
        neighbors[atom_b].add(atom_a)
    quads = []
    for center_a, center_b in rotatable_bond_index.t().tolist():
        left = sorted(neighbors[center_a].difference({center_b}))
        right = sorted(neighbors[center_b].difference({center_a}))
        if left and right:
            quads.append((left[0], center_a, center_b, right[0]))
    if not quads:
        return torch.empty((4, 0), dtype=torch.long, device=edge_index.device)
    return torch.tensor(quads, dtype=torch.long, device=edge_index.device).t()


def bond_lengths(pos: Tensor, edges: Tensor) -> Tensor:
    if edges.size(1) == 0:
        return pos.new_empty(0)
    return torch.linalg.norm(pos[edges[0]] - pos[edges[1]], dim=-1)


def bond_angles(pos: Tensor, triplets: Tensor, eps: float = 1.0e-8) -> Tensor:
    if triplets.size(1) == 0:
        return pos.new_empty(0)
    left = pos[triplets[0]] - pos[triplets[1]]
    right = pos[triplets[2]] - pos[triplets[1]]
    cosine = (left * right).sum(dim=-1) / (
        torch.linalg.norm(left, dim=-1) * torch.linalg.norm(right, dim=-1)
    ).clamp_min(eps)
    return torch.acos(cosine.clamp(-1.0, 1.0))


def dihedral_angles(pos: Tensor, quads: Tensor, eps: float = 1.0e-8) -> Tensor:
    if quads.size(1) == 0:
        return pos.new_empty(0)
    p0, p1, p2, p3 = (pos[quads[index]] for index in range(4))
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    axis = b1 / torch.linalg.norm(b1, dim=-1, keepdim=True).clamp_min(eps)
    v = b0 - (b0 * axis).sum(dim=-1, keepdim=True) * axis
    w = b2 - (b2 * axis).sum(dim=-1, keepdim=True) * axis
    x = (v * w).sum(dim=-1)
    y = (torch.cross(axis, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


def wrapped_angle_delta(first: Tensor, second: Tensor) -> Tensor:
    return torch.atan2(torch.sin(first - second), torch.cos(first - second))


def _mean_max_fraction(error: Tensor, threshold: float) -> tuple[float, float, float]:
    if error.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(error.mean()),
        float(error.max()),
        float((error > threshold).float().mean()),
    )


def path_geometry_metrics(
    x_init: Tensor,
    x_ref_aligned: Tensor,
    x_t: Tensor,
    t: float,
    edge_index: Tensor,
    rotatable_bond_index: Tensor,
    *,
    collision_distance: float = 0.7,
) -> dict[str, float | int]:
    """Compare Cartesian x_t geometry with endpoint internal-coordinate interpolation."""

    edges = unique_undirected_edges(edge_index)
    triplets = angle_triplets(edge_index, x_init.size(0))
    quads = rotatable_dihedral_quads(
        edge_index, rotatable_bond_index, x_init.size(0)
    )

    length_init, length_ref, length_t = (
        bond_lengths(pos, edges) for pos in (x_init, x_ref_aligned, x_t)
    )
    expected_length = (1.0 - t) * length_init + t * length_ref
    length_error = (length_t - expected_length).abs() / expected_length.clamp_min(1.0e-8)
    length_mean, length_max, length_bad = _mean_max_fraction(length_error, 0.10)
    length_low = torch.minimum(length_init, length_ref) * 0.9
    length_high = torch.maximum(length_init, length_ref) * 1.1
    envelope_violation = (
        ((length_t < length_low) | (length_t > length_high)).float().mean()
        if length_t.numel()
        else x_t.new_tensor(float("nan"))
    )

    angle_init, angle_ref, angle_t = (
        bond_angles(pos, triplets) for pos in (x_init, x_ref_aligned, x_t)
    )
    expected_angle = (1.0 - t) * angle_init + t * angle_ref
    angle_error = (angle_t - expected_angle).abs()
    angle_mean, angle_max, angle_bad = _mean_max_fraction(
        angle_error, math.radians(15.0)
    )

    torsion_init, torsion_ref, torsion_t = (
        dihedral_angles(pos, quads) for pos in (x_init, x_ref_aligned, x_t)
    )
    torsion_delta = wrapped_angle_delta(torsion_ref, torsion_init)
    expected_torsion = torsion_init + t * torsion_delta
    torsion_error = wrapped_angle_delta(torsion_t, expected_torsion).abs()
    torsion_mean, torsion_max, torsion_bad = _mean_max_fraction(
        torsion_error, math.radians(30.0)
    )

    bonded = {tuple(sorted(pair)) for pair in edges.t().tolist()}
    nonbonded_distances = []
    for atom_a in range(x_t.size(0)):
        for atom_b in range(atom_a + 1, x_t.size(0)):
            if (atom_a, atom_b) not in bonded:
                nonbonded_distances.append(torch.linalg.norm(x_t[atom_a] - x_t[atom_b]))
    if nonbonded_distances:
        distances = torch.stack(nonbonded_distances)
        collision_fraction = float((distances < collision_distance).float().mean())
        minimum_nonbonded_distance = float(distances.min())
    else:
        collision_fraction = 0.0
        minimum_nonbonded_distance = float("nan")

    return {
        "num_bonds": int(edges.size(1)),
        "num_angles": int(triplets.size(1)),
        "num_torsions": int(quads.size(1)),
        "bond_length_rel_error_mean": length_mean,
        "bond_length_rel_error_max": length_max,
        "bond_length_fraction_gt_10pct": length_bad,
        "bond_length_envelope_violation_fraction": float(envelope_violation),
        "angle_error_deg_mean": math.degrees(angle_mean),
        "angle_error_deg_max": math.degrees(angle_max),
        "angle_fraction_gt_15deg": angle_bad,
        "torsion_error_deg_mean": math.degrees(torsion_mean),
        "torsion_error_deg_max": math.degrees(torsion_max),
        "torsion_fraction_gt_30deg": torsion_bad,
        "nonbonded_collision_fraction": collision_fraction,
        "minimum_nonbonded_distance": minimum_nonbonded_distance,
    }
