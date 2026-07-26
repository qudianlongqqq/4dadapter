"""Frozen LSGO-B mechanism diagnostics; no training or xTB supervision."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor

from .bat_refinement import dihedral_angles
from .geometry import bond_lengths
from .learned_geometry import (
    GraphGeometry, direct_gradient_update, gaussian_nll, geometry_values,
    remove_rigid_component, stable_angle_cosine, trust_project,
)


def primitive_nll(coordinates: Tensor, graph: GraphGeometry, parameters: Mapping[str, Tensor]):
    bonds, angles = geometry_values(coordinates, graph)
    return (
        gaussian_nll(bonds, parameters["bond_mu"], parameters["bond_sigma"]),
        gaussian_nll(angles, parameters["angle_mu"], parameters["angle_sigma"]),
    )


def primitive_z(coordinates: Tensor, graph: GraphGeometry, parameters: Mapping[str, Tensor]):
    bonds, angles = geometry_values(coordinates, graph)
    return (
        (bonds - parameters["bond_mu"]) / parameters["bond_sigma"],
        (angles - parameters["angle_mu"]) / parameters["angle_sigma"],
    )


def ba_abnormality(coordinates: Tensor, graph: GraphGeometry, parameters: Mapping[str, Tensor]):
    bond_z, angle_z = primitive_z(coordinates, graph, parameters)
    bond = bond_z.square().mean() if bond_z.numel() else coordinates.new_zeros(())
    angle = angle_z.square().mean() if angle_z.numel() else coordinates.new_zeros(())
    return {"bond": bond, "angle": angle, "ba": torch.stack((bond, angle)).mean()}


def masked_objective(
    coordinates: Tensor, graph: GraphGeometry, parameters: Mapping[str, Tensor], family: str,
    *, bond_mask: Tensor | None = None, angle_mask: Tensor | None = None,
):
    family = family.upper(); bond_nll, angle_nll = primitive_nll(coordinates, graph, parameters)
    groups = []
    if family in {"B", "BA"}:
        selected = bond_nll if bond_mask is None else bond_nll[torch.as_tensor(bond_mask, device=bond_nll.device)]
        if selected.numel(): groups.append(selected.mean())
    if family in {"A", "BA"}:
        selected = angle_nll if angle_mask is None else angle_nll[torch.as_tensor(angle_mask, device=angle_nll.device)]
        if selected.numel(): groups.append(selected.mean())
    if not groups: return coordinates.sum() * 0.0, {"active_groups": 0, "active_bonds": 0, "active_angles": 0}
    return torch.stack(groups).mean(), {
        "active_groups": len(groups),
        "active_bonds": int(bond_nll.numel() if bond_mask is None else torch.as_tensor(bond_mask).sum()) if family in {"B", "BA"} else 0,
        "active_angles": int(angle_nll.numel() if angle_mask is None else torch.as_tensor(angle_mask).sum()) if family in {"A", "BA"} else 0,
    }


def masked_gradient_update(
    source: Tensor, graph: GraphGeometry, parameters: Mapping[str, Tensor], family: str, *,
    rms_budget: float, atom_cap: float, bond_mask: Tensor | None = None, angle_mask: Tensor | None = None,
) -> dict[str, Any]:
    family = family.upper()
    if family == "BA" and bond_mask is None and angle_mask is None:
        return direct_gradient_update(source, graph, parameters, rms_budget=rms_budget, atom_cap=atom_cap, steps=1)
    source = torch.as_tensor(source, dtype=torch.float64); coordinates = source.detach().clone().requires_grad_(True)
    objective, active = masked_objective(coordinates, graph, parameters, family, bond_mask=bond_mask, angle_mask=angle_mask)
    if active["active_groups"] == 0:
        return {"coordinates": source.clone(), "trace": [], "finite": True, "no_op": True, **active}
    gradient, = torch.autograd.grad(objective, coordinates)
    gradient = remove_rigid_component(gradient, coordinates)
    rms = torch.sqrt(gradient.square().sum(-1).mean())
    if not bool(torch.isfinite(gradient).all()) or float(rms.detach()) <= 1e-14:
        return {"coordinates": source.clone(), "trace": [], "finite": False, "no_op": True, **active}
    output, trust = trust_project(source, -gradient / rms * float(rms_budget), rms_budget=rms_budget, atom_cap=atom_cap)
    return {"coordinates": output, "trace": [{"objective": float(objective.detach()), "gradient_rms": float(rms.detach()), **trust}], "finite": bool(torch.isfinite(output).all()), "no_op": False, **active}


def internal_jacobians(coordinates: Tensor, graph: GraphGeometry, torsions: Tensor):
    coordinates = torch.as_tensor(coordinates, dtype=torch.float64)
    flat = coordinates.detach().clone().reshape(-1).requires_grad_(True)
    shape = coordinates.shape
    def bond_fn(value): return bond_lengths(value.reshape(shape), graph.bonds)
    def angle_fn(value): return stable_angle_cosine(value.reshape(shape), graph.angles)
    def torsion_fn(value): return dihedral_angles(value.reshape(shape), torsions)
    def jacobian(function, count):
        if count == 0: return coordinates.new_empty((0, flat.numel()))
        return torch.autograd.functional.jacobian(function, flat, vectorize=True).reshape(count, -1)
    return {
        "B": jacobian(bond_fn, graph.bonds.size(1)),
        "A": jacobian(angle_fn, graph.angles.size(0)),
        "T": jacobian(torsion_fn, torch.as_tensor(torsions).reshape(-1, 4).size(0)),
    }


def orthonormal_row_basis(matrix: Tensor, *, relative_tolerance: float = 1e-8, absolute_tolerance: float = 1e-10):
    matrix = torch.as_tensor(matrix, dtype=torch.float64)
    if not matrix.numel(): return matrix.new_empty((matrix.size(1), 0)), {"rank": 0, "maximum_singular": 0.0, "minimum_retained": 0.0}
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    threshold = max(float(absolute_tolerance), float(relative_tolerance) * float(singular.max()))
    rank = int((singular > threshold).sum())
    basis = vh[:rank].T.contiguous()
    return basis, {"rank": rank, "maximum_singular": float(singular.max()), "minimum_retained": float(singular[rank - 1]) if rank else 0.0, "threshold": threshold}


def force_projection(force: Tensor, matrix: Tensor, coordinates: Tensor, **tolerances):
    """Project a force after removing translation and rotation at its coordinates."""
    force = torch.as_tensor(force, dtype=torch.float64)
    coordinates = torch.as_tensor(coordinates, dtype=torch.float64, device=force.device)
    if force.shape != coordinates.shape:
        raise ValueError(f"force/coordinate shape mismatch: {force.shape} != {coordinates.shape}")
    force = remove_rigid_component(force, coordinates).reshape(-1)
    basis, audit = orthonormal_row_basis(matrix, **tolerances)
    projected = basis @ (basis.T @ force) if basis.size(1) else torch.zeros_like(force)
    norm = torch.linalg.vector_norm(force)
    return {"force_norm": float(norm), "projection_norm": float(torch.linalg.vector_norm(projected)), "fraction": float(torch.linalg.vector_norm(projected) / norm.clamp_min(1e-15)), **audit}


def joint_matrix(*matrices: Tensor):
    rows = [torch.as_tensor(value) for value in matrices if torch.as_tensor(value).numel()]
    if rows: return torch.cat(rows, dim=0)
    width = torch.as_tensor(matrices[0]).size(1) if matrices else 0
    return torch.empty((0, width), dtype=torch.float64)


def finite_difference_row(function, coordinates: Tensor, *, step: float = 1e-4):
    coordinates = torch.as_tensor(coordinates, dtype=torch.float64); flat = coordinates.reshape(-1)
    result = []
    for index in range(flat.numel()):
        plus, minus = flat.clone(), flat.clone(); plus[index] += step; minus[index] -= step
        result.append((function(plus.reshape_as(coordinates)) - function(minus.reshape_as(coordinates))) / (2 * step))
    return torch.stack(result, dim=-1)


def circular_finite_difference(function, coordinates: Tensor, *, step: float = 1e-4):
    coordinates = torch.as_tensor(coordinates, dtype=torch.float64); flat = coordinates.reshape(-1)
    result = []
    for index in range(flat.numel()):
        plus, minus = flat.clone(), flat.clone(); plus[index] += step; minus[index] -= step
        delta = torch.remainder(function(plus.reshape_as(coordinates)) - function(minus.reshape_as(coordinates)) + math.pi, 2 * math.pi) - math.pi
        result.append(delta / (2 * step))
    return torch.stack(result, dim=-1)
