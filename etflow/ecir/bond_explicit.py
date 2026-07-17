"""Global bond-Jacobian projection primitives for MCVR Stage D."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def bond_length_residual(
    input_coordinates: Tensor, target_coordinates: Tensor, bonds: Tensor
) -> Tensor:
    input_coordinates = torch.as_tensor(input_coordinates)
    target_coordinates = torch.as_tensor(
        target_coordinates, device=input_coordinates.device, dtype=input_coordinates.dtype
    )
    bonds = torch.as_tensor(bonds, device=input_coordinates.device, dtype=torch.long)
    input_lengths = torch.linalg.vector_norm(
        input_coordinates[bonds[1]] - input_coordinates[bonds[0]], dim=-1
    )
    target_lengths = torch.linalg.vector_norm(
        target_coordinates[bonds[1]] - target_coordinates[bonds[0]], dim=-1
    )
    return target_lengths - input_lengths


def bond_length_jacobian(coordinates: Tensor, bonds: Tensor) -> Tensor:
    coordinates = torch.as_tensor(coordinates)
    bonds = torch.as_tensor(bonds, device=coordinates.device, dtype=torch.long)
    atom_count = int(coordinates.shape[0])
    left, right = bonds[0], bonds[1]
    vectors = coordinates[right] - coordinates[left]
    directions = vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp_min(1.0e-12)
    jacobian = coordinates.new_zeros((bonds.shape[1], atom_count, 3))
    rows = torch.arange(bonds.shape[1], device=coordinates.device)
    jacobian[rows, left] = -directions
    jacobian[rows, right] = directions
    return jacobian.reshape(bonds.shape[1], atom_count * 3)


def solve_bond_cartesian_correction(
    coordinates: Tensor,
    bonds: Tensor,
    residual: Tensor,
    *,
    damping: float = 1.0e-4,
    max_condition: float = 1.0e10,
) -> tuple[Tensor, dict[str, Any]]:
    coordinates = torch.as_tensor(coordinates)
    bonds = torch.as_tensor(bonds, device=coordinates.device, dtype=torch.long)
    residual = torch.as_tensor(residual, device=coordinates.device, dtype=coordinates.dtype)
    atom_count = int(coordinates.shape[0])
    correction = torch.zeros_like(coordinates)
    diagnostics: dict[str, Any] = {
        "status": "FALLBACK_ZERO", "bond_count": int(bonds.shape[1]),
        "rank": 0, "condition": math.inf, "linear_residual_norm": math.inf,
        "relative_linear_residual": math.inf, "translation_norm": 0.0,
    }
    if bonds.numel() == 0:
        diagnostics.update({
            "status": "NO_BONDS", "condition": 1.0,
            "linear_residual_norm": 0.0, "relative_linear_residual": 0.0,
        })
        return correction, diagnostics
    try:
        jacobian = bond_length_jacobian(coordinates, bonds)
        gram = jacobian @ jacobian.transpose(0, 1)
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        damped = gram + float(damping) * identity
        singular = torch.linalg.svdvals(damped)
        condition = float(singular.max() / singular.min().clamp_min(1.0e-15))
        rank = int(torch.linalg.matrix_rank(jacobian).item())
        if not math.isfinite(condition) or condition > float(max_condition):
            diagnostics.update({"rank": rank, "condition": condition, "status": "CONDITION_FALLBACK"})
            return correction, diagnostics
        dual = torch.linalg.solve(damped, residual)
        correction = (jacobian.transpose(0, 1) @ dual).reshape(atom_count, 3)
        correction = correction - correction.mean(dim=0, keepdim=True)
        linear_error = jacobian @ correction.reshape(-1) - residual
        error_norm = float(torch.linalg.vector_norm(linear_error))
        relative_error = error_norm / max(float(torch.linalg.vector_norm(residual)), 1.0e-12)
        translation_norm = float(torch.linalg.vector_norm(correction.mean(dim=0)))
        if not bool(torch.isfinite(correction).all()):
            correction = torch.zeros_like(coordinates)
            diagnostics.update({"rank": rank, "condition": condition, "status": "NONFINITE_FALLBACK"})
            return correction, diagnostics
        diagnostics.update({
            "status": "SOLVED", "rank": rank, "condition": condition,
            "linear_residual_norm": error_norm,
            "relative_linear_residual": relative_error,
            "translation_norm": translation_norm,
        })
        return correction, diagnostics
    except (RuntimeError, ValueError) as error:
        diagnostics["error"] = f"{type(error).__name__}: {error}"
        return torch.zeros_like(coordinates), diagnostics


def bounded_bond_residual(
    raw_residual: Tensor, confidence_logit: Tensor, *, max_abs_residual: float
) -> Tensor:
    return (
        float(max_abs_residual)
        * torch.tanh(torch.as_tensor(raw_residual))
        * torch.sigmoid(torch.as_tensor(confidence_logit))
    )


def differentiable_bond_projection(
    coordinates: Tensor,
    bonds: Tensor,
    residual: Tensor,
    *,
    damping: float = 1.0e-4,
) -> tuple[Tensor, Tensor]:
    """Differentiable damped projection with a fail-closed zero correction."""

    coordinates = torch.as_tensor(coordinates)
    bonds = torch.as_tensor(bonds, device=coordinates.device, dtype=torch.long)
    residual = torch.as_tensor(residual, device=coordinates.device, dtype=coordinates.dtype)
    if bonds.numel() == 0:
        return torch.zeros_like(coordinates), coordinates.new_zeros(())
    jacobian = bond_length_jacobian(coordinates, bonds)
    gram = jacobian @ jacobian.transpose(0, 1)
    damped = gram + float(damping) * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    dual, info = torch.linalg.solve_ex(damped, residual)
    correction = (jacobian.transpose(0, 1) @ dual).reshape_as(coordinates)
    correction = correction - correction.mean(dim=0, keepdim=True)
    valid = (info == 0) & torch.isfinite(correction).all()
    correction = torch.where(valid, correction, torch.zeros_like(correction))
    return correction, (~valid).to(coordinates.dtype)


def batched_bond_projection(
    coordinates: Tensor,
    bonds: Tensor,
    residual: Tensor,
    atom_batch: Tensor,
    *,
    damping: float = 1.0e-4,
) -> tuple[Tensor, Tensor]:
    """Project unique-bond residuals independently for each contiguous graph."""

    coordinates = torch.as_tensor(coordinates)
    bonds = torch.as_tensor(bonds, device=coordinates.device, dtype=torch.long)
    residual = torch.as_tensor(residual, device=coordinates.device, dtype=coordinates.dtype)
    atom_batch = torch.as_tensor(atom_batch, device=coordinates.device, dtype=torch.long)
    graphs = int(atom_batch.max()) + 1 if atom_batch.numel() else 1
    corrections = []
    failures = []
    bond_graph = atom_batch[bonds[0]] if bonds.numel() else atom_batch.new_empty(0)
    for graph in range(graphs):
        atom_ids = torch.nonzero(atom_batch == graph, as_tuple=False).reshape(-1)
        start = int(atom_ids[0]) if atom_ids.numel() else 0
        keep = bond_graph == graph
        local_bonds = bonds[:, keep] - start
        correction, failure = differentiable_bond_projection(
            coordinates[atom_ids], local_bonds, residual[keep], damping=damping
        )
        corrections.append(correction)
        failures.append(failure)
    return torch.cat(corrections, dim=0), torch.stack(failures)
