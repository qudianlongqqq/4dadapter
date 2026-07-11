"""Weighted global least-squares solvers for the coupled 4D Jacobian."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ProjectionResult:
    coefficients: Tensor
    projected: Tensor
    residual: Tensor
    singular_values: Tensor
    effective_rank: int
    condition_number: float
    explained_ratio: Tensor
    reconstruction_error: Tensor
    orthogonality_error: Tensor
    solver_backend: str
    solver_fallback_count: int


def _flat_weights(vector: Tensor, weights: Tensor | None) -> Tensor:
    if weights is None:
        return vector.new_ones(vector.numel())
    value = torch.as_tensor(weights, device=vector.device, dtype=vector.dtype).reshape(-1)
    if value.numel() == vector.shape[0] and vector.ndim == 2:
        value = value.repeat_interleave(vector.shape[1])
    if value.numel() != vector.numel():
        raise ValueError("weights must contain N atom weights or 3N component weights")
    if bool((value < 0).any()) or not bool(torch.isfinite(value).all()):
        raise ValueError("weights must be finite and non-negative")
    return value


def _diagnostics(
    jacobian: Tensor,
    vector: Tensor,
    coefficients: Tensor,
    weights: Tensor,
    singular_values: Tensor,
    rank: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, float]:
    flat = vector.reshape(-1)
    projected_flat = jacobian @ coefficients
    residual_flat = flat - projected_flat
    weighted_energy = (weights * flat.square()).sum()
    residual_energy = (weights * residual_flat.square()).sum()
    explained = 1.0 - residual_energy / weighted_energy.clamp_min(1.0e-20)
    reconstruction = residual_energy.sqrt() / weighted_energy.sqrt().clamp_min(1.0e-20)
    normal_residual = jacobian.transpose(0, 1) @ (weights * residual_flat)
    scale = (
        torch.linalg.norm(jacobian * weights.sqrt()[:, None])
        * torch.linalg.norm(weights.sqrt() * residual_flat)
    ).clamp_min(1.0e-20)
    orthogonality = torch.linalg.norm(normal_residual) / scale
    if rank == 0:
        condition = 0.0
    else:
        condition = float((singular_values[0] / singular_values[rank - 1]).detach())
    return (
        projected_flat.reshape_as(vector),
        residual_flat.reshape_as(vector),
        explained,
        reconstruction,
        condition,
    ), orthogonality


def _rank(singular_values: Tensor, rank_tol: float) -> int:
    if singular_values.numel() == 0:
        return 0
    threshold = float(rank_tol) * singular_values.max()
    return int((singular_values > threshold).sum().item())


def svd_oracle(
    jacobian: Tensor,
    vector: Tensor,
    *,
    weights: Tensor | None = None,
    rank_tol: float = 1.0e-6,
) -> ProjectionResult:
    """Undamped minimum-norm weighted pseudoinverse projection."""

    if jacobian.ndim != 2 or jacobian.size(0) != vector.numel():
        raise ValueError("jacobian must have shape [vector.numel(), K]")
    flat_weights = _flat_weights(vector, weights)
    if jacobian.size(1) == 0:
        zero = vector.new_empty((0,))
        return ProjectionResult(
            zero,
            torch.zeros_like(vector),
            vector.clone(),
            zero,
            0,
            0.0,
            vector.new_zeros(()),
            vector.new_ones(()) if vector.numel() else vector.new_zeros(()),
            vector.new_zeros(()),
            "svd_oracle",
            0,
        )
    solve_dtype = torch.float64 if jacobian.dtype == torch.float64 else torch.float32
    weighted_j = flat_weights.sqrt()[:, None] * jacobian
    weighted_u = flat_weights.sqrt() * vector.reshape(-1)
    u, singular_values, vh = torch.linalg.svd(
        weighted_j.to(solve_dtype), full_matrices=False
    )
    rank = _rank(singular_values, rank_tol)
    if rank:
        coefficients = vh[:rank].transpose(0, 1) @ (
            (u[:, :rank].transpose(0, 1) @ weighted_u.to(solve_dtype))
            / singular_values[:rank]
        )
    else:
        coefficients = weighted_j.new_zeros((jacobian.size(1),), dtype=solve_dtype)
    coefficients = coefficients.to(jacobian.dtype)
    values, orthogonality = _diagnostics(
        jacobian, vector, coefficients, flat_weights, singular_values.to(jacobian.dtype), rank
    )
    projected, residual, explained, reconstruction, condition = values
    return ProjectionResult(
        coefficients,
        projected,
        residual,
        singular_values.to(jacobian.dtype),
        rank,
        condition,
        explained,
        reconstruction,
        orthogonality,
        "svd_oracle",
        0,
    )


def gram_solve(
    jacobian: Tensor,
    vector: Tensor,
    *,
    weights: Tensor | None = None,
    damping: float = 0.0,
    rank_tol: float = 1.0e-6,
) -> ProjectionResult:
    """Solve the complete Gram system with robust ordered fallbacks.

    Backend order is Cholesky, dense solve, least squares, then SVD.  A zero
    damping value gives an exact orthogonal projection whenever the fallback
    solver can resolve rank deficiency; positive damping is available for
    coefficient targets but is not used for strict residual orthogonalization.
    """

    if jacobian.ndim != 2 or jacobian.size(0) != vector.numel():
        raise ValueError("jacobian must have shape [vector.numel(), K]")
    if damping < 0:
        raise ValueError("damping must be non-negative")
    if jacobian.size(1) == 0:
        result = svd_oracle(jacobian, vector, weights=weights, rank_tol=rank_tol)
        result.solver_backend = "empty"
        return result
    flat_weights = _flat_weights(vector, weights)
    weighted_j = flat_weights[:, None] * jacobian
    gram = jacobian.transpose(0, 1) @ weighted_j
    rhs = jacobian.transpose(0, 1) @ (flat_weights * vector.reshape(-1))
    singular_values = torch.linalg.svdvals(flat_weights.sqrt()[:, None] * jacobian)
    rank = _rank(singular_values, rank_tol)
    if damping:
        gram = gram + float(damping) * torch.eye(
            gram.size(0), device=gram.device, dtype=gram.dtype
        )

    if damping == 0.0 and rank < jacobian.size(1):
        oracle = svd_oracle(jacobian, vector, weights=weights, rank_tol=rank_tol)
        oracle.solver_backend = "svd_fallback"
        oracle.solver_fallback_count = 3
        return oracle

    backend = "cholesky"
    fallback_count = 0
    chol, info = torch.linalg.cholesky_ex(gram)
    if int(info.max().detach()) == 0 and bool(torch.isfinite(chol).all()):
        coefficients = torch.cholesky_solve(rhs[:, None], chol).squeeze(-1)
    else:
        fallback_count = 1
        # Solving a singular Gram system can return a finite but basis-dependent
        # coefficient vector.  For the undamped orthogonal projector, use the
        # original weighted J SVD so the projected Cartesian vector remains
        # unique and rotation equivariant.
        backend = "solve"
        coefficients, info = torch.linalg.solve_ex(gram, rhs)
        if int(info.max().detach()) != 0 or not bool(torch.isfinite(coefficients).all()):
            fallback_count = 2
            backend = "lstsq"
            try:
                coefficients = torch.linalg.lstsq(gram, rhs[:, None]).solution.squeeze(-1)
            except RuntimeError:
                coefficients = torch.full_like(rhs, float("nan"))
        if not bool(torch.isfinite(coefficients).all()):
            fallback_count = 3
            oracle = svd_oracle(jacobian, vector, weights=weights, rank_tol=rank_tol)
            oracle.solver_backend = "svd_fallback"
            oracle.solver_fallback_count = fallback_count
            return oracle

    values, orthogonality = _diagnostics(
        jacobian, vector, coefficients, flat_weights, singular_values, rank
    )
    projected, residual, explained, reconstruction, condition = values
    return ProjectionResult(
        coefficients,
        projected,
        residual,
        singular_values,
        rank,
        condition,
        explained,
        reconstruction,
        orthogonality,
        backend,
        fallback_count,
    )


def project_orthogonal_residual(
    jacobian: Tensor,
    vector: Tensor,
    *,
    weights: Tensor | None = None,
    rank_tol: float = 1.0e-6,
) -> ProjectionResult:
    """Project a raw Cartesian vector onto the full 4D orthogonal complement."""

    return gram_solve(
        jacobian, vector, weights=weights, damping=0.0, rank_tol=rank_tol
    )
