"""Rank-aware exact projection and damped global torsion targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class KinematicProjection:
    basis: Tensor
    rank: int
    singular_values: Tensor
    u_kin_star: Tensor
    u_res_star: Tensor
    rate_star_exact: Tensor
    rate_star_damped: Tensor


def compute_column_basis(jacobian: Tensor, rank_tol: float = 1.0e-6):
    """Return an exact SVD column-space basis with a relative rank threshold."""

    if jacobian.ndim != 2:
        raise ValueError("jacobian must be a matrix")
    if jacobian.size(1) == 0:
        return jacobian[:, :0], 0, jacobian.new_empty((0,))
    original_dtype = jacobian.dtype
    solve = jacobian.float() if jacobian.dtype in (torch.float16, torch.bfloat16) else jacobian
    u, singular_values, _ = torch.linalg.svd(solve, full_matrices=False)
    threshold = float(rank_tol) * singular_values.max().clamp_min(1.0)
    rank = int((singular_values > threshold).sum().item())
    return u[:, :rank].to(original_dtype), rank, singular_values.to(original_dtype)


def exact_project(vector: Tensor, basis: Tensor) -> Tensor:
    flat = vector.reshape(-1)
    return (basis @ (basis.transpose(0, 1) @ flat)).reshape_as(vector) if basis.size(1) else torch.zeros_like(vector)


def damped_global_rate_target(jacobian: Tensor, target: Tensor, ridge: float = 1.0e-4) -> Tensor:
    """Stable global joint-rate target; this is not an orthogonal projector."""

    joints = jacobian.size(1)
    if joints == 0:
        return target.new_empty((0,))
    original_dtype=jacobian.dtype
    solve_j=jacobian.float() if original_dtype in (torch.float16,torch.bfloat16) else jacobian
    solve_target=target.reshape(-1).to(solve_j.dtype)
    gram = solve_j.transpose(0, 1) @ solve_j
    rhs = solve_j.transpose(0, 1) @ solve_target
    identity = torch.eye(joints, device=jacobian.device, dtype=solve_j.dtype)
    return torch.linalg.solve(gram + float(ridge) * identity, rhs).to(original_dtype)


def decompose_target(
    jacobian: Tensor,
    target: Tensor,
    *,
    rank_tol: float = 1.0e-6,
    rate_target_ridge: float = 1.0e-4,
) -> KinematicProjection:
    basis, rank, singular_values = compute_column_basis(jacobian, rank_tol)
    u_kin = exact_project(target, basis)
    u_res = target - u_kin
    pinv_j=jacobian.float() if jacobian.dtype in (torch.float16,torch.bfloat16) else jacobian
    exact_rate = (
        (torch.linalg.pinv(pinv_j, rtol=rank_tol) @ target.reshape(-1).to(pinv_j.dtype)).to(jacobian.dtype)
        if jacobian.size(1) else target.new_empty((0,))
    )
    damped = damped_global_rate_target(jacobian, target, rate_target_ridge)
    return KinematicProjection(
        basis, rank, singular_values, u_kin, u_res, exact_rate, damped
    )


def soft_gate_target(
    rate_star: Tensor,
    *,
    threshold: float = 0.05,
    temperature: float = 0.02,
    method: str = "sigmoid_threshold",
) -> Tensor:
    if temperature <= 0:
        raise ValueError("gate temperature must be positive")
    if method == "sigmoid_threshold":
        return torch.sigmoid((rate_star.abs() - float(threshold)) / float(temperature))
    if method == "ratio":
        return rate_star.abs() / (rate_star.abs() + float(temperature))
    raise ValueError(f"Unknown soft gate target method: {method}")
