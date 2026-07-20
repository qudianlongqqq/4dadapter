"""Differentiable per-molecule normal-equation solver for MCVR V8."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DifferentiableSolveConfig:
    lambda_bond: float = 1.0
    lambda_angle: float = 1.0
    lambda_move: float = 0.0
    damping: float = 1.0e-6
    backend: str = "cholesky"
    fail_closed: bool = True

    def validate(self) -> None:
        if self.lambda_bond < 0.0 or self.lambda_angle < 0.0 or self.lambda_move < 0.0:
            raise ValueError("V8 solver lambdas must be nonnegative")
        if self.damping <= 0.0:
            raise ValueError("V8 solver damping must be positive")
        if self.backend not in {"cholesky", "solve"}:
            raise ValueError(f"unsupported differentiable solver backend: {self.backend}")


def _normal_contribution(
    jacobian: Tensor, residual: Tensor, weight: float
) -> tuple[Tensor, Tensor]:
    if not jacobian.numel() or float(weight) == 0.0:
        width = jacobian.size(1)
        return jacobian.new_zeros((width, width)), jacobian.new_zeros(width)
    return float(weight) * (jacobian.transpose(0, 1) @ jacobian), -float(weight) * (
        jacobian.transpose(0, 1) @ residual
    )


def solve_unified_delta(
    delta_prior: Tensor,
    prior_confidence: Tensor,
    bond_residual: Tensor,
    bond_jacobian: Tensor,
    angle_residual: Tensor,
    angle_jacobian: Tensor,
    config: DifferentiableSolveConfig,
) -> tuple[Tensor, dict[str, Tensor | str | bool]]:
    """Solve one graph without an inverse, detach, NumPy, or CPU round-trip."""

    config.validate()
    if delta_prior.ndim != 2 or delta_prior.size(-1) != 3:
        raise ValueError("delta_prior must have shape [N, 3]")
    flat_prior = delta_prior.reshape(-1)
    confidence = torch.as_tensor(
        prior_confidence, device=delta_prior.device, dtype=delta_prior.dtype
    ).reshape(-1)
    if confidence.numel() == delta_prior.size(0):
        confidence = confidence.repeat_interleave(3)
    if confidence.numel() != flat_prior.numel():
        raise ValueError("prior confidence must be per atom or Cartesian component")
    if bool((confidence <= 0.0).any()) or not bool(torch.isfinite(confidence).all()):
        raise ValueError("prior confidence must be finite and positive")
    no_constraints = not bool(bond_jacobian.numel() or angle_jacobian.numel())
    if no_constraints and config.lambda_move == 0.0:
        zero = flat_prior.new_zeros(())
        return delta_prior, {
            "status": "NO_ACTIVE_CONSTRAINT",
            "fallback": False,
            "condition_estimate": flat_prior.new_ones(()),
            "bond_normal_trace": zero,
            "angle_normal_trace": zero,
            "solver_bond_contribution": zero,
            "solver_angle_contribution": zero,
        }

    width = flat_prior.numel()
    diagonal = confidence + float(config.lambda_move)
    normal = torch.diag(diagonal)
    rhs = confidence * flat_prior
    bond_normal, bond_rhs = _normal_contribution(bond_jacobian, bond_residual, config.lambda_bond)
    angle_normal, angle_rhs = _normal_contribution(
        angle_jacobian, angle_residual, config.lambda_angle
    )
    normal = normal + bond_normal + angle_normal
    rhs = rhs + bond_rhs + angle_rhs
    # Damping is a positive numerical stabilizer. The analytically exact inactive
    # case above bypasses it so lambda_move=0 still reproduces the D1 prior.
    normal = normal + float(config.damping) * torch.eye(
        width, device=normal.device, dtype=normal.dtype
    )
    status = "SOLVED"
    fallback = False
    if config.backend == "cholesky":
        factor, info = torch.linalg.cholesky_ex(normal)
        failed = bool((info != 0).any()) or not bool(torch.isfinite(factor).all())
        solution = torch.cholesky_solve(rhs[:, None], factor).reshape(-1) if not failed else rhs
    else:
        try:
            solution = torch.linalg.solve(normal, rhs)
            failed = not bool(torch.isfinite(solution).all())
        except RuntimeError:
            solution = rhs
            failed = True
    if failed or not bool(torch.isfinite(solution).all()):
        status = "SOLVER_FAILURE_FAIL_CLOSED"
        fallback = True
        if not config.fail_closed:
            raise RuntimeError("V8 differentiable constraint solve failed")
        solution = flat_prior
    # eigvalsh is diagnostic-only and does not participate in the loss graph.
    with torch.no_grad():
        eigenvalues = torch.linalg.eigvalsh(normal.detach())
        condition = eigenvalues[-1] / eigenvalues[0].clamp_min(torch.finfo(normal.dtype).tiny)
    diagnostics: dict[str, Tensor | str | bool] = {
        "status": status,
        "fallback": fallback,
        "condition_estimate": condition,
        "bond_normal_trace": torch.trace(bond_normal),
        "angle_normal_trace": torch.trace(angle_normal),
        "solver_bond_contribution": torch.linalg.vector_norm(bond_rhs),
        "solver_angle_contribution": torch.linalg.vector_norm(angle_rhs),
    }
    return solution.reshape_as(delta_prior), diagnostics
