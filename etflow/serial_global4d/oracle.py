"""Damped residual projection and benefit-aware gate calibration targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from etflow.commons.global_coupled_4d_jacobian import decompose_joint_rates
from etflow.commons.global_coupled_4d_projection import ProjectionResult, gram_solve


@dataclass
class SerialResidualOracle:
    q_res_star: Tensor
    r_j_star: Tensor
    residual: Tensor
    residual_norm: Tensor
    projected_residual_norm: Tensor
    projection_energy_ratio: Tensor
    oracle_residual_error: Tensor
    stretch_energy: Tensor
    bending_energy: Tensor
    torsion_energy: Tensor
    projection: ProjectionResult


def solve_serial_residual_oracle(
    jacobian: Tensor,
    residual: Tensor,
    axes: Tensor,
    *,
    ridge: float = 1.0e-5,
    weights: Tensor | None = None,
    rank_tol: float = 1.0e-6,
) -> SerialResidualOracle:
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    projection = gram_solve(
        jacobian,
        residual,
        weights=weights,
        damping=float(ridge),
        rank_tol=float(rank_tol),
    )
    q = projection.coefficients.reshape(-1, 4)
    if axes.shape != (q.size(0), 3):
        raise ValueError("axes must have shape [num_joints, 3]")
    parts = decompose_joint_rates(q, axes) if q.numel() else {
        "stretch": residual.new_empty((0,)),
        "bending_vector": residual.new_empty((0, 3)),
        "torsion_vector": residual.new_empty((0, 3)),
    }
    if q.numel():
        zeros = torch.zeros_like(q[:, 1:])
        q_stretch = torch.cat((parts["stretch"][:, None], zeros), dim=-1)
        q_bending = torch.cat(
            (torch.zeros_like(parts["stretch"][:, None]), parts["bending_vector"]),
            dim=-1,
        )
        q_torsion = torch.cat(
            (torch.zeros_like(parts["stretch"][:, None]), parts["torsion_vector"]),
            dim=-1,
        )
        stretch = (jacobian @ q_stretch.reshape(-1)).square().sum()
        bending = (jacobian @ q_bending.reshape(-1)).square().sum()
        torsion = (jacobian @ q_torsion.reshape(-1)).square().sum()
    else:
        stretch = bending = torsion = residual.new_zeros(())
    residual_energy = residual.square().sum()
    projected_energy = projection.projected.square().sum()
    return SerialResidualOracle(
        q_res_star=q,
        r_j_star=projection.projected,
        residual=residual,
        residual_norm=torch.linalg.vector_norm(residual),
        projected_residual_norm=torch.linalg.vector_norm(projection.projected),
        projection_energy_ratio=projected_energy / residual_energy.clamp_min(1.0e-20),
        oracle_residual_error=torch.linalg.vector_norm(residual - projection.projected),
        stretch_energy=stretch,
        bending_energy=bending,
        torsion_energy=torsion,
        projection=projection,
    )


def benefit_aware_gate_target(
    residual: Tensor,
    prediction: Tensor,
    *,
    beta: float = 1.0,
    epsilon: float = 1.0e-12,
    gain_threshold: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return analytic gate, gain, and beneficial target for one or many graphs.

    Leading dimensions are treated independently; the last two dimensions are
    flattened as Cartesian atoms/components.
    """

    if beta <= 0:
        raise ValueError("beta must be positive")
    if residual.shape != prediction.shape:
        raise ValueError("residual and prediction shapes must match")
    if residual.ndim < 2:
        raise ValueError("Cartesian residuals must have at least two dimensions")
    if residual.ndim == 2:
        residual = residual.unsqueeze(0)
        prediction = prediction.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    flat_r = residual.flatten(start_dim=1)
    flat_z = prediction.flatten(start_dim=1)
    dot = (flat_r * flat_z).sum(-1)
    z_energy = flat_z.square().sum(-1)
    gate = (dot / (float(beta) * z_energy + float(epsilon))).clamp(0.0, 1.0)
    corrected = flat_r - float(beta) * gate[:, None] * flat_z
    gain = flat_r.square().sum(-1) - corrected.square().sum(-1)
    gate = torch.where(gain > float(gain_threshold), gate, torch.zeros_like(gate))
    beneficial = gain > float(gain_threshold)
    if squeeze:
        return gate.squeeze(0), gain.squeeze(0), beneficial.squeeze(0)
    return gate, gain, beneficial
