"""Materialize deterministic structured residual targets for Stage 2."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from etflow.commons.global_coupled_4d_jacobian import (
    build_global_coupled_4d_jacobian,
    decompose_joint_rates,
    joint_geometry,
)
from etflow.commons.global_coupled_4d_topology import GlobalCoupled4DTopologyCache
from etflow.serial_global4d.oracle import solve_serial_residual_oracle


def materialize_stage2_targets(
    record: Mapping[str, Any],
    *,
    target_time: float,
    ridge: float = 1.0e-5,
    rank_tol: float = 1.0e-6,
    topology_cache: GlobalCoupled4DTopologyCache | None = None,
) -> dict[str, Any]:
    if not 0.0 <= float(target_time) <= 1.0:
        raise ValueError("target_time must be in [0, 1]")
    x_cart = torch.as_tensor(record["x_cart"], dtype=torch.float32)
    x_ref = torch.as_tensor(record["x_ref_aligned"], dtype=torch.float32)
    residual = x_ref - x_cart
    edge_index = torch.as_tensor(record["edge_index"], dtype=torch.long)
    rotatable = torch.as_tensor(record["rotatable_bond_index"], dtype=torch.long)
    cache = topology_cache or GlobalCoupled4DTopologyCache()
    prepared = cache.get_prepared(x_cart.size(0), edge_index, rotatable)
    x_t = (1.0 - float(target_time)) * x_cart + float(target_time) * x_ref
    topology = prepared.topology
    jacobian, _ = build_global_coupled_4d_jacobian(
        x_t, topology, flat_index=prepared.jacobian_flat_index
    )
    axes = joint_geometry(x_t, topology).axis
    oracle = solve_serial_residual_oracle(
        jacobian, residual, axes, ridge=ridge, rank_tol=rank_tol
    )
    q = oracle.q_res_star
    if q.numel():
        parts = decompose_joint_rates(q, axes)
        zero_s = torch.zeros_like(parts["stretch"][:, None])
        zero_v = torch.zeros_like(parts["omega"])
        q_stretch = torch.cat((parts["stretch"][:, None], zero_v), dim=-1)
        q_bending = torch.cat((zero_s, parts["bending_vector"]), dim=-1)
        q_torsion = torch.cat((zero_s, parts["torsion_vector"]), dim=-1)
        stretch = (jacobian @ q_stretch.reshape(-1)).reshape_as(x_cart)
        bending = (jacobian @ q_bending.reshape(-1)).reshape_as(x_cart)
        torsion = (jacobian @ q_torsion.reshape(-1)).reshape_as(x_cart)
    else:
        stretch = bending = torsion = torch.zeros_like(x_cart)
    reconstructed = (
        (jacobian @ q.reshape(-1)).reshape_as(x_cart)
        if q.numel()
        else torch.zeros_like(x_cart)
    )
    error = torch.linalg.vector_norm(reconstructed - oracle.r_j_star)
    return {
        "target_time": float(target_time),
        "q_target_mode": "damped_global4d_residual",
        "ridge": float(ridge),
        "rank_tol": float(rank_tol),
        "jacobian_schema": "global-coupled-4d-v1",
        "q_res_star": q.cpu(),
        "r_J_star": oracle.r_j_star.cpu(),
        "residual_norm": float(oracle.residual_norm),
        "projected_residual_norm": float(oracle.projected_residual_norm),
        "projection_energy_ratio": float(oracle.projection_energy_ratio),
        "stretch_target": stretch.cpu(),
        "bending_target": bending.cpu(),
        "torsion_target": torsion.cpu(),
        "jacobian_rank": int(oracle.projection.effective_rank),
        "condition_number": float(oracle.projection.condition_number),
        "solver_mode": str(oracle.projection.solver_backend),
        "solver_fallback": bool(oracle.projection.solver_fallback_count),
        "target_reconstruction_error": float(error),
    }
