"""Offline Jacobian guidance for a frozen Cartesian BAC proposal."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Mapping

import torch
from torch import Tensor

from .bac_jacobian import (
    ConstraintSystem,
    JacobianBACConfig,
    build_constraint_system,
    remove_rigid_update,
    solve_damped_system,
)
from .bac_safety import BACSafetyConfig, evaluate_bac_proposal


GUIDED_SCHEMA_VERSION = "mcvr-v4-jacobian-guided-v1"


def _config(
    config: JacobianBACConfig | Mapping[str, Any] | None,
) -> JacobianBACConfig:
    if isinstance(config, JacobianBACConfig):
        result = config
    else:
        result = JacobianBACConfig.from_mapping(config)
    result.validate()
    return result


def _safety(config: JacobianBACConfig, safety: BACSafetyConfig | None) -> BACSafetyConfig:
    return safety or BACSafetyConfig(
        max_atom_displacement=config.max_atom_displacement,
        max_molecule_rms_displacement=config.max_molecule_rms_displacement,
    )


def _system(
    coordinates: Tensor,
    bonds: Tensor,
    bond_ranges: Tensor,
    angles: Tensor,
    angle_ranges: Tensor,
    config: JacobianBACConfig,
) -> ConstraintSystem:
    return build_constraint_system(coordinates, bonds, bond_ranges, angles, angle_ranges, config)


def _movement(update: Tensor) -> dict[str, float]:
    if not update.numel():
        return {"graph_rms": 0.0, "atom_max": 0.0}
    return {
        "graph_rms": float(torch.sqrt(update.square().sum(-1).mean())),
        "atom_max": float(torch.linalg.vector_norm(update, dim=-1).max()),
    }


def _trust_scale(update: Tensor, config: JacobianBACConfig) -> tuple[Tensor, float]:
    movement = _movement(update)
    scale = min(
        1.0,
        config.max_molecule_rms_displacement / max(movement["graph_rms"], 1.0e-30),
        config.max_atom_displacement / max(movement["atom_max"], 1.0e-30),
    )
    return update * scale, scale


def _within_trust(update: Tensor, config: JacobianBACConfig) -> bool:
    movement = _movement(update)
    return bool(
        movement["graph_rms"] <= config.max_molecule_rms_displacement + 1.0e-12
        and movement["atom_max"] <= config.max_atom_displacement + 1.0e-12
    )


def _fallback(
    *,
    candidate: str,
    source: Tensor,
    coordinates: Tensor,
    accepted: bool,
    status: str,
    started: float,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": GUIDED_SCHEMA_VERSION,
        "candidate": candidate,
        "coordinates": coordinates.to(dtype=source.dtype),
        "accepted": bool(accepted),
        "rolled_back": True,
        "status": status,
        "runtime_seconds": time.perf_counter() - started,
        "diagnostics": dict(diagnostics),
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "validation_only": True,
    }


def _success(
    *,
    candidate: str,
    source: Tensor,
    coordinates: Tensor,
    status: str,
    started: float,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": GUIDED_SCHEMA_VERSION,
        "candidate": candidate,
        "coordinates": coordinates.to(dtype=source.dtype),
        "accepted": True,
        "rolled_back": False,
        "status": status,
        "runtime_seconds": time.perf_counter() - started,
        "diagnostics": dict(diagnostics),
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "validation_only": True,
    }


def _prepare(source: Tensor, d1_coordinates: Tensor) -> tuple[Tensor, Tensor, str | None]:
    source64 = torch.as_tensor(source).detach().to(torch.float64).cpu()
    d1_64 = torch.as_tensor(d1_coordinates).detach().to(torch.float64).cpu()
    if source64.shape != d1_64.shape:
        return source64, source64.clone(), "IDENTITY_SHAPE_CHANGED"
    if not bool(torch.isfinite(source64).all() and torch.isfinite(d1_64).all()):
        return source64, source64.clone(), "NONFINITE_INPUT"
    return source64, d1_64, None


def posthoc_jacobian_correction(
    source: Tensor,
    d1_coordinates: Tensor,
    record: Any,
    validity: Any,
    *,
    bonds: Tensor,
    bond_ranges: Tensor,
    angles: Tensor,
    angle_ranges: Tensor,
    alpha: float,
    d1_accepted: bool,
    atomic_numbers: Tensor | None = None,
    config: JacobianBACConfig | Mapping[str, Any] | None = None,
    safety_config: BACSafetyConfig | None = None,
) -> dict[str, Any]:
    """Candidate A: add one bounded DLS correction to the frozen D1 result."""

    started = time.perf_counter()
    original = torch.as_tensor(source)
    jacobian_config = _config(config)
    if float(alpha) not in {0.25, 0.5, 1.0}:
        raise ValueError("Candidate A alpha must be one of 0.25, 0.5, or 1.0")
    source64, d1_64, input_error = _prepare(source, d1_coordinates)
    base = {"alpha": float(alpha), "config": asdict(jacobian_config)}
    if input_error:
        return _fallback(
            candidate=f"A{int(alpha * 100):03d}",
            source=original,
            coordinates=source64,
            accepted=False,
            status=input_error,
            started=started,
            diagnostics=base,
        )
    system = _system(d1_64, bonds, bond_ranges, angles, angle_ranges, jacobian_config)
    objective_before = float(system.objective)
    flat, solve = solve_damped_system(
        system,
        d1_64.size(0),
        jacobian_config,
        atomic_numbers=atomic_numbers,
    )
    diagnostics: dict[str, Any] = {
        **base,
        "objective_before": objective_before,
        "constraint_counts": system.counts,
        **system.diagnostics,
        **solve,
    }
    candidate_name = f"A{int(alpha * 100):03d}"
    if solve["solver_status"] != "SOLVED":
        return _fallback(
            candidate=candidate_name,
            source=original,
            coordinates=d1_64,
            accepted=d1_accepted,
            status=solve["solver_status"],
            started=started,
            diagnostics=diagnostics,
        )
    correction, rigid = remove_rigid_update(d1_64, flat.reshape_as(d1_64))
    correction, correction_scale = _trust_scale(correction, jacobian_config)
    combined_delta = d1_64 - source64 + float(alpha) * correction
    diagnostics.update(
        {
            **rigid,
            "correction_trust_scale": correction_scale,
            "correction_movement": _movement(correction),
            "combined_movement": _movement(combined_delta),
        }
    )
    if not _within_trust(combined_delta, jacobian_config):
        return _fallback(
            candidate=candidate_name,
            source=original,
            coordinates=d1_64,
            accepted=d1_accepted,
            status="COMBINED_TRUST_REJECTED",
            started=started,
            diagnostics=diagnostics,
        )
    proposal = source64 + combined_delta
    candidate_system = _system(proposal, bonds, bond_ranges, angles, angle_ranges, jacobian_config)
    objective_after = float(candidate_system.objective)
    decision = evaluate_bac_proposal(
        source64.to(torch.float32),
        proposal.to(torch.float32),
        record,
        validity,
        _safety(jacobian_config, safety_config),
    )
    diagnostics.update(
        {
            "objective_after": objective_after,
            "objective_reduction": objective_before - objective_after,
            "hard_safety_accepted": bool(decision["accepted"]),
            "hard_safety_reasons": list(decision["reasons"]),
        }
    )
    if objective_after >= objective_before - 1.0e-15:
        return _fallback(
            candidate=candidate_name,
            source=original,
            coordinates=d1_64,
            accepted=d1_accepted,
            status="OBJECTIVE_NOT_REDUCED",
            started=started,
            diagnostics=diagnostics,
        )
    if not decision["accepted"]:
        return _fallback(
            candidate=candidate_name,
            source=original,
            coordinates=d1_64,
            accepted=d1_accepted,
            status="HARD_SAFETY_REJECTED",
            started=started,
            diagnostics=diagnostics,
        )
    return _success(
        candidate=candidate_name,
        source=original,
        coordinates=proposal,
        status="CORRECTION_ACCEPTED",
        started=started,
        diagnostics=diagnostics,
    )


def _project_row_space(
    system: ConstraintSystem,
    delta: Tensor,
    config: JacobianBACConfig,
) -> tuple[Tensor, dict[str, Any]]:
    zero = delta.new_zeros(delta.numel())
    base = {
        "projection_status": "NO_ACTIVE_CONSTRAINT",
        "effective_rank": 0,
        "singular_value_max": 0.0,
        "singular_value_min_retained": 0.0,
        "condition_number": 0.0,
        "truncated_direction_count": 0,
    }
    if not system.residual.numel():
        return zero, base
    if not all(
        bool(torch.isfinite(value).all())
        for value in (system.residual, system.jacobian, system.weights, delta)
    ):
        return zero, {**base, "projection_status": "NONFINITE_SYSTEM"}
    weighted = torch.sqrt(system.weights)[:, None] * system.jacobian
    try:
        _, singular, right_h = torch.linalg.svd(weighted, full_matrices=False)
    except RuntimeError:
        return zero, {**base, "projection_status": "SVD_FAILED"}
    sigma_max = float(singular.max()) if singular.numel() else 0.0
    retain = singular >= config.rank_tol * max(sigma_max, 1.0e-30)
    rank = int(retain.sum())
    if rank == 0:
        return zero, {**base, "projection_status": "EFFECTIVE_RANK_ZERO"}
    sigma_min = float(singular[retain].min())
    factors = torch.where(
        retain,
        singular.square() / (singular.square() + config.damping_lambda),
        torch.zeros_like(singular),
    )
    coefficients = right_h @ delta.reshape(-1)
    projection = right_h.mT @ (factors * coefficients)
    if not bool(torch.isfinite(projection).all()):
        return zero, {**base, "projection_status": "NONFINITE_PROJECTION"}
    return projection, {
        **base,
        "projection_status": "PROJECTED",
        "effective_rank": rank,
        "singular_value_max": sigma_max,
        "singular_value_min_retained": sigma_min,
        "condition_number": sigma_max / max(sigma_min, 1.0e-30),
        "truncated_direction_count": int(singular.numel() - rank),
    }


def _masked_system(system: ConstraintSystem, mask: Tensor) -> ConstraintSystem:
    indices = torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()
    return ConstraintSystem(
        residual=system.residual[mask],
        jacobian=system.jacobian[mask],
        weights=system.weights[mask],
        constraint_types=tuple(system.constraint_types[index] for index in indices),
        counts=dict(system.counts),
        diagnostics=dict(system.diagnostics),
    )


def jacobian_projection(
    source: Tensor,
    d1_coordinates: Tensor,
    record: Any,
    validity: Any,
    *,
    bonds: Tensor,
    bond_ranges: Tensor,
    angles: Tensor,
    angle_ranges: Tensor,
    d1_accepted: bool,
    config: JacobianBACConfig | Mapping[str, Any] | None = None,
    safety_config: BACSafetyConfig | None = None,
) -> dict[str, Any]:
    """Candidate B: suppress D1 row-space components predicted to worsen BAC."""

    started = time.perf_counter()
    original = torch.as_tensor(source)
    jacobian_config = _config(config)
    source64, d1_64, input_error = _prepare(source, d1_coordinates)
    base = {"config": asdict(jacobian_config)}
    if input_error:
        return _fallback(
            candidate="B",
            source=original,
            coordinates=source64,
            accepted=False,
            status=input_error,
            started=started,
            diagnostics=base,
        )
    delta_cart = d1_64 - source64
    source_system = _system(source64, bonds, bond_ranges, angles, angle_ranges, jacobian_config)
    full_projection, full_diag = _project_row_space(source_system, delta_cart, jacobian_config)
    predicted_change = source_system.jacobian @ delta_cart.reshape(-1)
    violating = source_system.residual * predicted_change > 0.0
    selected_system = _masked_system(source_system, violating)
    harmful_projection, selected_diag = _project_row_space(
        selected_system, delta_cart, jacobian_config
    )
    correction, rigid = remove_rigid_update(source64, harmful_projection.reshape_as(source64))
    correction, correction_scale = _trust_scale(correction, jacobian_config)
    proposal_delta = delta_cart - correction
    d1_system = _system(d1_64, bonds, bond_ranges, angles, angle_ranges, jacobian_config)
    diagnostics: dict[str, Any] = {
        **base,
        "source_objective": float(source_system.objective),
        "d1_objective": float(d1_system.objective),
        "violating_row_count": int(violating.sum()),
        "full_projection_norm": float(torch.linalg.vector_norm(full_projection)),
        "parallel_norm": float(torch.linalg.vector_norm(full_projection)),
        "perpendicular_norm": float(
            torch.linalg.vector_norm(delta_cart.reshape(-1) - full_projection)
        ),
        "selected_projection_norm": float(torch.linalg.vector_norm(harmful_projection)),
        "full_projection": full_diag,
        "selected_projection": selected_diag,
        **rigid,
        "correction_trust_scale": correction_scale,
        "correction_movement": _movement(correction),
        "combined_movement": _movement(proposal_delta),
    }
    if selected_diag["projection_status"] != "PROJECTED":
        return _fallback(
            candidate="B",
            source=original,
            coordinates=d1_64,
            accepted=d1_accepted,
            status=selected_diag["projection_status"],
            started=started,
            diagnostics=diagnostics,
        )
    if not _within_trust(proposal_delta, jacobian_config):
        return _fallback(
            candidate="B",
            source=original,
            coordinates=d1_64,
            accepted=d1_accepted,
            status="COMBINED_TRUST_REJECTED",
            started=started,
            diagnostics=diagnostics,
        )
    proposal = source64 + proposal_delta
    candidate_system = _system(proposal, bonds, bond_ranges, angles, angle_ranges, jacobian_config)
    objective_after = float(candidate_system.objective)
    decision = evaluate_bac_proposal(
        source64.to(torch.float32),
        proposal.to(torch.float32),
        record,
        validity,
        _safety(jacobian_config, safety_config),
    )
    diagnostics.update(
        {
            "objective_after": objective_after,
            "hard_safety_accepted": bool(decision["accepted"]),
            "hard_safety_reasons": list(decision["reasons"]),
        }
    )
    if objective_after > float(d1_system.objective) + 1.0e-15:
        return _fallback(
            candidate="B",
            source=original,
            coordinates=d1_64,
            accepted=d1_accepted,
            status="OBJECTIVE_WORSENED",
            started=started,
            diagnostics=diagnostics,
        )
    if not decision["accepted"]:
        return _fallback(
            candidate="B",
            source=original,
            coordinates=d1_64,
            accepted=d1_accepted,
            status="HARD_SAFETY_REJECTED",
            started=started,
            diagnostics=diagnostics,
        )
    return _success(
        candidate="B",
        source=original,
        coordinates=proposal,
        status="PROJECTION_ACCEPTED",
        started=started,
        diagnostics=diagnostics,
    )


def trust_region_hybrid(
    source: Tensor,
    d1_coordinates: Tensor,
    record: Any,
    validity: Any,
    *,
    bonds: Tensor,
    bond_ranges: Tensor,
    angles: Tensor,
    angle_ranges: Tensor,
    config: JacobianBACConfig | Mapping[str, Any] | None = None,
    safety_config: BACSafetyConfig | None = None,
    line_search_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125),
) -> dict[str, Any]:
    """Candidate C: use the Jacobian objective to judge the D1 direction."""

    started = time.perf_counter()
    original = torch.as_tensor(source)
    if tuple(line_search_scales) != (1.0, 0.5, 0.25, 0.125):
        raise ValueError("Candidate C line-search schedule is frozen")
    jacobian_config = _config(config)
    source64, d1_64, input_error = _prepare(source, d1_coordinates)
    base = {"config": asdict(jacobian_config), "line_search_scales": list(line_search_scales)}
    if input_error:
        return _fallback(
            candidate="C",
            source=original,
            coordinates=source64,
            accepted=False,
            status=input_error,
            started=started,
            diagnostics=base,
        )
    delta_cart = d1_64 - source64
    source_system = _system(source64, bonds, bond_ranges, angles, angle_ranges, jacobian_config)
    objective_before = float(source_system.objective)
    attempts = []
    safety = _safety(jacobian_config, safety_config)
    for scale in line_search_scales:
        proposal_delta = float(scale) * delta_cart
        trust_ok = _within_trust(proposal_delta, jacobian_config)
        proposal = source64 + proposal_delta
        candidate_system = _system(
            proposal, bonds, bond_ranges, angles, angle_ranges, jacobian_config
        )
        objective_after = float(candidate_system.objective)
        objective_decreased = objective_after < objective_before - 1.0e-15
        decision = evaluate_bac_proposal(
            source64.to(torch.float32),
            proposal.to(torch.float32),
            record,
            validity,
            safety,
        )
        accepted = bool(trust_ok and objective_decreased and decision["accepted"])
        attempts.append(
            {
                "scale": float(scale),
                "objective_after": objective_after,
                "objective_decreased": objective_decreased,
                "within_trust": trust_ok,
                "hard_safety_accepted": bool(decision["accepted"]),
                "hard_safety_reasons": list(decision["reasons"]),
                "accepted": accepted,
            }
        )
        if accepted:
            return _success(
                candidate="C",
                source=original,
                coordinates=proposal,
                status="D1_DIRECTION_ACCEPTED",
                started=started,
                diagnostics={
                    **base,
                    "source_objective": objective_before,
                    "selected_scale": float(scale),
                    "attempts": attempts,
                    "combined_movement": _movement(proposal_delta),
                },
            )
    return _fallback(
        candidate="C",
        source=original,
        coordinates=source64,
        accepted=False,
        status="LINE_SEARCH_REJECTED",
        started=started,
        diagnostics={
            **base,
            "source_objective": objective_before,
            "selected_scale": 0.0,
            "attempts": attempts,
            "combined_movement": _movement(torch.zeros_like(delta_cart)),
        },
    )
