"""Independent analytic Jacobian solver for local BAC constraints."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from .bac_constraints import sparse_clash_edges
from .bac_safety import BACSafetyConfig, evaluate_bac_proposal
from .geometry import bond_angles, bond_lengths


JACOBIAN_SCHEMA_VERSION = "mcvr-v3-bac-jacobian-v1"


@dataclass(frozen=True)
class JacobianBACConfig:
    bond_weight: float = 1.0
    angle_weight: float = 1.0
    clash_weight: float = 1.0
    damping_lambda: float = 1.0e-3
    rank_tol: float = 1.0e-6
    max_condition_number: float = 1.0e8
    minimum_distance: float = 1.0e-8
    near_linear_sine_threshold: float = 1.0e-3
    near_linear_weight: float = 0.1
    heavy_atom_mobility: float = 1.0
    hydrogen_mobility: float = 1.0
    max_relinearizations: int = 2
    backtracking_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    max_molecule_rms_displacement: float = 0.06
    max_atom_displacement: float = 0.12
    clash_cutoff: float = 2.0
    clash_allowed_contact: float = 1.0
    clash_exclude_topology_distance: int = 2
    max_clash_edges: int = 128

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None
    ) -> "JacobianBACConfig":
        if values is None:
            return cls()
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown Jacobian BAC settings: {sorted(unknown)}")
        payload = dict(values)
        if "backtracking_scales" in payload:
            payload["backtracking_scales"] = tuple(payload["backtracking_scales"])
        result = cls(**payload)
        result.validate()
        return result

    def validate(self) -> None:
        positive = {
            "damping_lambda": self.damping_lambda,
            "rank_tol": self.rank_tol,
            "max_condition_number": self.max_condition_number,
            "minimum_distance": self.minimum_distance,
            "max_molecule_rms_displacement": self.max_molecule_rms_displacement,
            "max_atom_displacement": self.max_atom_displacement,
        }
        if any(float(value) <= 0 for value in positive.values()):
            raise ValueError(f"Jacobian settings must be positive: {positive}")
        if self.max_relinearizations < 1:
            raise ValueError("max_relinearizations must be positive")
        if not self.backtracking_scales or any(
            not 0.0 < float(scale) <= 1.0 for scale in self.backtracking_scales
        ):
            raise ValueError("backtracking scales must be in (0, 1]")


@dataclass
class ConstraintSystem:
    residual: Tensor
    jacobian: Tensor
    weights: Tensor
    constraint_types: tuple[str, ...]
    counts: dict[str, int]
    diagnostics: dict[str, Any]

    @property
    def objective(self) -> Tensor:
        if not self.residual.numel():
            return self.residual.new_zeros(())
        return (self.weights * self.residual.square()).sum()


def constraint_type_statistics(system: ConstraintSystem) -> dict[str, dict[str, float]]:
    """Summarize active residuals by constraint type."""

    result = {}
    for name in ("bond", "angle", "clash"):
        mask = torch.tensor(
            [value == name for value in system.constraint_types],
            device=system.residual.device,
            dtype=torch.bool,
        )
        values = system.residual[mask]
        weights = system.weights[mask]
        result[name] = {
            "count": int(values.numel()),
            "residual_norm": float(torch.linalg.vector_norm(values))
            if values.numel()
            else 0.0,
            "residual_max_abs": float(values.abs().max()) if values.numel() else 0.0,
            "weighted_objective": float((weights * values.square()).sum())
            if values.numel()
            else 0.0,
        }
    return result


def _safe_direction(relative: Tensor, minimum_distance: float) -> tuple[Tensor, Tensor]:
    distance = torch.linalg.vector_norm(relative, dim=-1)
    fallback = relative.new_tensor([1.0, 0.0, 0.0]).expand_as(relative)
    direction = torch.where(
        (distance > minimum_distance)[:, None],
        relative / distance.clamp_min(minimum_distance)[:, None],
        fallback,
    )
    return direction, distance


def bond_residual_jacobian(
    coordinates: Tensor,
    pairs: Tensor,
    target_distances: Tensor,
    *,
    minimum_distance: float = 1.0e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return distance residuals, analytic rows, and degeneracy mask."""

    coordinates = torch.as_tensor(coordinates, dtype=torch.float64)
    pairs = torch.as_tensor(pairs, device=coordinates.device, dtype=torch.long).reshape(2, -1)
    targets = torch.as_tensor(
        target_distances, device=coordinates.device, dtype=coordinates.dtype
    ).reshape(-1)
    if pairs.size(1) != targets.numel():
        raise ValueError("bond pair and target counts differ")
    if not pairs.numel():
        return (
            coordinates.new_empty(0),
            coordinates.new_empty((0, coordinates.numel())),
            torch.empty(0, device=coordinates.device, dtype=torch.bool),
        )
    left, right = pairs
    direction, distance = _safe_direction(
        coordinates[left] - coordinates[right], minimum_distance
    )
    jacobian = coordinates.new_zeros((pairs.size(1), coordinates.numel()))
    rows = torch.arange(pairs.size(1), device=coordinates.device)
    for axis in range(3):
        jacobian[rows, 3 * left + axis] = direction[:, axis]
        jacobian[rows, 3 * right + axis] = -direction[:, axis]
    return distance - targets, jacobian, distance <= minimum_distance


def cosine_angle_residual_jacobian(
    coordinates: Tensor,
    triplets: Tensor,
    target_cosines: Tensor,
    *,
    minimum_distance: float = 1.0e-8,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return cosine residuals and analytic rows without arccos derivatives."""

    coordinates = torch.as_tensor(coordinates, dtype=torch.float64)
    triplets = torch.as_tensor(
        triplets, device=coordinates.device, dtype=torch.long
    ).reshape(-1, 3)
    targets = torch.as_tensor(
        target_cosines, device=coordinates.device, dtype=coordinates.dtype
    ).reshape(-1)
    if triplets.size(0) != targets.numel():
        raise ValueError("angle triplet and target counts differ")
    if not triplets.numel():
        return (
            coordinates.new_empty(0),
            coordinates.new_empty((0, coordinates.numel())),
            torch.empty(0, device=coordinates.device, dtype=torch.bool),
            coordinates.new_empty(0),
        )
    left, center, right = triplets.unbind(-1)
    u = coordinates[left] - coordinates[center]
    v = coordinates[right] - coordinates[center]
    u_norm = torch.linalg.vector_norm(u, dim=-1)
    v_norm = torch.linalg.vector_norm(v, dim=-1)
    degenerate = (u_norm <= minimum_distance) | (v_norm <= minimum_distance)
    safe_u = u_norm.clamp_min(minimum_distance)
    safe_v = v_norm.clamp_min(minimum_distance)
    cosine = ((u * v).sum(-1) / (safe_u * safe_v)).clamp(-1.0, 1.0)
    derivative_u = v / (safe_u * safe_v)[:, None] - (
        cosine / safe_u.square()
    )[:, None] * u
    derivative_v = u / (safe_u * safe_v)[:, None] - (
        cosine / safe_v.square()
    )[:, None] * v
    derivative_u = torch.where(
        degenerate[:, None], torch.zeros_like(derivative_u), derivative_u
    )
    derivative_v = torch.where(
        degenerate[:, None], torch.zeros_like(derivative_v), derivative_v
    )
    derivative_center = -(derivative_u + derivative_v)
    jacobian = coordinates.new_zeros((triplets.size(0), coordinates.numel()))
    rows = torch.arange(triplets.size(0), device=coordinates.device)
    for atoms, values in (
        (left, derivative_u),
        (center, derivative_center),
        (right, derivative_v),
    ):
        for axis in range(3):
            jacobian[rows, 3 * atoms + axis] = values[:, axis]
    sine = torch.sqrt((1.0 - cosine.square()).clamp_min(0.0))
    return cosine - targets, jacobian, degenerate, sine


def clash_residual_jacobian(
    coordinates: Tensor,
    pairs: Tensor,
    safe_distances: Tensor,
    *,
    minimum_distance: float = 1.0e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return active penetration residuals and separation Jacobian rows."""

    coordinates = torch.as_tensor(coordinates, dtype=torch.float64)
    pairs = torch.as_tensor(pairs, device=coordinates.device, dtype=torch.long).reshape(2, -1)
    safe = torch.as_tensor(
        safe_distances, device=coordinates.device, dtype=coordinates.dtype
    ).reshape(-1)
    if pairs.size(1) != safe.numel():
        raise ValueError("clash pair and safe-distance counts differ")
    if not pairs.numel():
        return (
            coordinates.new_empty(0),
            coordinates.new_empty((0, coordinates.numel())),
            torch.empty(0, device=coordinates.device, dtype=torch.bool),
        )
    left, right = pairs
    direction, distance = _safe_direction(
        coordinates[left] - coordinates[right], minimum_distance
    )
    jacobian = coordinates.new_zeros((pairs.size(1), coordinates.numel()))
    rows = torch.arange(pairs.size(1), device=coordinates.device)
    for axis in range(3):
        jacobian[rows, 3 * left + axis] = -direction[:, axis]
        jacobian[rows, 3 * right + axis] = direction[:, axis]
    return safe - distance, jacobian, distance <= minimum_distance


def _interval_boundary(values: Tensor, ranges: Tensor) -> tuple[Tensor, Tensor]:
    lower, upper = ranges[:, 0], ranges[:, 1]
    active = (values < lower) | (values > upper)
    boundary = torch.where(values < lower, lower, torch.where(values > upper, upper, values))
    return boundary, active


def _normalized_type_weights(
    count: int, value: float, template: Tensor
) -> Tensor:
    if count == 0:
        return template.new_empty(0)
    return template.new_full((count,), float(value) / float(count))


def build_constraint_system(
    coordinates: Tensor,
    bonds: Tensor,
    bond_ranges: Tensor,
    angles: Tensor,
    angle_ranges: Tensor,
    config: JacobianBACConfig,
) -> ConstraintSystem:
    """Assemble active local analytic rows for one molecular graph."""

    config.validate()
    coordinates = torch.as_tensor(coordinates, dtype=torch.float64)
    device = coordinates.device
    bonds = torch.as_tensor(bonds, device=device, dtype=torch.long).reshape(2, -1)
    bond_ranges = torch.as_tensor(
        bond_ranges, device=device, dtype=coordinates.dtype
    ).reshape(-1, 3)
    angles = torch.as_tensor(angles, device=device, dtype=torch.long)
    if angles.ndim == 2 and angles.size(0) == 3:
        angles = angles.t()
    angles = angles.reshape(-1, 3)
    angle_ranges = torch.as_tensor(
        angle_ranges, device=device, dtype=coordinates.dtype
    ).reshape(-1, 3)

    current_bonds = bond_lengths(coordinates, bonds)
    bond_targets, bond_active = _interval_boundary(current_bonds, bond_ranges)
    active_bonds = bonds[:, bond_active]
    bond_residual, bond_jacobian, degenerate_bonds = bond_residual_jacobian(
        coordinates,
        active_bonds,
        bond_targets[bond_active],
        minimum_distance=config.minimum_distance,
    )

    current_angles = bond_angles(coordinates, angles)
    angle_targets, angle_active = _interval_boundary(current_angles, angle_ranges)
    active_angles = angles[angle_active]
    angle_residual, angle_jacobian, degenerate_angles, angle_sine = (
        cosine_angle_residual_jacobian(
            coordinates,
            active_angles,
            torch.cos(angle_targets[angle_active]),
            minimum_distance=config.minimum_distance,
        )
    )
    valid_angles = ~degenerate_angles
    angle_residual = angle_residual[valid_angles]
    angle_jacobian = angle_jacobian[valid_angles]
    angle_sine = angle_sine[valid_angles]

    clash = sparse_clash_edges(
        coordinates,
        bonds,
        cutoff=config.clash_cutoff,
        allowed_contact=config.clash_allowed_contact,
        exclude_topology_distance=config.clash_exclude_topology_distance,
        max_edges_per_graph=config.max_clash_edges,
    )
    clash_active = clash["active_mask"]
    active_clashes = clash["edge_index"][:, clash_active]
    clash_residual, clash_jacobian, degenerate_clashes = clash_residual_jacobian(
        coordinates,
        active_clashes,
        clash["allowed_contact"][clash_active],
        minimum_distance=config.minimum_distance,
    )

    near_linear = angle_sine < config.near_linear_sine_threshold
    bond_weights = _normalized_type_weights(
        bond_residual.numel(), config.bond_weight, coordinates
    )
    angle_weights = _normalized_type_weights(
        angle_residual.numel(), config.angle_weight, coordinates
    )
    if angle_weights.numel():
        angle_weights = angle_weights * torch.where(
            near_linear,
            angle_weights.new_full(angle_weights.shape, config.near_linear_weight),
            torch.ones_like(angle_weights),
        )
    clash_weights = _normalized_type_weights(
        clash_residual.numel(), config.clash_weight, coordinates
    )
    residual = torch.cat((bond_residual, angle_residual, clash_residual))
    jacobian = torch.cat((bond_jacobian, angle_jacobian, clash_jacobian), dim=0)
    weights = torch.cat((bond_weights, angle_weights, clash_weights))
    types = (
        ("bond",) * bond_residual.numel()
        + ("angle",) * angle_residual.numel()
        + ("clash",) * clash_residual.numel()
    )
    return ConstraintSystem(
        residual=residual,
        jacobian=jacobian,
        weights=weights,
        constraint_types=types,
        counts={
            "total_bond": int(bonds.size(1)),
            "active_bond": int(bond_residual.numel()),
            "total_angle": int(angles.size(0)),
            "active_angle": int(angle_residual.numel()),
            "candidate_clash": int(clash["edge_index"].size(1)),
            "active_clash": int(clash_residual.numel()),
        },
        diagnostics={
            "degenerate_bond_count": int(degenerate_bonds.sum()),
            "degenerate_angle_count": int(degenerate_angles.sum()),
            "degenerate_clash_count": int(degenerate_clashes.sum()),
            "near_linear_angle_count": int(near_linear.sum()),
            "jacobian_shape": [int(jacobian.size(0)), int(jacobian.size(1))],
        },
    )


def remove_rigid_update(
    coordinates: Tensor, update: Tensor
) -> tuple[Tensor, dict[str, float]]:
    """Project centroid translation and infinitesimal global rotation."""

    coordinates = torch.as_tensor(coordinates, dtype=torch.float64)
    update = torch.as_tensor(update, device=coordinates.device, dtype=coordinates.dtype)
    before = torch.linalg.vector_norm(update)
    translated = update - update.mean(0, keepdim=True)
    centered = coordinates - coordinates.mean(0, keepdim=True)
    blocks = []
    for point in centered:
        x, y, z = point.unbind()
        blocks.append(
            torch.stack(
                (
                    torch.stack((x.new_zeros(()), z, -y)),
                    torch.stack((-z, x.new_zeros(()), x)),
                    torch.stack((y, -x, x.new_zeros(()))),
                )
            )
        )
    rotation_matrix = torch.cat(blocks, dim=0) if blocks else coordinates.new_empty((0, 3))
    if rotation_matrix.numel() and int(torch.linalg.matrix_rank(rotation_matrix)) > 0:
        omega = torch.linalg.lstsq(rotation_matrix, translated.reshape(-1, 1)).solution
        rotation = (rotation_matrix @ omega).reshape_as(translated)
        projected = translated - rotation
    else:
        projected = translated
    return projected, {
        "raw_update_norm": float(before),
        "translation_removed_norm": float(torch.linalg.vector_norm(translated)),
        "rigid_projected_norm": float(torch.linalg.vector_norm(projected)),
        "centroid_update_norm": float(torch.linalg.vector_norm(projected.mean(0))),
    }


def _mobility_vector(
    atom_count: int,
    atomic_numbers: Tensor | None,
    config: JacobianBACConfig,
    template: Tensor,
) -> Tensor:
    if atomic_numbers is None:
        atom = template.new_full((atom_count,), config.heavy_atom_mobility)
    else:
        numbers = torch.as_tensor(atomic_numbers, device=template.device).reshape(-1)
        if numbers.numel() != atom_count:
            raise ValueError("atomic-number count differs from coordinate count")
        atom = torch.where(
            numbers == 1,
            template.new_full((atom_count,), config.hydrogen_mobility),
            template.new_full((atom_count,), config.heavy_atom_mobility),
        )
    return atom.clamp_min(1.0e-8).repeat_interleave(3)


def solve_damped_system(
    system: ConstraintSystem,
    atom_count: int,
    config: JacobianBACConfig,
    *,
    atomic_numbers: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Solve one weighted linearization with fail-closed diagnostics."""

    width = 3 * int(atom_count)
    zero = system.jacobian.new_zeros(width)
    base = {
        "solver_backend": "none",
        "solver_status": "NO_ACTIVE_CONSTRAINT",
        "effective_rank": 0,
        "singular_value_max": 0.0,
        "singular_value_min_retained": 0.0,
        "condition_number": 0.0,
        "truncated_direction_count": 0,
        "damping": config.damping_lambda,
        "predicted_reduction": 0.0,
    }
    if not system.residual.numel():
        return zero, base
    if not all(
        bool(torch.isfinite(value).all())
        for value in (system.residual, system.jacobian, system.weights)
    ):
        return zero, {**base, "solver_status": "NONFINITE_SYSTEM"}
    mobility = _mobility_vector(
        atom_count, atomic_numbers, config, system.jacobian
    )
    mobility_sqrt = torch.sqrt(mobility)
    weighted_jacobian = torch.sqrt(system.weights)[:, None] * system.jacobian
    scaled_jacobian = weighted_jacobian * mobility_sqrt[None, :]
    weighted_residual = torch.sqrt(system.weights) * system.residual
    try:
        _, singular, _ = torch.linalg.svd(
            scaled_jacobian, full_matrices=False
        )
    except RuntimeError:
        return zero, {**base, "solver_status": "SVD_DIAGNOSTIC_FAILED"}
    sigma_max = float(singular.max()) if singular.numel() else 0.0
    retain = singular >= config.rank_tol * max(sigma_max, 1.0e-30)
    effective_rank = int(retain.sum())
    sigma_min = float(singular[retain].min()) if effective_rank else 0.0
    condition = sigma_max / max(sigma_min, 1.0e-30) if effective_rank else math.inf
    diagnostics = {
        **base,
        "effective_rank": effective_rank,
        "singular_value_max": sigma_max,
        "singular_value_min_retained": sigma_min,
        "condition_number": condition,
        "truncated_direction_count": int(singular.numel() - effective_rank),
    }
    if effective_rank == 0:
        return zero, {**diagnostics, "solver_status": "EFFECTIVE_RANK_ZERO"}
    backend = "augmented_lstsq"
    try:
        identity = torch.eye(width, device=system.jacobian.device, dtype=system.jacobian.dtype)
        augmented = torch.cat(
            (scaled_jacobian, math.sqrt(config.damping_lambda) * identity), dim=0
        )
        right_hand = torch.cat(((-weighted_residual)[:, None], zero[:, None]), dim=0)
        scaled_update = torch.linalg.lstsq(augmented, right_hand).solution[:, 0]
        if condition > config.max_condition_number or not bool(
            torch.isfinite(scaled_update).all()
        ):
            backend = "damped_truncated_svd"
            left, singular, right_h = torch.linalg.svd(
                scaled_jacobian, full_matrices=False
            )
            retain = singular >= config.rank_tol * singular.max().clamp_min(1.0e-30)
            factors = torch.where(
                retain,
                singular / (singular.square() + config.damping_lambda),
                torch.zeros_like(singular),
            )
            scaled_update = -right_h.mT @ (factors * (left.mT @ weighted_residual))
    except RuntimeError:
        return zero, {**diagnostics, "solver_status": "FACTORIZATION_FAILED"}
    update = mobility_sqrt * scaled_update
    if not bool(torch.isfinite(update).all()):
        return zero, {**diagnostics, "solver_backend": backend, "solver_status": "NONFINITE_UPDATE"}
    predicted_before = float(system.objective)
    predicted_after = float(
        (system.weights * (system.residual + system.jacobian @ update).square()).sum()
    )
    predicted_reduction = predicted_before - predicted_after
    if predicted_reduction <= 0.0:
        return zero, {
            **diagnostics,
            "solver_backend": backend,
            "solver_status": "NONPOSITIVE_PREDICTED_REDUCTION",
            "predicted_reduction": predicted_reduction,
        }
    return update, {
        **diagnostics,
        "solver_backend": backend,
        "solver_status": "SOLVED",
        "predicted_reduction": predicted_reduction,
    }


def _trust_scale(update: Tensor, config: JacobianBACConfig) -> tuple[Tensor, float]:
    if not update.numel():
        return update, 1.0
    atom_max = float(torch.linalg.vector_norm(update, dim=-1).max())
    graph_rms = float(torch.sqrt(update.square().sum(-1).mean()))
    scale = min(
        1.0,
        config.max_atom_displacement / max(atom_max, 1.0e-30),
        config.max_molecule_rms_displacement / max(graph_rms, 1.0e-30),
    )
    return update * scale, scale


def solve_bac_jacobian(
    source: Tensor,
    record: Any,
    validity: Any,
    *,
    bonds: Tensor,
    bond_ranges: Tensor,
    angles: Tensor,
    angle_ranges: Tensor,
    atomic_numbers: Tensor | None = None,
    config: JacobianBACConfig | Mapping[str, Any] | None = None,
    safety_config: BACSafetyConfig | None = None,
) -> dict[str, Any]:
    """Run bounded relinearized BAC correction for one graph."""

    if not isinstance(config, JacobianBACConfig):
        config = JacobianBACConfig.from_mapping(config)
    config.validate()
    safety = safety_config or BACSafetyConfig(
        max_atom_displacement=config.max_atom_displacement,
        max_molecule_rms_displacement=config.max_molecule_rms_displacement,
    )
    original = torch.as_tensor(source)
    output_dtype = original.dtype
    source64 = original.detach().to(torch.float64).cpu()
    current = source64.clone()
    started = time.perf_counter()
    iterations = []
    accepted_any = False
    initial_system = build_constraint_system(
        current, bonds, bond_ranges, angles, angle_ranges, config
    )
    initial_objective = float(initial_system.objective)
    initial_type_statistics = constraint_type_statistics(initial_system)
    status = "NO_ACTIVE_CONSTRAINT"
    for iteration in range(config.max_relinearizations):
        system = build_constraint_system(
            current, bonds, bond_ranges, angles, angle_ranges, config
        )
        if not system.residual.numel():
            status = "CONVERGED" if accepted_any else "NO_ACTIVE_CONSTRAINT"
            break
        flat_update, solve = solve_damped_system(
            system,
            current.size(0),
            config,
            atomic_numbers=atomic_numbers,
        )
        if solve["solver_status"] != "SOLVED":
            iterations.append(
                {
                    "iteration": iteration + 1,
                    "counts": system.counts,
                    **system.diagnostics,
                    **solve,
                    "attempted_step_scales": [],
                }
            )
            status = solve["solver_status"]
            break
        projected, rigid = remove_rigid_update(current, flat_update.reshape(-1, 3))
        projected_flat = projected.reshape(-1)
        predicted_after_projection = float(
            (
                system.weights
                * (system.residual + system.jacobian @ projected_flat).square()
            ).sum()
        )
        predicted_reduction = float(system.objective) - predicted_after_projection
        if predicted_reduction <= 0.0:
            iterations.append(
                {
                    "iteration": iteration + 1,
                    "counts": system.counts,
                    **system.diagnostics,
                    **solve,
                    **rigid,
                    "solver_status": "RIGID_PROJECTION_REMOVED_REDUCTION",
                    "attempted_step_scales": [],
                }
            )
            status = "RIGID_PROJECTION_REMOVED_REDUCTION"
            break
        trusted, trust_scale = _trust_scale(projected, config)
        attempts = []
        selected = None
        objective_before = float(system.objective)
        for scale in config.backtracking_scales:
            proposal = current + float(scale) * trusted
            candidate_system = build_constraint_system(
                proposal, bonds, bond_ranges, angles, angle_ranges, config
            )
            objective_after = float(candidate_system.objective)
            decision = evaluate_bac_proposal(
                source64.to(torch.float32),
                proposal.to(torch.float32),
                record,
                validity,
                safety,
            )
            objective_decreased = objective_after < objective_before - 1.0e-15
            accepted = bool(decision["accepted"] and objective_decreased)
            attempt = {
                "scale": float(scale),
                "objective_after": objective_after,
                "objective_decreased": objective_decreased,
                "hard_safety_accepted": bool(decision["accepted"]),
                "accepted": accepted,
                "reasons": list(decision["reasons"]),
            }
            attempts.append(attempt)
            if accepted:
                selected = (proposal, candidate_system, decision, float(scale))
                break
        iteration_diag = {
            "iteration": iteration + 1,
            "counts": system.counts,
            **system.diagnostics,
            **solve,
            **rigid,
            "trust_region_scale": trust_scale,
            "trust_scaled_norm": float(torch.linalg.vector_norm(trusted)),
            "objective_before": objective_before,
            "attempted_step_scales": attempts,
        }
        if selected is None:
            iterations.append(
                {
                    **iteration_diag,
                    "solver_status": "BACKTRACKING_REJECTED",
                    "accepted_step_scale": 0.0,
                    "actual_reduction": 0.0,
                    "reduction_ratio": 0.0,
                }
            )
            status = "BACKTRACKING_REJECTED"
            break
        proposal, candidate_system, decision, selected_scale = selected
        actual_reduction = objective_before - float(candidate_system.objective)
        iterations.append(
            {
                **iteration_diag,
                "solver_status": "ITERATION_ACCEPTED",
                "accepted_step_scale": selected_scale,
                "actual_reduction": actual_reduction,
                "reduction_ratio": actual_reduction / max(predicted_reduction, 1.0e-30),
                "bac_gain": float(decision["bac_gain"]),
            }
        )
        current = proposal
        accepted_any = True
        status = "ITERATION_ACCEPTED"
    final_decision = evaluate_bac_proposal(
        source64.to(torch.float32),
        current.to(torch.float32),
        record,
        validity,
        safety,
    )
    if not accepted_any or not final_decision["accepted"]:
        current = source64.clone()
        accepted_any = False
        if status == "ITERATION_ACCEPTED":
            status = "FINAL_SAFETY_ROLLBACK"
    final_system = build_constraint_system(
        current, bonds, bond_ranges, angles, angle_ranges, config
    )
    result_coordinates = current.to(output_dtype)
    return {
        "schema_version": JACOBIAN_SCHEMA_VERSION,
        "coordinates": result_coordinates,
        "accepted": accepted_any,
        "solver_status": status,
        "iteration_count": len(iterations),
        "iterations": iterations,
        "initial_constraint_counts": initial_system.counts,
        "initial_constraint_diagnostics": initial_system.diagnostics,
        "initial_objective": initial_objective,
        "initial_type_statistics": initial_type_statistics,
        "final_objective": float(final_system.objective),
        "final_type_statistics": constraint_type_statistics(final_system),
        "objective_reduction": initial_objective - float(final_system.objective),
        "final_safety": final_decision,
        "runtime_seconds": time.perf_counter() - started,
        "config": asdict(config),
        "test_records_read": 0,
        "test_assets_opened": False,
        "frozen_holdout_records_opened": 0,
        "validation_only": True,
    }
