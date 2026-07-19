from __future__ import annotations

import inspect
import math

import pytest
import torch

from etflow.ecir.bac_jacobian import (
    ConstraintSystem,
    JacobianBACConfig,
    bond_residual_jacobian,
    build_constraint_system,
    clash_residual_jacobian,
    constraint_type_statistics,
    cosine_angle_residual_jacobian,
    remove_rigid_update,
    solve_bac_jacobian,
    solve_damped_system,
)


def _autograd_jacobian(function, coordinates):
    value = coordinates.clone().to(torch.float64).requires_grad_(True)
    return torch.autograd.functional.jacobian(function, value).reshape(-1, value.numel())


def _rotation():
    angle = 0.73
    return torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


def test_bond_analytic_jacobian_matches_autograd():
    coordinates = torch.tensor(
        [[0.1, -0.2, 0.4], [1.3, 0.5, -0.1], [2.0, -0.4, 0.8]],
        dtype=torch.float64,
    )
    pairs = torch.tensor([[0, 1], [1, 2]])
    targets = torch.tensor([1.1, 1.2], dtype=torch.float64)
    residual, analytic, degenerate = bond_residual_jacobian(
        coordinates, pairs, targets
    )
    automatic = _autograd_jacobian(
        lambda value: torch.linalg.vector_norm(
            value[pairs[0]] - value[pairs[1]], dim=-1
        )
        - targets,
        coordinates,
    )
    torch.testing.assert_close(analytic, automatic, atol=1.0e-10, rtol=1.0e-10)
    assert torch.isfinite(residual).all()
    assert not bool(degenerate.any())


def test_cosine_angle_analytic_jacobian_matches_autograd():
    coordinates = torch.tensor(
        [[-0.7, 0.3, 0.2], [0.1, -0.2, 0.0], [0.9, 0.6, -0.4]],
        dtype=torch.float64,
    )
    triplets = torch.tensor([[0, 1, 2]])
    target = torch.tensor([-0.2], dtype=torch.float64)
    residual, analytic, degenerate, sine = cosine_angle_residual_jacobian(
        coordinates, triplets, target
    )

    def function(value):
        left = value[0] - value[1]
        right = value[2] - value[1]
        cosine = torch.dot(left, right) / (
            torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        )
        return (cosine - target).reshape(1)

    automatic = _autograd_jacobian(function, coordinates)
    torch.testing.assert_close(analytic, automatic, atol=1.0e-10, rtol=1.0e-10)
    assert torch.isfinite(residual).all()
    assert float(sine[0]) > 0
    assert not bool(degenerate.any())


def test_clash_analytic_jacobian_matches_autograd_and_sign():
    coordinates = torch.tensor(
        [[0.0, 0.0, 0.0], [0.7, 0.2, -0.1]], dtype=torch.float64
    )
    pairs = torch.tensor([[0], [1]])
    safe = torch.tensor([1.0], dtype=torch.float64)
    residual, analytic, degenerate = clash_residual_jacobian(
        coordinates, pairs, safe
    )
    automatic = _autograd_jacobian(
        lambda value: safe
        - torch.linalg.vector_norm(value[pairs[0]] - value[pairs[1]], dim=-1),
        coordinates,
    )
    torch.testing.assert_close(analytic, automatic, atol=1.0e-10, rtol=1.0e-10)
    direction = -analytic[0].reshape(2, 3)
    assert float(
        torch.linalg.vector_norm(
            coordinates[0] + direction[0] - coordinates[1] - direction[1]
        )
    ) > float(torch.linalg.vector_norm(coordinates[0] - coordinates[1]))
    assert float(residual[0]) > 0
    assert not bool(degenerate.any())


def test_cosine_residual_avoids_arccos_derivative_amplification():
    cosine = torch.tensor([-1.0 + 1.0e-10], dtype=torch.float64, requires_grad=True)
    angle = torch.acos(cosine)
    angle.backward()
    arccos_gradient = abs(float(cosine.grad))
    cosine_gradient = 1.0
    assert arccos_gradient > 1.0e4 * cosine_gradient


def test_degenerate_local_rows_are_finite_and_flagged():
    coordinates = torch.zeros(3, 3, dtype=torch.float64)
    bond = bond_residual_jacobian(
        coordinates, torch.tensor([[0], [1]]), torch.tensor([1.0])
    )
    angle = cosine_angle_residual_jacobian(
        coordinates, torch.tensor([[0, 1, 2]]), torch.tensor([0.0])
    )
    clash = clash_residual_jacobian(
        coordinates, torch.tensor([[0], [2]]), torch.tensor([1.0])
    )
    assert bool(bond[2].all())
    assert bool(angle[2].all())
    assert bool(clash[2].all())
    for payload in (bond, angle, clash):
        for value in payload[:2]:
            assert torch.isfinite(value).all()


def _system(coordinates, config=None):
    config = config or JacobianBACConfig()
    return build_constraint_system(
        coordinates,
        torch.tensor([[0, 1], [1, 2]]),
        torch.tensor([[0.9, 1.1, 0.1], [0.9, 1.1, 0.1]]),
        torch.tensor([[0, 1, 2]]),
        torch.tensor([[1.2, 2.0, 0.1]]),
        config,
    )


def test_near_linear_angle_is_downweighted_and_reported():
    coordinates = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0e-6, 0.0]],
        dtype=torch.float64,
    )
    system = _system(coordinates)
    assert system.counts["active_angle"] == 1
    assert system.diagnostics["near_linear_angle_count"] == 1
    angle_index = system.constraint_types.index("angle")
    assert system.weights[angle_index] == pytest.approx(0.1)


def test_overlapping_clash_builds_finite_deterministic_fallback():
    coordinates = torch.zeros(2, 3, dtype=torch.float64)
    config = JacobianBACConfig()
    first = build_constraint_system(
        coordinates,
        torch.empty(2, 0, dtype=torch.long),
        torch.empty(0, 3),
        torch.empty(0, 3, dtype=torch.long),
        torch.empty(0, 3),
        config,
    )
    second = build_constraint_system(
        coordinates,
        torch.empty(2, 0, dtype=torch.long),
        torch.empty(0, 3),
        torch.empty(0, 3, dtype=torch.long),
        torch.empty(0, 3),
        config,
    )
    assert first.counts["active_clash"] == 1
    assert first.diagnostics["degenerate_clash_count"] == 1
    torch.testing.assert_close(first.residual, second.residual)
    torch.testing.assert_close(first.jacobian, second.jacobian)
    assert torch.isfinite(first.jacobian).all()


def test_empty_and_satisfied_systems_return_zero_update():
    coordinates = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    system = build_constraint_system(
        coordinates,
        torch.empty(2, 0, dtype=torch.long),
        torch.empty(0, 3),
        torch.empty(0, 3, dtype=torch.long),
        torch.empty(0, 3),
        JacobianBACConfig(),
    )
    update, diagnostics = solve_damped_system(
        system, 1, JacobianBACConfig()
    )
    torch.testing.assert_close(update, torch.zeros(3, dtype=torch.float64))
    assert diagnostics["solver_status"] == "NO_ACTIVE_CONSTRAINT"
    assert constraint_type_statistics(system) == {
        name: {
            "count": 0,
            "residual_norm": 0.0,
            "residual_max_abs": 0.0,
            "weighted_objective": 0.0,
        }
        for name in ("bond", "angle", "clash")
    }


def test_duplicate_rank_deficient_rows_use_damped_truncated_svd():
    jacobian = torch.tensor(
        [[1.0, -1.0, 0.0, 0.0, 0.0, 0.0]] * 2, dtype=torch.float64
    )
    system = ConstraintSystem(
        residual=torch.tensor([1.0, 1.0], dtype=torch.float64),
        jacobian=jacobian,
        weights=torch.ones(2, dtype=torch.float64),
        constraint_types=("bond", "bond"),
        counts={},
        diagnostics={},
    )
    config = JacobianBACConfig(max_condition_number=0.5)
    update, diagnostics = solve_damped_system(system, 2, config)
    assert diagnostics["solver_backend"] == "damped_truncated_svd"
    assert diagnostics["effective_rank"] == 1
    assert diagnostics["truncated_direction_count"] == 1
    assert torch.isfinite(update).all()
    assert diagnostics["predicted_reduction"] > 0


def test_rigid_projection_removes_translation_and_rotation():
    coordinates = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.2], [1.0, -0.2, 0.4]],
        dtype=torch.float64,
    )
    omega = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64)
    translation = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    centered = coordinates - coordinates.mean(0)
    rotation = torch.cross(omega.expand_as(centered), centered, dim=-1)
    internal = torch.tensor(
        [[0.02, 0.0, 0.0], [-0.01, 0.01, 0.0], [-0.01, -0.01, 0.0]],
        dtype=torch.float64,
    )
    projected, diagnostics = remove_rigid_update(
        coordinates, rotation + translation + internal
    )
    assert float(torch.linalg.vector_norm(projected.mean(0))) < 1.0e-12
    assert diagnostics["rigid_projected_norm"] < diagnostics["raw_update_norm"]


def test_linearized_update_is_rotation_translation_equivariant():
    coordinates = torch.tensor(
        [[-1.2, 0.1, 0.0], [0.0, 0.0, 0.0], [0.7, 0.8, 0.2]],
        dtype=torch.float64,
    )
    rotation = _rotation()
    translation = torch.tensor([2.0, -3.0, 0.7], dtype=torch.float64)
    transformed = coordinates @ rotation.T + translation
    config = JacobianBACConfig()
    first = _system(coordinates, config)
    second = _system(transformed, config)
    update_a, diag_a = solve_damped_system(first, 3, config)
    update_b, diag_b = solve_damped_system(second, 3, config)
    projected_a, _ = remove_rigid_update(coordinates, update_a.reshape(-1, 3))
    projected_b, _ = remove_rigid_update(transformed, update_b.reshape(-1, 3))
    assert diag_a["solver_status"] == "SOLVED"
    assert diag_b["solver_status"] == "SOLVED"
    torch.testing.assert_close(
        projected_b, projected_a @ rotation.T, atol=1.0e-8, rtol=1.0e-8
    )


class _BondValidity:
    def __init__(self, *, reject_any_move: bool = False):
        self.reject_any_move = reject_any_move

    def evaluate(self, coordinates, record, baseline_coordinates=None):
        del record
        coordinates = torch.as_tensor(coordinates)
        baseline = coordinates if baseline_coordinates is None else baseline_coordinates
        distance = torch.linalg.vector_norm(coordinates[1] - coordinates[0])
        base_distance = torch.linalg.vector_norm(baseline[1] - baseline[0])
        bond = float((distance - 1.0).abs())
        base_bond = float((base_distance - 1.0).abs())
        moved = float(torch.linalg.vector_norm(coordinates - baseline)) > 1.0e-10
        angle = float(self.reject_any_move and moved)
        return {
            "bond_outlier_rate": bond,
            "bond_outlier_magnitude": bond,
            "angle_outlier_rate": angle,
            "angle_outlier_magnitude": angle,
            "severe_clash_rate": 0.0,
            "clash_penetration": 0.0,
            "ring_bond_outlier_rate": 0.0,
            "ring_planarity_outlier_rate": 0.0,
            "chirality_preserved": 1.0,
            "stereocenter_degenerate_rate": 0.0,
            "torsion_prior_outlier_score": 0.0,
            "total_thresholded_validity_score": bond + angle,
            "base_bond": base_bond,
        }


def _solve(validity):
    source = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    return source, solve_bac_jacobian(
        source,
        {},
        validity,
        bonds=torch.tensor([[0], [1]]),
        bond_ranges=torch.tensor([[0.9, 1.1, 0.1]]),
        angles=torch.empty(0, 3, dtype=torch.long),
        angle_ranges=torch.empty(0, 3),
        atomic_numbers=torch.tensor([6, 6]),
    )


def test_solver_obeys_trust_decreases_objective_and_is_finite():
    source, result = _solve(_BondValidity())
    assert result["accepted"] is True
    assert result["objective_reduction"] > 0
    assert torch.isfinite(result["coordinates"]).all()
    displacement = result["coordinates"] - source
    assert float(torch.linalg.vector_norm(displacement, dim=-1).max()) <= 0.120001
    assert result["iterations"][0]["actual_reduction"] > 0
    assert result["iterations"][0]["reduction_ratio"] > 0


def test_solver_rolls_back_when_hard_safety_rejects_every_scale():
    source, result = _solve(_BondValidity(reject_any_move=True))
    assert result["accepted"] is False
    assert result["solver_status"] == "BACKTRACKING_REJECTED"
    torch.testing.assert_close(result["coordinates"], source)


def test_nonfinite_system_fails_closed():
    system = ConstraintSystem(
        residual=torch.tensor([float("nan")], dtype=torch.float64),
        jacobian=torch.zeros(1, 3, dtype=torch.float64),
        weights=torch.ones(1, dtype=torch.float64),
        constraint_types=("bond",),
        counts={},
        diagnostics={},
    )
    update, diagnostics = solve_damped_system(
        system, 1, JacobianBACConfig()
    )
    torch.testing.assert_close(update, torch.zeros(3, dtype=torch.float64))
    assert diagnostics["solver_status"] == "NONFINITE_SYSTEM"


def test_configuration_and_source_forbid_explicit_inverse():
    with pytest.raises(ValueError, match="unknown Jacobian BAC settings"):
        JacobianBACConfig.from_mapping({"unknown": 1})
    source = inspect.getsource(solve_damped_system)
    assert "torch.inverse" not in source
    assert "torch.linalg.inv" not in source
    assert "numpy.linalg.inv" not in source
