from __future__ import annotations

import math

import pytest
import torch

from etflow.ecir.bond_explicit import (
    bond_length_jacobian,
    bond_length_residual,
    bounded_bond_residual,
    solve_bond_cartesian_correction,
)


def test_bond_jacobian_direction_and_sign():
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    bonds = torch.tensor([[0], [1]])
    jacobian = bond_length_jacobian(coordinates, bonds)
    assert jacobian.shape == (1, 6)
    assert torch.equal(jacobian, torch.tensor([[-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]))


def test_single_bond_stretch_has_minimum_norm_analytic_solution():
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    bonds = torch.tensor([[0], [1]])
    correction, diagnostics = solve_bond_cartesian_correction(
        coordinates, bonds, torch.tensor([0.2], dtype=torch.float64), damping=1.0e-12
    )
    assert diagnostics["status"] == "SOLVED"
    assert correction[:, 0].tolist() == pytest.approx([-0.1, 0.1], abs=1.0e-10)
    assert correction[:, 1:].abs().max() == 0.0


def test_multibond_solution_is_globally_consistent():
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64)
    bonds = torch.tensor([[0, 1], [1, 2]])
    residual = torch.tensor([0.1, -0.1], dtype=torch.float64)
    correction, diagnostics = solve_bond_cartesian_correction(
        coordinates, bonds, residual, damping=1.0e-10
    )
    realized = bond_length_jacobian(coordinates, bonds) @ correction.reshape(-1)
    assert diagnostics["status"] == "SOLVED"
    assert realized.tolist() == pytest.approx(residual.tolist(), abs=1.0e-8)


def test_translation_nullspace_is_removed():
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [1.5, 1.0, 0.0]])
    bonds = torch.tensor([[0, 1], [1, 2]])
    correction, diagnostics = solve_bond_cartesian_correction(
        coordinates, bonds, torch.tensor([0.03, -0.02])
    )
    assert torch.linalg.vector_norm(correction.mean(0)) < 1.0e-7
    assert diagnostics["translation_norm"] < 1.0e-7


def test_solver_reports_small_damped_residual():
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    correction, diagnostics = solve_bond_cartesian_correction(
        coordinates, torch.tensor([[0], [1]]),
        torch.tensor([0.05], dtype=torch.float64), damping=1.0e-6,
    )
    assert torch.isfinite(correction).all()
    assert diagnostics["relative_linear_residual"] < 1.0e-5


def test_atom_permutation_invariance():
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.7, 0.8, 0.2]])
    bonds = torch.tensor([[0, 1], [1, 2]])
    residual = torch.tensor([0.03, -0.02])
    original, _ = solve_bond_cartesian_correction(coordinates, bonds, residual)
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(3)
    permuted_coordinates = coordinates[permutation]
    permuted_bonds = inverse[bonds]
    permuted, _ = solve_bond_cartesian_correction(permuted_coordinates, permuted_bonds, residual)
    assert torch.allclose(permuted[inverse], original, atol=1.0e-6)


def test_rotation_equivariance():
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [1.6, 0.9, 0.1]])
    bonds = torch.tensor([[0, 1], [1, 2]])
    residual = torch.tensor([0.02, -0.03])
    angle = 0.7
    rotation = torch.tensor([
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    original, _ = solve_bond_cartesian_correction(coordinates, bonds, residual)
    rotated, _ = solve_bond_cartesian_correction(coordinates @ rotation.T, bonds, residual)
    assert torch.allclose(rotated, original @ rotation.T, atol=1.0e-6)


def test_zero_target_residual_produces_clean_identity():
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    bonds = torch.tensor([[0], [1]])
    residual = bond_length_residual(coordinates, coordinates.clone(), bonds)
    correction, diagnostics = solve_bond_cartesian_correction(coordinates, bonds, residual)
    assert diagnostics["status"] == "SOLVED"
    assert torch.equal(correction, torch.zeros_like(coordinates))


def test_condition_failure_falls_back_to_zero():
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    bonds = torch.tensor([[0], [1]])
    correction, diagnostics = solve_bond_cartesian_correction(
        coordinates, bonds, torch.tensor([0.1]), max_condition=0.5
    )
    assert diagnostics["status"] == "CONDITION_FALLBACK"
    assert torch.equal(correction, torch.zeros_like(coordinates))


def test_confidence_gate_and_residual_bound():
    raw = torch.tensor([100.0, -100.0, 0.5])
    closed = bounded_bond_residual(raw, torch.full_like(raw, -100.0), max_abs_residual=0.05)
    open_gate = bounded_bond_residual(raw, torch.full_like(raw, 100.0), max_abs_residual=0.05)
    assert closed.abs().max() < 1.0e-20
    assert open_gate.abs().max() <= 0.05
    assert open_gate[:2].tolist() == pytest.approx([0.05, -0.05])
