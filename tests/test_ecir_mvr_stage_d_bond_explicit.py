from __future__ import annotations

import math

import pytest
import torch
import yaml
from torch_geometric.data import Data

from etflow.ecir.bond_explicit import (
    differentiable_bond_projection,
    bond_length_jacobian,
    bond_length_residual,
    bounded_bond_residual,
    solve_bond_cartesian_correction,
)
from etflow.ecir.mvr_loss import MCVRLoss
from etflow.ecir.mvr_model import MCVRModel


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


def _stage_d_data() -> Data:
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    x_input = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.2, 0.2, 0.0]])
    x_target = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    return Data(
        num_nodes=3, node_attr=torch.randn(3, 10), edge_index=edge_index,
        edge_attr=torch.ones(4, 1), x_input=x_input, x_init=x_input,
        x_target=x_target, rotatable_bond_index=torch.empty(2, 0, dtype=torch.long),
        active_mode_mask=torch.tensor([[1.0, 0, 0, 0, 0, 0]]),
        affected_atom_mask=torch.ones(3), deterministic_error_features=torch.zeros(1, 10),
        difficulty_target=torch.zeros(1), num_rotatable_bonds=torch.zeros(1, dtype=torch.long),
    )


def _stage_d_model(**updates) -> MCVRModel:
    values = dict(
        hidden_dim=24, edge_hidden_dim=24, time_embedding_dim=8,
        num_layers=1, encoder_num_layers=1, error_embedding_dim=8,
        torsion_scale=0.0, high_flex_torsion_scale=0.0,
        torsion_gate_fixed_zero=True, bond_head_enabled=True,
        max_abs_bond_residual=0.05, bond_projection_damping=1.0e-4,
    )
    values.update(updates)
    return MCVRModel(**values)


def test_differentiable_projection_backpropagates_to_residual():
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=True)
    residual = torch.tensor([0.03], requires_grad=True)
    correction, failure = differentiable_bond_projection(
        coordinates, torch.tensor([[0], [1]]), residual
    )
    correction.square().sum().backward()
    assert float(failure) == 0.0
    assert residual.grad is not None and float(residual.grad.abs().sum()) > 0.0


def test_auxiliary_only_strictly_degrades_to_run_a_output():
    torch.manual_seed(11)
    base = MCVRModel(
        hidden_dim=24, edge_hidden_dim=24, time_embedding_dim=8,
        num_layers=1, encoder_num_layers=1, error_embedding_dim=8,
        torsion_scale=0.0, high_flex_torsion_scale=0.0, torsion_gate_fixed_zero=True,
    )
    torch.manual_seed(12)
    auxiliary = _stage_d_model(bond_explicit_alpha=0.0)
    state = auxiliary.state_dict()
    state.update(base.state_dict())
    auxiliary.load_state_dict(state, strict=True)
    with torch.no_grad():
        auxiliary.bond_explicit_head[-1].bias[0] = 2.0
        auxiliary.bond_explicit_head[-1].bias[1] = 2.0
    data = _stage_d_data()
    base_output = base(data, data.x_input, torch.tensor([0.5]))
    auxiliary_output = auxiliary(data, data.x_input, torch.tensor([0.5]))
    assert float(auxiliary_output["v_bond_correction"].detach().abs().sum()) > 0.0
    torch.testing.assert_close(auxiliary_output["v_final"], base_output["v_final"])


def test_explicit_inference_does_not_read_target_or_reference_coordinates():
    model = _stage_d_model(bond_explicit_alpha=1.0)
    data = _stage_d_data()
    first = model(data, data.x_input, torch.tensor([0.5]))["v_final"]
    data.x_target = torch.randn_like(data.x_target) * 100.0
    data.x_ref_aligned = torch.randn_like(data.x_target) * 100.0
    second = model(data, data.x_input, torch.tensor([0.5]))["v_final"]
    torch.testing.assert_close(first, second)


def test_stage_d_target_residual_losses_are_finite_and_reported():
    model = _stage_d_model(bond_explicit_alpha=1.0)
    losses = MCVRLoss({
        "bond_residual": 0.5, "bond_direction": 0.1, "bond_sparse": 0.1,
        "bond_confidence": 0.05, "bond_uncertainty": 0.05,
        "bond_consistency": 0.1,
    })(model, _stage_d_data())
    names = {
        "bond_residual_loss", "bond_direction_loss", "bond_sparse_loss",
        "bond_confidence_loss", "bond_uncertainty_loss", "bond_consistency_loss",
    }
    assert names.issubset(losses)
    assert all(torch.isfinite(losses[name]) for name in names)
    losses["loss"].backward()
    assert model.bond_explicit_head[-1].weight.grad is not None


def test_stage_d_torsion_stays_exactly_disabled_and_checkpoint_load_is_strict():
    model = _stage_d_model(bond_explicit_alpha=1.0)
    clone = _stage_d_model(bond_explicit_alpha=1.0)
    clone.load_state_dict(model.state_dict(), strict=True)
    output = clone(_stage_d_data(), _stage_d_data().x_input, torch.tensor([0.5]))
    assert torch.equal(output["torsion_gate"], torch.zeros_like(output["torsion_gate"]))
    assert torch.equal(
        output["v_torsion_contribution"], torch.zeros_like(output["v_torsion_contribution"])
    )


def test_stage_d_configs_are_fixed_test_free_and_have_only_two_methods():
    paths = [
        "configs/ecir_mvr_stage_d_d1_a_aux_only_seed42_5k.yaml",
        "configs/ecir_mvr_stage_d_d1_b_explicit_bond_seed42_5k.yaml",
    ]
    configs = [yaml.safe_load(open(path, encoding="utf-8")) for path in paths]
    assert [config["stage_d_method"] for config in configs] == ["auxiliary_only", "explicit_bond"]
    for config in configs:
        assert config["training"]["optimizer_steps"] == 5000
        assert config["training"]["checkpoint_steps"] == [500, 1000, 1500, 2000, 3000, 5000]
        assert config["model"]["max_abs_bond_residual"] == 0.05
        assert config["model"]["torsion_gate_fixed_zero"] is True
        assert config["initialize_from_checkpoint"] is None
        assert config["resume_checkpoint"] is None
        assert not any("test" in str(value).lower() for value in config["data"].values())
