from __future__ import annotations

import torch

from etflow.ecir.bac_jacobian import (
    JacobianBACConfig,
    build_constraint_system,
)
from etflow.ecir.bac_jacobian_guided import (
    _project_row_space,
    jacobian_projection,
    posthoc_jacobian_correction,
    trust_region_hybrid,
)
from etflow.ecir.mvr_v2_bac import MCVRBACModel, V2_D_BOND_ANGLE_CLASH


class _Validity:
    def __init__(self, *, reject_atom_move_above: float | None = None):
        self.reject_atom_move_above = reject_atom_move_above

    def evaluate(self, coordinates, record, baseline_coordinates=None):
        del record
        coordinates = torch.as_tensor(coordinates)
        baseline = (
            coordinates if baseline_coordinates is None else torch.as_tensor(baseline_coordinates)
        )
        distances = torch.linalg.vector_norm(coordinates[1:] - coordinates[:-1], dim=-1)
        baseline_distances = torch.linalg.vector_norm(baseline[1:] - baseline[:-1], dim=-1)
        bond = float((distances - 1.0).abs().mean())
        base_bond = float((baseline_distances - 1.0).abs().mean())
        max_move = float(torch.linalg.vector_norm(coordinates - baseline, dim=-1).max())
        rejected = bool(
            self.reject_atom_move_above is not None and max_move > self.reject_atom_move_above
        )
        angle = float(rejected)
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


def _constraints(atom_count: int):
    bonds = torch.stack((torch.arange(atom_count - 1), torch.arange(1, atom_count)))
    return {
        "bonds": bonds,
        "bond_ranges": torch.tensor([[0.9, 1.1, 0.1]] * (atom_count - 1)),
        "angles": torch.empty(0, 3, dtype=torch.long),
        "angle_ranges": torch.empty(0, 3),
    }


def test_guided_module_does_not_change_d1_state_dict_contract():
    model = MCVRBACModel(
        atom_feature_dim=4,
        hidden_dim=8,
        edge_hidden_dim=8,
        time_embedding_dim=8,
        num_layers=1,
        encoder_num_layers=1,
        error_embedding_dim=4,
        bac_mode=V2_D_BOND_ANGLE_CLASH,
    )
    frozen = {name: value.clone() for name, value in model.state_dict().items()}
    clone = MCVRBACModel(
        atom_feature_dim=4,
        hidden_dim=8,
        edge_hidden_dim=8,
        time_embedding_dim=8,
        num_layers=1,
        encoder_num_layers=1,
        error_embedding_dim=4,
        bac_mode=V2_D_BOND_ANGLE_CLASH,
    )
    incompatible = clone.load_state_dict(frozen, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys


def test_candidate_a_fixed_alphas_apply_increasing_bounded_correction():
    source = torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]])
    d1 = torch.tensor([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])
    config = JacobianBACConfig(
        max_molecule_rms_displacement=0.3,
        max_atom_displacement=0.3,
    )
    results = [
        posthoc_jacobian_correction(
            source,
            d1,
            {},
            _Validity(),
            alpha=alpha,
            d1_accepted=True,
            atomic_numbers=torch.tensor([6, 6]),
            config=config,
            **_constraints(2),
        )
        for alpha in (0.25, 0.5, 1.0)
    ]
    assert all(result["accepted"] for result in results)
    corrections = [
        float(torch.linalg.vector_norm(result["coordinates"] - d1)) for result in results
    ]
    assert corrections[0] < corrections[1] < corrections[2]
    assert all(result["diagnostics"]["objective_reduction"] > 0 for result in results)
    assert all(
        result["diagnostics"]["correction_movement"]["atom_max"] <= 0.120001 for result in results
    )


def test_candidate_a_hard_safety_falls_back_exactly_to_d1():
    source = torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]])
    d1 = torch.tensor([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]])
    result = posthoc_jacobian_correction(
        source,
        d1,
        {},
        _Validity(reject_atom_move_above=0.05),
        alpha=1.0,
        d1_accepted=True,
        config=JacobianBACConfig(
            max_molecule_rms_displacement=0.3,
            max_atom_displacement=0.3,
        ),
        **_constraints(2),
    )
    assert result["rolled_back"] is True
    assert result["status"] == "HARD_SAFETY_REJECTED"
    torch.testing.assert_close(result["coordinates"], d1)


def test_damped_projection_decomposes_parallel_and_perpendicular_parts():
    source = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float64)
    config = JacobianBACConfig(damping_lambda=1.0e-9)
    system = build_constraint_system(source, config=config, **_constraints(2))
    delta = torch.tensor([[0.1, 0.2, 0.0], [-0.1, 0.2, 0.0]], dtype=torch.float64)
    parallel, diagnostics = _project_row_space(system, delta, config)
    perpendicular = delta.reshape(-1) - parallel
    assert diagnostics["projection_status"] == "PROJECTED"
    assert float(torch.linalg.vector_norm(system.jacobian @ perpendicular)) < 1.0e-8
    torch.testing.assert_close(parallel + perpendicular, delta.reshape(-1))


def test_candidate_b_removes_only_predicted_worsening_component():
    source = torch.tensor([[-1.4, 0.0, 0.0], [0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    d1 = torch.tensor([[-1.45, 0.0, 0.0], [0.05, 0.0, 0.0], [1.3, 0.0, 0.0]])
    result = jacobian_projection(
        source,
        d1,
        {},
        _Validity(),
        d1_accepted=False,
        **_constraints(3),
    )
    assert result["accepted"] is True
    assert result["status"] == "PROJECTION_ACCEPTED"
    assert result["diagnostics"]["violating_row_count"] == 1
    candidate_distances = torch.linalg.vector_norm(
        result["coordinates"][1:] - result["coordinates"][:-1], dim=-1
    )
    assert float(candidate_distances[0]) < 1.5
    assert float(candidate_distances[1]) < 1.4


def test_candidate_c_uses_frozen_finite_line_search_and_no_new_direction():
    source = torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]])
    d1 = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    result = trust_region_hybrid(
        source,
        d1,
        {},
        _Validity(reject_atom_move_above=0.15),
        config=JacobianBACConfig(
            max_molecule_rms_displacement=0.3,
            max_atom_displacement=0.3,
        ),
        **_constraints(2),
    )
    assert result["accepted"] is True
    assert result["diagnostics"]["selected_scale"] == 0.5
    expected = source + 0.5 * (d1 - source)
    torch.testing.assert_close(result["coordinates"], expected)
    assert [attempt["scale"] for attempt in result["diagnostics"]["attempts"]] == [
        1.0,
        0.5,
    ]


def test_candidate_c_rolls_back_to_source_when_every_scale_is_unsafe():
    source = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    d1 = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    result = trust_region_hybrid(
        source,
        d1,
        {},
        _Validity(reject_atom_move_above=0.0),
        **_constraints(2),
    )
    assert result["accepted"] is False
    assert result["status"] == "LINE_SEARCH_REJECTED"
    torch.testing.assert_close(result["coordinates"], source)


def test_nonfinite_and_combined_trust_fail_closed():
    source = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    nonfinite = source.clone()
    nonfinite[0, 0] = float("nan")
    result = jacobian_projection(
        source,
        nonfinite,
        {},
        _Validity(),
        d1_accepted=False,
        **_constraints(2),
    )
    assert result["status"] == "NONFINITE_INPUT"
    assert torch.isfinite(result["coordinates"]).all()

    d1 = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    trusted = posthoc_jacobian_correction(
        source,
        d1,
        {},
        _Validity(),
        alpha=1.0,
        d1_accepted=True,
        config=JacobianBACConfig(
            max_molecule_rms_displacement=0.01,
            max_atom_displacement=0.02,
        ),
        **_constraints(2),
    )
    assert trusted["status"] == "COMBINED_TRUST_REJECTED"
    torch.testing.assert_close(trusted["coordinates"], d1)


def test_guided_outputs_remove_rigid_motion_from_added_corrections():
    source = torch.tensor([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.8, 0.2, 0.0]])
    d1 = source + torch.tensor([0.1, -0.2, 0.3])
    result = posthoc_jacobian_correction(
        source,
        d1,
        {},
        _Validity(),
        alpha=0.5,
        d1_accepted=False,
        **_constraints(3),
    )
    assert result["rolled_back"] is True
    torch.testing.assert_close(result["coordinates"], d1)
    assert result["test_records_read"] == 0
    assert result["test_assets_opened"] is False
    assert result["frozen_holdout_records_opened"] == 0
