from __future__ import annotations

import inspect

import torch
from torch_geometric.data import Batch, Data

from etflow.ecir.mvr_v2_bac import MCVRBACModel, V2_D_BOND_ANGLE_CLASH
from etflow.ecir.mvr_v7_constraint_specific import (
    MCVRConstraintSpecificHybrid,
    build_angle_constraint_system,
)


def _kwargs() -> dict[str, object]:
    return {
        "hidden_dim": 16,
        "edge_hidden_dim": 16,
        "time_embedding_dim": 8,
        "num_layers": 2,
        "encoder_num_layers": 2,
        "error_embedding_dim": 8,
        "bond_head_enabled": True,
        "bond_explicit_alpha": 1.0,
        "torsion_gate_fixed_zero": True,
    }


def _data(*, broad_angles: bool = False, coincident_clash: bool = False) -> Data:
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [2.3, 0.4, 0.0],
            [0.3, 0.3, 0.2],
        ],
        dtype=torch.float32,
    )
    if coincident_clash:
        coordinates[3] = coordinates[0]
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    angle_ranges = (
        torch.tensor([[0.0, 3.14159, 0.1], [0.0, 3.14159, 0.1]])
        if broad_angles
        else torch.tensor([[1.2, 1.5, 0.1], [1.2, 1.5, 0.1]])
    )
    return Data(
        num_nodes=4,
        node_attr=torch.tensor(
            [
                [6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        ),
        edge_index=edge_index,
        edge_attr=torch.ones(edge_index.size(1), 1),
        active_bond_constraint_index=torch.tensor(
            [[0, 1, 2], [1, 2, 3]], dtype=torch.long
        ),
        bond_allowed_range=torch.tensor(
            [[0.9, 1.4, 0.1], [0.9, 1.4, 0.1], [0.7, 1.3, 0.1]]
        ),
        active_angle_constraint_index=torch.tensor(
            [[0, 1], [1, 2], [2, 3]], dtype=torch.long
        ),
        angle_allowed_range=angle_ranges,
        deterministic_error_features=torch.zeros(1, 10),
        active_mode_mask=torch.tensor([[1, 1, 0, 1, 0, 0]], dtype=torch.float32),
        difficulty_target=torch.zeros(1),
        affected_atom_mask=torch.ones(4),
        x_init=coordinates,
        x_input=coordinates,
        x_target=coordinates,
    )


def _model(**kwargs: object) -> MCVRConstraintSpecificHybrid:
    prior = MCVRBACModel(**_kwargs(), bac_mode=V2_D_BOND_ANGLE_CLASH)
    with torch.no_grad():
        prior.rigid_edge[-1].bias.fill_(0.05)
    return MCVRConstraintSpecificHybrid(prior, **kwargs)


def test_v7_has_no_trainable_parameters_or_learned_gate() -> None:
    model = _model()
    assert list(model.parameters())
    assert all(not parameter.requires_grad for parameter in model.parameters())
    names = [name for name, _ in model.named_modules()]
    assert not any("adaptive_gate" in name or "fusion_gate" in name for name in names)


def test_v7_reduces_to_d1_when_angle_and_clash_are_inactive() -> None:
    model = _model(clash_allowed_contact=0.0).eval()
    batch = Batch.from_data_list([_data(broad_angles=True)])
    with torch.inference_mode():
        baseline = model.prior(batch, batch.x_input, torch.tensor([0.5]))
        output = model(batch, batch.x_input, torch.tensor([0.5]))
    torch.testing.assert_close(output["v_final"], baseline["v_final"], atol=2.0e-6, rtol=2.0e-6)
    assert float(output["constraint_alpha_angle"][0]) == 0.0
    assert float(output["constraint_alpha_clash"][0]) == 0.0


def test_v7_angle_system_and_update_are_angle_only_bounded_and_finite() -> None:
    model = _model(clash_allowed_contact=0.0).eval()
    batch = Batch.from_data_list([_data()])
    system = build_angle_constraint_system(
        batch.x_input.double(),
        batch.active_angle_constraint_index,
        batch.angle_allowed_range,
        model.jacobian_config,
    )
    assert system.residual.numel() > 0
    assert set(system.constraint_types) == {"angle"}
    assert system.counts["active_bond"] == 0
    assert system.counts["active_clash"] == 0
    with torch.inference_mode():
        output = model(batch, batch.x_input, torch.tensor([0.5]))
    angle = output["v_angle_jacobian_coordinate"]
    assert torch.isfinite(angle).all()
    assert float(torch.linalg.vector_norm(angle, dim=-1).max()) <= 0.020001
    assert float(torch.sqrt(angle.square().sum(-1).mean())) <= 0.010001
    assert model.angle_solver_summary()["solver_failure_count"] == 0


def test_v7_clash_repulsion_is_equivariant_and_separates_pair() -> None:
    model = _model().eval()
    model.trace_enabled = False
    batch = Batch.from_data_list([_data(broad_angles=True)])
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([2.0, -1.0, 0.5])
    transformed = batch.x_input @ rotation.T + translation
    with torch.inference_mode():
        original = model(batch, batch.x_input, torch.tensor([0.3]))
        moved = model(batch, transformed, torch.tensor([0.3]))
    clash = original["v_clash_repulsion_coordinate"]
    assert torch.linalg.vector_norm(clash) > 0
    torch.testing.assert_close(
        moved["v_clash_repulsion_coordinate"],
        clash @ rotation.T,
        atol=3.0e-6,
        rtol=3.0e-6,
    )
    before = torch.linalg.vector_norm(batch.x_input[0] - batch.x_input[3])
    after_coordinates = batch.x_input + clash
    after = torch.linalg.vector_norm(after_coordinates[0] - after_coordinates[3])
    assert after > before


def test_v7_coincident_clash_fails_closed_without_fixed_axis_update() -> None:
    model = _model().eval()
    batch = Batch.from_data_list([_data(broad_angles=True, coincident_clash=True)])
    with torch.inference_mode():
        output = model(batch, batch.x_input, torch.tensor([0.4]))
    assert torch.isfinite(output["v_final"]).all()
    torch.testing.assert_close(
        output["v_clash_repulsion_coordinate"],
        torch.zeros_like(output["v_clash_repulsion_coordinate"]),
    )
    assert model.component_summary()["degenerate_clash_count"] > 0


def test_v7_fusion_respects_original_d1_trust_limits() -> None:
    model = _model().eval()
    batch = Batch.from_data_list([_data()])
    with torch.inference_mode():
        output = model(batch, batch.x_input, torch.tensor([0.2]))
    velocity = output["v_trust_clipped"]
    assert float(torch.linalg.vector_norm(velocity, dim=-1).max()) <= 0.120001
    assert float(torch.sqrt(velocity.square().sum(-1).mean())) <= 0.060001
    assert torch.isfinite(output["v_final"]).all()


def test_v7_prior_state_strict_roundtrip_and_source_forbids_inverse() -> None:
    model = _model()
    clone = _model()
    incompatible = clone.prior.load_state_dict(model.prior.state_dict(), strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    source = inspect.getsource(MCVRConstraintSpecificHybrid)
    assert "torch.inverse" not in source
    assert "torch.linalg.inv" not in source
    assert "arccos" not in source
    assert "adaptive_gate" not in source
