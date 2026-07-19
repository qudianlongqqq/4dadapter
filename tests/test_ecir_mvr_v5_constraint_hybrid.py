from __future__ import annotations

import inspect

import torch
from torch_geometric.data import Batch, Data

from etflow.ecir.mvr_model import MCVRModel
from etflow.ecir.mvr_v2_bac import MCVRBACModel, V2_D_BOND_ANGLE_CLASH
from etflow.ecir.mvr_v5_constraint_hybrid import (
    MCVRConstraintMultiHeadModel,
    MCVRNeuralJacobianHybrid,
)
from etflow.ecir.mvr_v5_constraint_loss import MCVRConstraintMultiHeadLoss


def _kwargs():
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


def _data(*, degenerate: bool = False) -> Data:
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [2.3, 0.4, 0.0],
            [0.3, 0.3, 0.2],
        ]
    )
    if degenerate:
        coordinates[1] = coordinates[0]
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
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
        active_bond_constraint_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        bond_allowed_range=torch.tensor([[0.9, 1.4, 0.1], [0.9, 1.4, 0.1], [0.7, 1.3, 0.1]]),
        active_angle_constraint_index=torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long),
        angle_allowed_range=torch.tensor([[1.2, 2.3, 0.1], [1.2, 2.3, 0.1]], dtype=torch.float32),
        deterministic_error_features=torch.zeros(1, 10),
        active_mode_mask=torch.tensor([[1, 1, 0, 1, 0, 0]], dtype=torch.float32),
        difficulty_target=torch.zeros(1),
        affected_atom_mask=torch.ones(4),
        x_init=coordinates,
        x_input=coordinates,
        x_target=coordinates
        + torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [-0.1, 0.05, 0.0],
                [0.05, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )


def _rotation():
    return torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_prototype_a_zero_initialized_heads_preserve_d1_exactly():
    torch.manual_seed(71)
    baseline = MCVRModel(**_kwargs()).eval()
    candidate = MCVRConstraintMultiHeadModel(**_kwargs()).eval()
    missing, unexpected = candidate.load_d1b_state_dict(baseline.state_dict(), strict=True)
    assert missing == []
    assert unexpected == []
    batch = _data()
    with torch.inference_mode():
        expected = baseline(batch, batch.x_input, torch.tensor([0.4]))
        actual = candidate(batch, batch.x_input, torch.tensor([0.4]))
    torch.testing.assert_close(actual["v_final"], expected["v_final"], rtol=0, atol=0)
    assert int(actual["unified_delta_count"]) == 1


def test_prototype_a_is_equivariant_normalized_and_finite():
    torch.manual_seed(72)
    model = MCVRConstraintMultiHeadModel(**_kwargs()).eval()
    with torch.no_grad():
        for head in (
            model.bond_constraint_head,
            model.angle_constraint_head,
            model.clash_constraint_head,
            model.multihead_fusion,
        ):
            head[-1].weight.fill_(0.01)
    batch = _data()
    rotation = _rotation()
    translation = torch.tensor([2.0, -1.0, 0.5])
    transformed = batch.x_input @ rotation.T + translation
    with torch.inference_mode():
        original = model(batch, batch.x_input, torch.tensor([0.2]))
        moved = model(batch, transformed, torch.tensor([0.2]))
    torch.testing.assert_close(
        moved["v_final"], original["v_final"] @ rotation.T, atol=2.0e-5, rtol=2.0e-5
    )
    assert torch.isfinite(original["v_final"]).all()
    assert bool((original["constraint_allocation"].sum(-1) <= 1.000001).all())
    for name in ("bond", "angle", "clash"):
        component = original[f"v_{name}_component"]
        rms = torch.sqrt(component.square().sum(-1).mean())
        assert float(rms) <= model.component_max_graph_rms + 1.0e-6


def test_prototype_a_specialized_loss_backpropagates_to_all_heads():
    torch.manual_seed(73)
    model = MCVRConstraintMultiHeadModel(**_kwargs()).train()
    loss = MCVRConstraintMultiHeadLoss()(model, _data())
    assert torch.isfinite(loss["loss"])
    loss["loss"].backward()
    prefixes = (
        "bond_constraint_head",
        "angle_constraint_head",
        "clash_constraint_head",
        "multihead_fusion",
    )
    for prefix in prefixes:
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.requires_grad
        ]
        assert gradients
        assert any(value is not None and torch.isfinite(value).all() for value in gradients), prefix


def test_prototype_b_adds_no_trainable_residual_parameters():
    prior = MCVRBACModel(**_kwargs(), bac_mode=V2_D_BOND_ANGLE_CLASH)
    hybrid = MCVRNeuralJacobianHybrid(prior)
    assert sum(parameter.numel() for parameter in hybrid.parameters()) == sum(
        parameter.numel() for parameter in prior.parameters()
    )
    assert all(name.startswith("prior.") for name in hybrid.state_dict())


def test_prototype_b_is_finite_equivariant_and_records_solver_rank():
    torch.manual_seed(74)
    prior = MCVRBACModel(**_kwargs(), bac_mode=V2_D_BOND_ANGLE_CLASH).eval()
    hybrid = MCVRNeuralJacobianHybrid(prior).eval()
    data = _data()
    batch = Batch.from_data_list([data])
    rotation = _rotation()
    translation = torch.tensor([1.0, 2.0, -0.5])
    transformed = batch.x_input @ rotation.T + translation
    with torch.inference_mode():
        original = hybrid(batch, batch.x_input, torch.tensor([0.5]))
        hybrid.reset_solver_statistics()
        moved = hybrid(batch, transformed, torch.tensor([0.5]))
    assert torch.isfinite(original["v_final"]).all()
    torch.testing.assert_close(
        moved["v_final"], original["v_final"] @ rotation.T, atol=3.0e-5, rtol=3.0e-5
    )
    summary = hybrid.solver_summary()
    assert summary["calls"] == 1
    assert summary["solver_failure_count"] == 0
    assert summary["effective_rank_mean"] >= 0
    assert summary["singular_value_max"] >= 0


def test_prototype_b_degenerate_geometry_fails_closed_without_nan():
    prior = MCVRBACModel(**_kwargs(), bac_mode=V2_D_BOND_ANGLE_CLASH).eval()
    hybrid = MCVRNeuralJacobianHybrid(prior).eval()
    batch = Batch.from_data_list([_data(degenerate=True)])
    with torch.inference_mode():
        output = hybrid(batch, batch.x_input, torch.tensor([0.5]))
    assert torch.isfinite(output["v_final"]).all()
    assert hybrid.solver_summary()["calls"] == 1


def test_prototype_b_source_forbids_inverse_and_arccos():
    source = inspect.getsource(MCVRNeuralJacobianHybrid)
    assert "torch.inverse" not in source
    assert "torch.linalg.inv" not in source
    assert "arccos" not in source
