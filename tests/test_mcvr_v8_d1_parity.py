import pytest
import torch

from etflow.ecir.mcvr_v8_full import MCVRV8FullRefiner
from etflow.ecir.v8_losses import MCVRV8Loss
from tests.v8_test_utils import D1_CHECKPOINT, D1_SHA256, batch


def test_constraint_disabled_is_exact_d1_prior():
    if not D1_CHECKPOINT.is_file():
        pytest.skip("frozen D1 artifact is not installed")
    model = MCVRV8FullRefiner.from_d1_checkpoint(
        D1_CHECKPOINT,
        expected_sha256=D1_SHA256,
        constraint_layer={"enabled": False},
        error_state_enabled=False,
        step_embedding_enabled=False,
        unroll_steps=1,
    ).eval()
    data = batch()
    t = torch.tensor([0.5])
    with torch.no_grad():
        prior = model.prior(data, data.x_input, t)["v_final"]
        output = model(data, data.x_input, t)
    assert output["d1_parity_mode"]
    assert torch.equal(output["delta_prior"], prior)
    assert torch.equal(output["delta_final"], prior)
    assert not any(parameter.requires_grad for parameter in model.error_state_head.parameters())
    assert not any(parameter.requires_grad for parameter in model.step_embedding.parameters())
    assert not any(group["name"] == "v8_new_heads" for group in model.parameter_groups(
        new_head_lr=2.0e-4,
        d1_head_lr=5.0e-5,
        d1_backbone_lr=2.0e-5,
    ))
    assert not bool(output["step_outputs"][0]["solver_failure"].sum())
    assert output["step_outputs"][0]["solver_status"] == ("DISABLED",)


def test_matched_d1_objective_has_no_v8_auxiliary_terms_and_trains_d1():
    if not D1_CHECKPOINT.is_file():
        pytest.skip("frozen D1 artifact is not installed")
    model = MCVRV8FullRefiner.from_d1_checkpoint(
        D1_CHECKPOINT,
        expected_sha256=D1_SHA256,
        constraint_layer={"enabled": False},
        error_state_enabled=False,
        step_embedding_enabled=False,
        unroll_steps=1,
    )
    data = batch()
    output = model(data, data.x_input, torch.tensor([0.5]))
    losses = MCVRV8Loss(
        {
            "target_weight": 100.0,
            "movement_weight": 0.1,
            "error_state_weight": 0.0,
            "confidence_regularization_weight": 0.0,
            "bond_weight": 0.0,
            "angle_weight": 0.0,
            "clash_weight": 0.0,
            "ring_weight": 0.0,
            "chirality_weight": 0.0,
            "step_consistency_weight": 0.0,
        }
    )(output, data)
    for name in (
        "error_state_loss",
        "confidence_regularization_loss",
        "bond_loss",
        "angle_loss",
        "clash_loss",
        "ring_loss",
        "chirality_loss",
        "step_consistency_loss",
        "solver_call_count",
    ):
        assert float(losses[name]) == 0.0
    losses["loss"].backward()
    backbone_gradients = [
        parameter.grad
        for name, parameter in model.prior.named_parameters()
        if name.startswith(("error_encoder.", "backbone.", "deterministic_embedding."))
    ]
    head_gradients = [
        parameter.grad
        for name, parameter in model.prior.named_parameters()
        if not name.startswith(("error_encoder.", "backbone.", "deterministic_embedding."))
    ]
    assert any(value is not None and bool(torch.count_nonzero(value)) for value in backbone_gradients)
    assert any(value is not None and bool(torch.count_nonzero(value)) for value in head_gradients)
