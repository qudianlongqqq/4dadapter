import torch
import pytest

from etflow.ecir.mcvr_v8_full import MCVRV8FullRefiner
from etflow.ecir.v8_constraint_layer import DifferentiableMolecularConstraintLayer
from tests.v8_test_utils import D1_CHECKPOINT, D1_SHA256, batch


def test_solver_backward_reaches_prior_and_coordinates():
    data = batch()
    coordinates = data.x_input.clone().requires_grad_(True)
    prior = (torch.randn_like(coordinates) * 0.01).requires_grad_(True)
    layer = DifferentiableMolecularConstraintLayer({}, scales={"bond": 0.1, "angle": 0.1})
    delta = layer(coordinates, prior, torch.ones(4), data)["delta_final"]
    delta.square().sum().backward()
    assert prior.grad is not None and torch.isfinite(prior.grad).all()
    assert coordinates.grad is not None and torch.isfinite(coordinates.grad).all()
    assert float(prior.grad.abs().sum()) > 0


def test_gradient_crosses_solver_into_d1_head_and_egnn_backbone():
    if not D1_CHECKPOINT.is_file():
        pytest.skip("frozen D1 artifact is not installed")
    model = MCVRV8FullRefiner.from_d1_checkpoint(
        D1_CHECKPOINT,
        expected_sha256=D1_SHA256,
        constraint_layer={"solve_dtype": "float64"},
        residual_scales={"bond": 0.1, "angle": 0.1},
        unroll_steps=1,
    )
    data = batch()
    output = model(data, data.x_input, torch.tensor([0.5]))
    output["x_final"].square().sum().backward()
    head_gradients = [
        parameter.grad
        for name, parameter in model.prior.named_parameters()
        if name.startswith("rigid_") and parameter.grad is not None
    ]
    backbone_gradients = [
        parameter.grad
        for name, parameter in model.prior.named_parameters()
        if name.startswith("backbone.") and parameter.grad is not None
    ]
    assert any(float(value.abs().sum()) > 0 for value in head_gradients)
    assert any(float(value.abs().sum()) > 0 for value in backbone_gradients)
