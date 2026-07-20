import torch

from etflow.ecir.v8_constraint_layer import DifferentiableMolecularConstraintLayer
from tests.v8_test_utils import batch


def test_constraint_forward_is_finite_and_inactive_limit_is_exact():
    data = batch()
    prior = torch.randn_like(data.x_input) * 0.01
    layer = DifferentiableMolecularConstraintLayer(
        {"solver_lambda_move": 0.0}, scales={"bond": 0.1, "angle": 0.1}
    )
    output = layer(data.x_input, prior, torch.ones(4), data)
    assert torch.isfinite(output["delta_final"]).all()
    empty = {"ptr": torch.tensor([0, 4])}
    inactive = layer(data.x_input, prior, torch.ones(4), empty)
    assert torch.equal(inactive["delta_final"], prior)
