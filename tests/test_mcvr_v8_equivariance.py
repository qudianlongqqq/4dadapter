import torch

from etflow.ecir.v8_constraint_layer import DifferentiableMolecularConstraintLayer
from tests.v8_test_utils import batch, rotation


def test_constraint_layer_translation_and_rotation_equivariance():
    data = batch()
    layer = DifferentiableMolecularConstraintLayer({}, scales={"bond": 0.1, "angle": 0.1})
    prior = torch.randn_like(data.x_input) * 0.01
    confidence = torch.ones(4)
    reference = layer(data.x_input, prior, confidence, data)["delta_final"]
    shift = torch.tensor([3.0, -2.0, 0.4])
    translated = layer(data.x_input + shift, prior, confidence, data)["delta_final"]
    assert torch.allclose(reference, translated, atol=1e-6, rtol=1e-5)
    matrix = rotation(torch.float32)
    rotated = layer(data.x_input @ matrix.T, prior @ matrix.T, confidence, data)["delta_final"]
    assert torch.allclose(rotated, reference @ matrix.T, atol=2e-5, rtol=2e-4)
