import pytest
import torch

from etflow.ecir.mcvr_v8_full import MCVRV8FullRefiner
from tests.v8_test_utils import D1_CHECKPOINT, D1_SHA256, batch


def test_constraint_disabled_is_exact_d1_prior():
    if not D1_CHECKPOINT.is_file():
        pytest.skip("frozen D1 artifact is not installed")
    model = MCVRV8FullRefiner.from_d1_checkpoint(
        D1_CHECKPOINT,
        expected_sha256=D1_SHA256,
        constraint_layer={"enabled": False},
        error_state_enabled=False,
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
