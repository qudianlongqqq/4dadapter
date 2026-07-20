import pytest

from etflow.ecir.mcvr_v8_full import MCVRV8FullRefiner
from tests.v8_test_utils import D1_CHECKPOINT, D1_SHA256


def test_d1_checkpoint_strict_load():
    if not D1_CHECKPOINT.is_file():
        pytest.skip("frozen D1 artifact is not installed")
    model = MCVRV8FullRefiner.from_d1_checkpoint(
        D1_CHECKPOINT,
        expected_sha256=D1_SHA256,
        constraint_layer={"enabled": False},
        error_state_enabled=False,
        unroll_steps=1,
    )
    assert model.d1_checkpoint_identity["strict_load"] is True
    assert model.d1_checkpoint_identity["step"] == 25000
