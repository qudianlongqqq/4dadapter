import pytest
import torch

from etflow.ecir.v8_validation_cache import compare_prediction_records


def _record(value=1.0):
    coordinates = torch.tensor([[value, 0.0, 0.0]])
    return {
        "sample_id": "s1",
        "molecule_id": "m1",
        "source_coordinate_sha256": "source",
        "accepted": True,
        "rollback": False,
        "backtracking_decision": {"selected_scale": 1.0},
        "raw_coordinates": coordinates,
        "safe_coordinates": coordinates,
    }


def test_continuous_parity_contract_reports_differences():
    result = compare_prediction_records([_record()], [_record(1.0 + 1.0e-7)])
    assert result["status"] == "PARITY_OK"
    assert result["discrete_bitwise_equal"] is True
    assert result["max_absolute_difference"] > 0.0


def test_discrete_parity_is_exact():
    changed = _record()
    changed["accepted"] = False
    with pytest.raises(RuntimeError, match="accepted"):
        compare_prediction_records([_record()], [changed])
