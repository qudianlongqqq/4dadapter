import json

import pytest

from scripts.train_ecir_mvr_v8 import _read_graceful_stop_request


def _request():
    return {
        "schema_version": "mcvr-v8-graceful-stop-request-v1",
        "planned_original_total_steps": 200000,
        "user_requested_stop_step": 12500,
        "effective_batch": 64,
        "total_record_exposure": 800000,
        "equivalent_old_batch8_steps": 100000,
        "validation_mode": "FULL",
        "final_status": "MCVR_V8_FULL_V1_FORMAL_LARGE_12P5K_COMPLETED",
    }


def test_graceful_stop_request_binds_horizon_exposure_and_full_validation(tmp_path):
    path = tmp_path / "control" / "stop_request.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_request()), encoding="utf-8")
    loaded = _read_graceful_stop_request(
        tmp_path, current_step=10000, planned_total_steps=200000, effective_batch=64
    )
    assert loaded["user_requested_stop_step"] == 12500
    assert loaded["total_record_exposure"] == 800000


def test_graceful_stop_request_fails_closed_if_requested_step_has_passed(tmp_path):
    path = tmp_path / "control" / "stop_request.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_request()), encoding="utf-8")
    with pytest.raises(RuntimeError, match="arrived after"):
        _read_graceful_stop_request(
            tmp_path, current_step=12501, planned_total_steps=200000, effective_batch=64
        )


def test_graceful_stop_request_rejects_exposure_reinterpretation(tmp_path):
    request = _request()
    request["equivalent_old_batch8_steps"] = 99999
    path = tmp_path / "control" / "stop_request.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(RuntimeError, match="old-batch equivalence"):
        _read_graceful_stop_request(
            tmp_path, current_step=10000, planned_total_steps=200000, effective_batch=64
        )
