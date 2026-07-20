import json

from etflow.ecir.v8_validation_cache import ValidationStatus


def test_live_status_updates_and_failure_are_atomic(tmp_path):
    path = tmp_path / "status.json"
    tracker = ValidationStatus.start(
        path, phase="FULL_PREDICTING", training_step=5000, validation_mode="FULL"
    )
    tracker.update(current_validation_record=100, prediction_chunks_completed=1)
    assert json.loads(path.read_text())["current_validation_record"] == 100
    tracker.fail("boom")
    failed = json.loads(path.read_text())
    assert failed["status"] == "FAILED_CLOSED"
    assert failed["error"] == "boom"
