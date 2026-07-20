from pathlib import Path


def test_cached_evaluator_does_not_import_or_execute_v8_model():
    source = Path("scripts/evaluate_ecir_mvr_v8_prediction_cache.py").read_text()
    assert "MCVRV8FullRefiner" not in source
    assert "model(" not in source
    assert "prediction_manifest" in source


def test_prediction_uses_inference_mode_and_writes_required_fields():
    source = Path("scripts/predict_ecir_mvr_v8_validation.py").read_text()
    assert "torch.inference_mode()" in source
    assert "legacy-compatible no_grad remains the parity default" in source
    for field in (
        "raw_coordinates",
        "safe_coordinates",
        "accepted",
        "rollback",
        "backtracking_decision",
        "solver_diagnostics",
    ):
        assert field in source
