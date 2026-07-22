from __future__ import annotations

import json
from pathlib import Path

from scripts.train_ecir_mvr_v8 import (
    _materialize_ablation_stop_request,
    _read_graceful_stop_request,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/ecir_mvr_v8_full_v1_formal_large_200k.yaml"
CONFIGS = {
    "NO_CONSTRAINT": ROOT
    / "configs/ecir_mvr_v8_ablation_no_constraint_formal_large_200k.yaml",
    "NO_CONFIDENCE": ROOT
    / "configs/ecir_mvr_v8_ablation_no_confidence_formal_large_200k.yaml",
    "NO_ERROR_STATE": ROOT
    / "configs/ecir_mvr_v8_ablation_no_error_state_formal_large_200k.yaml",
    "NO_TYPE_NORMALIZATION": ROOT
    / "configs/ecir_mvr_v8_ablation_no_type_normalization_formal_large_200k.yaml",
}


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(child, path))
        return result
    return {prefix: value}


def test_formal_ablation_configs_have_only_preregistered_differences():
    baseline = _flatten(load_config(BASE))
    common_prefixes = ("ablation_protocol.", "ablation_registration.")
    common_paths = {
        "experiment_name",
        "long_run.parent_5k_checkpoint",
        "long_run.parent_5k_checkpoint_sha256",
        "long_run.resume_audit_required",
        "long_run.start_step",
    }
    factor_paths = {
        "NO_CONSTRAINT": {"constraint_layer.enabled"},
        "NO_CONFIDENCE": {
            "error_state.confidence_mode",
            "loss.confidence_regularization_weight",
        },
        "NO_ERROR_STATE": {
            "error_state.enabled",
            "error_state.confidence_mode",
            "loss.error_state_weight",
            "loss.confidence_regularization_weight",
        },
        "NO_TYPE_NORMALIZATION": {"type_normalization.enabled"},
    }
    for name, path in CONFIGS.items():
        candidate = _flatten(load_config(path))
        differing = {
            key
            for key in set(baseline) | set(candidate)
            if baseline.get(key, object()) != candidate.get(key, object())
        }
        unexpected = {
            key
            for key in differing
            if key not in common_paths
            and key not in factor_paths[name]
            and not key.startswith(common_prefixes)
        }
        assert not unexpected, (name, sorted(unexpected))


def test_each_formal_ablation_preserves_frozen_training_and_isolation():
    for name, path in CONFIGS.items():
        config = load_config(path)
        registration = config["ablation_registration"]
        assert registration["ablation_id"] == name
        assert config["seed"] == 43
        assert config["training"]["optimizer_steps"] == 200000
        assert config["training"]["effective_batch_size"] == 64
        assert config["training"]["gradient_accumulation_steps"] == 4
        assert registration["user_requested_stop_step"] == 12500
        assert registration["total_record_exposure"] == 800000
        assert registration["step10000_validation"] == "FAST"
        assert registration["step12500_validation"] == "FULL10K"
        assert config["data"]["allow_formal_test"] is False
        assert config["data"]["allow_frozen_holdout"] is False
        assert config["isolation"]["formal_test_records_read"] == 0
        assert config["isolation"]["frozen_holdout_records_read"] == 0


def test_single_factor_semantics_are_exact():
    no_constraint = load_config(CONFIGS["NO_CONSTRAINT"])
    assert no_constraint["constraint_layer"]["enabled"] is False
    assert no_constraint["error_state"]["enabled"] is True
    assert no_constraint["error_state"]["confidence_mode"] == "learned_bounded"
    assert no_constraint["constraint_layer"]["unroll_steps"] == 2
    assert no_constraint["loss"]["error_state_weight"] == 0.1

    no_confidence = load_config(CONFIGS["NO_CONFIDENCE"])
    assert no_confidence["error_state"]["enabled"] is True
    assert no_confidence["error_state"]["confidence_mode"] == "fixed"
    assert no_confidence["error_state"]["fixed_confidence"] == 1.0
    assert no_confidence["loss"]["confidence_regularization_weight"] == 0.0
    assert no_confidence["constraint_layer"]["enabled"] is True

    no_error = load_config(CONFIGS["NO_ERROR_STATE"])
    assert no_error["error_state"]["enabled"] is False
    assert no_error["error_state"]["confidence_mode"] == "fixed"
    assert no_error["loss"]["error_state_weight"] == 0.0
    assert no_error["loss"]["confidence_regularization_weight"] == 0.0
    assert no_error["constraint_layer"]["enabled"] is True
    for key in ("target_weight", "movement_weight", "bond_weight", "angle_weight"):
        assert no_error["loss"][key] > 0.0

    no_type = load_config(CONFIGS["NO_TYPE_NORMALIZATION"])
    assert no_type["type_normalization"]["enabled"] is False
    assert no_type["constraint_layer"]["enabled"] is True
    assert no_type["error_state"]["enabled"] is True
    assert no_type["error_state"]["confidence_mode"] == "learned_bounded"


def test_ablation_stop_request_exists_before_training(tmp_path):
    config = load_config(CONFIGS["NO_CONSTRAINT"])
    request = _materialize_ablation_stop_request(
        tmp_path, config, planned_total_steps=200000, effective_batch=64
    )
    assert request is not None
    path = tmp_path / "control/stop_request.json"
    assert path.is_file()
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["user_requested_stop_step"] == 12500
    assert persisted["request_origin"] == "ablation_resolved_config_before_first_optimizer_step"
    assert _read_graceful_stop_request(
        tmp_path,
        current_step=0,
        planned_total_steps=200000,
        effective_batch=64,
        expected_final_status=config["ablation_registration"]["final_status"],
    ) == persisted
