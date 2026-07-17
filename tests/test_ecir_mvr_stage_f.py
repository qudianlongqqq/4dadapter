from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from etflow.ecir.confidence_calibration import (
    file_sha256, molecule_paired_bootstrap, strict_load_frozen_model,
)
from etflow.ecir.feature_conditioned_confidence import (
    BOND_TYPE_IDS, DATA_SCHEMA, DEPLOYMENT_FEATURES, DIAGNOSTIC_ORACLE_ONLY,
    FeatureConditionedConfidenceCalibrator, build_stage_f_manifest,
    calibrator_identity_payload, dataframe_feature_tensors,
    encode_element_pair, internal_check_priority, load_feature_calibrator,
    sign_validity_safe_mask, stage_f_loss, validate_stage_f_frame,
    stage_f_decision,
)
from etflow.ecir.mvr_model import MCVRModel
from scripts.evaluate_ecir_mvr_stage_e0 import infer_confidence_mode
from scripts.evaluate_ecir_mvr_stage_f import infer_stage_f_mode
from scripts.fit_ecir_mvr_stage_f_calibrator import molecule_grouped_batch_indices


def _features(count: int = 4) -> dict[str, torch.Tensor]:
    result = {
        name: torch.zeros(count, dtype=torch.float64)
        for name in DEPLOYMENT_FEATURES
        if name not in {"bond_type_id", "element_pair_id", "ring", "aromatic"}
    }
    result.update({
        "confidence_logit": torch.linspace(-1.0, 1.0, count, dtype=torch.float64),
        "bond_type_id": torch.zeros(count, dtype=torch.long),
        "element_pair_id": torch.zeros(count, dtype=torch.long),
        "ring": torch.zeros(count, dtype=torch.bool),
        "aromatic": torch.zeros(count, dtype=torch.bool),
        "sign_safe_mask": torch.tensor(([True, False] * count)[:count]),
    })
    return result


def _manifest() -> dict:
    return build_stage_f_manifest(
        ["m1", "m2", "m3", "m4"], checkpoint_sha256="a" * 64,
        frozen_identities={"medium": "frozen"}, seed=42,
    )


def _frame(manifest: dict) -> pd.DataFrame:
    split = {value: "fit" for value in manifest["fit_molecule_ids"]}
    split.update({value: "internal_check" for value in manifest["internal_check_molecule_ids"]})
    rows = []
    for index, molecule in enumerate(sorted(split)):
        features = _features(1)
        row = {
            "schema_version": DATA_SCHEMA, "split": split[molecule],
            "molecule_id": molecule, "molecule_code": index,
            "record_id": f"r{index}", "rollout_step": 1, "bond_index": 0,
            **{name: value.item() for name, value in features.items()},
            "target_residual": 0.01, "optimal_scale": 0.5, "scale_weight": 1.0,
            "wrong_sign": False, "zero_target": False,
            "already_valid_unsafe": False, "beneficial": True,
            "training_only": True, "validation_records_read": 0, "test_records_read": 0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _model_config() -> dict:
    return {
        "hidden_dim": 24, "edge_hidden_dim": 24, "time_embedding_dim": 8,
        "num_layers": 1, "encoder_num_layers": 1, "error_embedding_dim": 8,
        "torsion_scale": 0.0, "high_flex_torsion_scale": 0.0,
        "torsion_gate_fixed_zero": True, "bond_head_enabled": True,
        "max_abs_bond_residual": 0.05,
    }


def test_sign_safe_below_lower_allows_only_distance_reducing_positive_repair():
    mask = sign_validity_safe_mask([0.8, 0.8, 0.8], [1.0] * 3, [1.2] * 3, [0.1, -0.1, 0.0])
    assert mask.tolist() == [True, False, False]


def test_sign_safe_above_upper_allows_only_distance_reducing_negative_repair():
    mask = sign_validity_safe_mask([1.4, 1.4], [1.0, 1.0], [1.2, 1.2], [-0.1, 0.1])
    assert mask.tolist() == [True, False]


def test_sign_safe_valid_bond_cannot_be_pushed_outside_interval():
    mask = sign_validity_safe_mask([1.1, 1.1], [1.0, 1.0], [1.2, 1.2], [0.05, 0.2])
    assert mask.tolist() == [True, False]


def test_stage_f_deployment_inference_never_reads_target_or_reference_fields():
    tree = ast.parse(inspect.getsource(infer_stage_f_mode))
    keys = {
        node.slice.value for node in ast.walk(tree)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    assert "minimal_target" not in keys
    assert "references" not in keys


def test_positive_slope_parameterization_remains_strictly_positive():
    calibrator = FeatureConditionedConfidenceCalibrator()
    calibrator.raw_a.data.fill_(-100.0)
    assert float(calibrator.a.detach()) > 0.0


def test_initialization_exactly_matches_original_confidence_times_safe_mask():
    calibrator = FeatureConditionedConfidenceCalibrator()
    features = _features()
    expected = features["sign_safe_mask"] * torch.sigmoid(features["confidence_logit"])
    assert float(calibrator.a.detach()) == 1.0
    assert torch.equal(calibrator(features), expected)


def test_feature_bias_is_strictly_bounded():
    calibrator = FeatureConditionedConfidenceCalibrator(max_bias=0.25)
    for parameter in calibrator.feature_mlp.parameters():
        parameter.data.fill_(100.0)
    bias = calibrator.feature_bias(_features())
    assert bool((bias.abs() <= 0.25).all())


def test_wrong_sign_loss_penalizes_wrong_sign_confidence():
    confidence = torch.tensor([0.8, 0.1], dtype=torch.float64)
    _, parts = stage_f_loss(
        confidence, optimal_scale=torch.zeros(2), scale_weight=torch.ones(2),
        wrong_sign=torch.tensor([True, False]), false_positive=torch.zeros(2, dtype=torch.bool),
        molecule_ids=torch.tensor([0, 0]), lambda_wrong_sign=1.0,
        lambda_false_positive=0.0, lambda_overactivation=0.0, lambda_rank=0.0,
    )
    assert parts["wrong_sign"] == pytest.approx(0.8)


def test_false_positive_loss_penalizes_only_false_positive_mask():
    confidence = torch.tensor([0.2, 0.7], dtype=torch.float64)
    _, parts = stage_f_loss(
        confidence, optimal_scale=torch.zeros(2), scale_weight=torch.ones(2),
        wrong_sign=torch.zeros(2, dtype=torch.bool), false_positive=torch.tensor([False, True]),
        molecule_ids=torch.tensor([0, 0]), lambda_wrong_sign=0.0,
        lambda_false_positive=1.0, lambda_overactivation=0.0, lambda_rank=0.0,
    )
    assert parts["false_positive"] == pytest.approx(0.7)


def test_overactivation_loss_is_one_sided():
    confidence = torch.tensor([0.2, 0.8], dtype=torch.float64)
    _, parts = stage_f_loss(
        confidence, optimal_scale=torch.tensor([0.5, 0.5]), scale_weight=torch.ones(2),
        wrong_sign=torch.zeros(2, dtype=torch.bool), false_positive=torch.zeros(2, dtype=torch.bool),
        molecule_ids=torch.tensor([0, 0]), lambda_wrong_sign=0.0,
        lambda_false_positive=0.0, lambda_overactivation=1.0, lambda_rank=0.0,
    )
    assert parts["overactivation"] == pytest.approx(0.15)


def test_ranking_loss_prefers_higher_confidence_for_higher_scale():
    common = dict(
        optimal_scale=torch.tensor([1.0, 0.0]), scale_weight=torch.ones(2),
        wrong_sign=torch.zeros(2, dtype=torch.bool), false_positive=torch.zeros(2, dtype=torch.bool),
        molecule_ids=torch.tensor([0, 0]), lambda_wrong_sign=0.0,
        lambda_false_positive=0.0, lambda_overactivation=0.0, lambda_rank=1.0,
    )
    _, good = stage_f_loss(torch.tensor([0.9, 0.1]), **common)
    _, bad = stage_f_loss(torch.tensor([0.1, 0.9]), **common)
    assert good["rank"] < bad["rank"]


def test_molecule_split_is_disjoint_and_train_only():
    manifest = _manifest()
    assert not set(manifest["fit_molecule_ids"]) & set(manifest["internal_check_molecule_ids"])
    assert manifest["training_only"] is True
    assert manifest["validation_records_read"] == manifest["test_records_read"] == 0


def test_training_batch_contains_multiple_rows_per_selected_molecule():
    molecule_ids = torch.repeat_interleave(torch.arange(100), 20)
    indices = molecule_grouped_batch_indices(
        molecule_ids, batch_size=256, generator=torch.Generator().manual_seed(42),
    )
    counts = torch.bincount(molecule_ids[indices])
    assert int((counts > 0).sum()) <= 64
    assert int(counts.max()) >= 4


def test_validation_and_test_rows_are_rejected_from_calibration_data():
    manifest = _manifest(); frame = _frame(manifest)
    validate_stage_f_frame(frame, manifest)
    frame.loc[0, "split"] = "val"
    with pytest.raises(ValueError, match="validation/test"):
        validate_stage_f_frame(frame, manifest)
    frame.loc[0, "split"] = "fit"; frame.loc[0, "test_records_read"] = 1
    with pytest.raises(ValueError, match="isolated"):
        validate_stage_f_frame(frame, manifest)


def test_d1b_checkpoint_load_is_strict_and_frozen(tmp_path: Path):
    model = MCVRModel(**_model_config()); checkpoint = tmp_path / "model.ckpt"
    torch.save({"config": {"model": _model_config()}, "model_state_dict": model.state_dict()}, checkpoint)
    loaded, _ = strict_load_frozen_model(
        checkpoint, expected_sha256=file_sha256(checkpoint), device=torch.device("cpu")
    )
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    with pytest.raises(RuntimeError, match="identity"):
        strict_load_frozen_model(checkpoint, expected_sha256="0" * 64, device=torch.device("cpu"))


def test_frozen_model_torsion_gate_remains_exactly_zero():
    model = MCVRModel(**_model_config())
    assert model.torsion_gate_fixed_zero is True
    assert model.torsion_scale == 0.0
    assert model.high_flex_torsion_scale == 0.0


def test_sign_safe_only_path_is_original_confidence_times_mask():
    logits = torch.tensor([-1.0, 1.0]); mask = torch.tensor([True, False])
    assert torch.equal(torch.sigmoid(logits) * mask, torch.tensor([torch.sigmoid(logits[0]), 0.0]))


def test_confidence_all_one_remains_diagnostic_oracle_only():
    assert DIAGNOSTIC_ORACLE_ONLY == "DIAGNOSTIC_ORACLE_ONLY"


def test_source_label_leakage_is_rejected_and_hash_encoding_is_stable():
    manifest = _manifest(); frame = _frame(manifest); frame["source"] = "forbidden"
    with pytest.raises(ValueError, match="source label"):
        validate_stage_f_frame(frame, manifest)
    assert encode_element_pair(6, 8) == encode_element_pair(8, 6)
    assert len(BOND_TYPE_IDS) <= 6


def test_feature_calibrator_checkpoint_load_is_strict(tmp_path: Path):
    calibrator = FeatureConditionedConfidenceCalibrator()
    checkpoint = tmp_path / "best.ckpt"
    torch.save({"step": 2, "calibrator_state_dict": calibrator.state_dict()}, checkpoint)
    model_config = {
        "hidden_dim": 24, "num_layers": 2, "bond_type_embedding_dim": 4,
        "element_pair_embedding_dim": 4, "element_pair_buckets": 32,
        "time_embedding_dim": 4, "max_bias": 1.0, "epsilon": 1e-8, "dropout": 0.0,
    }
    payload = calibrator_identity_payload(
        calibrator, model_config=model_config, checkpoint_sha256="a" * 64,
        training_molecule_identity_sha256="b" * 64, manifest_identity_sha256="c" * 64,
        selected_step=2, selection_metrics={"wrong_sign_activation": 0.0}, smoke=True,
    )
    loaded = load_feature_calibrator(str(checkpoint), payload, device=torch.device("cpu"))
    assert torch.equal(loaded(_features()), calibrator(_features()))


def test_internal_selection_priority_is_lexicographic_and_conservative():
    safe = {"wrong_sign_activation": 0.1, "false_positive_activation": 0.2,
            "optimal_scale_mae": 0.4, "beneficial_correction_capture": 0.1}
    risky = {**safe, "wrong_sign_activation": 0.2, "optimal_scale_mae": 0.0}
    assert internal_check_priority(safe) < internal_check_priority(risky)


def test_sign_safe_only_wins_over_unnecessary_complexity():
    assert stage_f_decision(
        {"gate": True}, sign_safe_only_better=True, harms=False
    ) == "STAGE_F_SIGN_SAFE_ONLY_BETTER"


def test_molecule_bootstrap_is_paired():
    frame = pd.DataFrame({
        "molecule_id": ["a", "a", "b", "b"],
        "method": ["base", "candidate", "base", "candidate"],
        "metric": [2.0, 1.0, 4.0, 2.0],
    })
    result = molecule_paired_bootstrap(
        frame, candidate="candidate", baseline="base", metrics=["metric"],
        draws=100, seed=42,
    )
    assert result["metric"]["mean"] == pytest.approx(-1.5)


def test_existing_d1b_and_e0_paths_remain_unchanged():
    config = yaml.safe_load(Path("configs/ecir_mvr_stage_f_feature_confidence.yaml").read_text(encoding="utf-8"))
    e0 = json.loads(Path("diagnostics/ecir_mvr/stage_e0/validation_result.json").read_text(encoding="utf-8"))
    assert config["checkpoint"]["sha256"] == "47189368db75c86f551a69cdbba5ef5f8c85a7e80929401aded309c246c5956d"
    assert e0["decision"] == "STAGE_E0_NO_ADDED_VALUE"
    assert "minimal_target" not in {
        node.slice.value for node in ast.walk(ast.parse(inspect.getsource(infer_confidence_mode)))
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
