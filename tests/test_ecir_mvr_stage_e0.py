from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torch_geometric.data import Data

from etflow.ecir.confidence_calibration import (
    CALIBRATION_DATA_SCHEMA, DIAGNOSTIC_ALL_ONE, MonotonicConfidenceCalibrator,
    build_calibration_manifest, calibrated_bond_residual, canonical_sha256,
    confidence_for_mode, file_sha256, fit_monotonic_calibrator,
    molecule_paired_bootstrap, optimal_scale_targets, split_calibration_molecules,
    strict_load_frozen_model, validate_calibration_frame, validate_calibration_manifest,
)
from etflow.ecir.mvr_model import MCVRModel
from scripts.evaluate_ecir_mvr_stage_e0 import infer_confidence_mode
from scripts.fit_ecir_mvr_stage_e0_calibrator import calibration_curve


def _model_config() -> dict:
    return {
        "hidden_dim": 24, "edge_hidden_dim": 24, "time_embedding_dim": 8,
        "num_layers": 1, "encoder_num_layers": 1, "error_embedding_dim": 8,
        "torsion_scale": 0.0, "high_flex_torsion_scale": 0.0,
        "torsion_gate_fixed_zero": True, "bond_head_enabled": True,
        "max_abs_bond_residual": 0.05,
    }


def _frame(manifest: dict) -> pd.DataFrame:
    rows = []
    split = {value: "fit" for value in manifest["fit_molecule_ids"]}
    split.update({value: "internal_check" for value in manifest["internal_check_molecule_ids"]})
    for index, molecule in enumerate(sorted(split)):
        rows.append({
            "schema_version": CALIBRATION_DATA_SCHEMA, "split": split[molecule],
            "molecule_id": molecule, "record_id": f"r{index}", "rollout_step": 1,
            "bond_index": 0, "confidence_logit": float(index - 2),
            "unattenuated_residual": 0.02, "target_residual": 0.01,
            "optimal_scale": 0.5, "weight": 1.0, "active_target": True,
            "outlier": True, "severe_outlier": False, "ring": bool(index % 2),
            "zero_target": False, "training_only": True, "test_records_read": 0,
        })
    return pd.DataFrame(rows)


def _manifest() -> dict:
    return build_calibration_manifest(
        ["m1", "m2", "m3", "m4"], checkpoint_sha256="a" * 64,
        frozen_identities={"medium": "frozen"}, seed=42,
    )


def test_slope_is_strictly_positive_and_only_two_parameters_are_learnable():
    calibrator = MonotonicConfidenceCalibrator()
    assert float(calibrator.a.detach()) > 0.0
    assert set(dict(calibrator.named_parameters())) == {"raw_a", "b"}


def test_calibrator_is_monotonic_in_original_logit():
    calibrator = MonotonicConfidenceCalibrator()
    calibrator.raw_a.data.fill_(-20.0)
    values = calibrator(torch.linspace(-20.0, 20.0, 100, dtype=torch.float64))
    assert bool((values[1:] > values[:-1]).all())


def test_initialization_strictly_degrades_to_original_confidence():
    logits = torch.tensor([-4.0, -1.0, 0.0, 2.0, 8.0], dtype=torch.float64)
    calibrator = MonotonicConfidenceCalibrator()
    assert float(calibrator.a.detach()) == 1.0
    assert float(calibrator.b.detach()) == 0.0
    assert torch.equal(calibrator(logits), torch.sigmoid(logits))
    assert torch.equal(confidence_for_mode(logits, mode="deployed"), torch.sigmoid(logits))
    residual = torch.linspace(-0.05, 0.05, len(logits), dtype=torch.float64)
    assert torch.equal(
        calibrated_bond_residual(residual, logits, calibrator),
        residual * torch.sigmoid(logits),
    )


def test_optimal_scale_soft_labels_follow_clipped_ratio():
    result = optimal_scale_targets([0.2, -0.2, 0.1], [0.1, -0.4, 0.2])
    assert result.tolist() == pytest.approx([0.5, 1.0, 1.0])


def test_optimal_scale_is_zero_for_wrong_sign():
    assert optimal_scale_targets([0.2, -0.2], [-0.1, 0.1]).tolist() == [0.0, 0.0]


def test_optimal_scale_is_stable_for_zero_prediction():
    result = optimal_scale_targets([0.0, 1.0e-12], [1.0, 1.0])
    assert torch.equal(result, torch.zeros_like(result))
    assert torch.isfinite(result).all()


def test_train_fit_check_are_molecule_disjoint_and_validation_test_are_rejected():
    fit, check = split_calibration_molecules([f"m{i}" for i in range(10)], seed=42)
    assert not set(fit) & set(check)
    manifest = _manifest()
    frame = _frame(manifest)
    validate_calibration_frame(frame, manifest)
    frame.loc[0, "split"] = "val"
    with pytest.raises(ValueError, match="forbidden split"):
        validate_calibration_frame(frame, manifest)
    frame.loc[0, "split"] = "fit"
    frame.loc[0, "test_records_read"] = 1
    with pytest.raises(ValueError, match="test-free"):
        validate_calibration_frame(frame, manifest)


def test_calibration_manifest_identity_detects_molecule_tampering():
    manifest = _manifest()
    validate_calibration_manifest(manifest)
    manifest["fit_molecule_ids"].append("tampered")
    with pytest.raises(ValueError, match="identity"):
        validate_calibration_manifest(manifest)


def test_checkpoint_load_is_strict_and_freezes_all_neural_weights(tmp_path: Path):
    model = MCVRModel(**_model_config())
    checkpoint = tmp_path / "checkpoint.ckpt"
    torch.save({"config": {"model": _model_config()}, "model_state_dict": model.state_dict()}, checkpoint)
    loaded, _ = strict_load_frozen_model(
        checkpoint, expected_sha256=file_sha256(checkpoint), device=torch.device("cpu")
    )
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    bad = tmp_path / "bad.ckpt"
    state = model.state_dict()
    state.pop(next(iter(state)))
    torch.save({"config": {"model": _model_config()}, "model_state_dict": state}, bad)
    with pytest.raises(RuntimeError):
        strict_load_frozen_model(bad, expected_sha256=file_sha256(bad), device=torch.device("cpu"))


def test_confidence_all_one_requires_diagnostic_oracle_marker():
    logits = torch.tensor([0.0, 1.0])
    with pytest.raises(PermissionError, match=DIAGNOSTIC_ALL_ONE):
        confidence_for_mode(logits, mode="confidence_all_one")
    assert torch.equal(
        confidence_for_mode(
            logits, mode="confidence_all_one", diagnostic_oracle_only=True
        ),
        torch.ones_like(logits),
    )


def test_torsion_gate_remains_exactly_zero():
    model = MCVRModel(**_model_config())
    edge_index = torch.tensor([[0, 1], [1, 0]])
    data = Data(
        num_nodes=2, node_attr=torch.randn(2, 10), edge_index=edge_index,
        edge_attr=torch.ones(2, 1), bond_is_in_ring=torch.zeros(2, dtype=torch.bool),
        rotatable_bond_index=torch.empty(2, 0, dtype=torch.long),
        atom_bond_influence_index=torch.empty(2, 0, dtype=torch.long),
        upstream_metadata=torch.zeros(1, 4), active_mode_mask=torch.zeros(1, 6),
    )
    output = model(data, torch.tensor([[0.0, 0, 0], [1.0, 0, 0]]), torch.tensor([0.5]))
    assert torch.equal(output["torsion_gate"], torch.zeros_like(output["torsion_gate"]))
    assert torch.equal(output["v_torsion_contribution"], torch.zeros_like(output["v_torsion_contribution"]))


def test_deployment_inference_function_does_not_read_target_or_reference_fields():
    tree = ast.parse(__import__("inspect").getsource(infer_confidence_mode))
    keys = {
        node.slice.value for node in ast.walk(tree)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    assert "minimal_target" not in keys
    assert "references" not in keys


def test_molecule_bootstrap_is_paired_and_deterministic():
    frame = pd.DataFrame({
        "molecule_id": ["a", "a", "b", "b"],
        "method": ["base", "candidate", "base", "candidate"],
        "metric": [2.0, 1.0, 4.0, 2.0],
    })
    first = molecule_paired_bootstrap(
        frame, candidate="candidate", baseline="base", metrics=["metric"], draws=100, seed=42
    )
    second = molecule_paired_bootstrap(
        frame, candidate="candidate", baseline="base", metrics=["metric"], draws=100, seed=42
    )
    assert first == second
    assert first["metric"]["mean"] == pytest.approx(-1.5)


def test_fit_uses_only_raw_a_and_b_and_never_validation_selection():
    manifest = _manifest()
    calibrator, metrics = fit_monotonic_calibrator(_frame(manifest), manifest, max_iter=5)
    assert set(dict(calibrator.named_parameters())) == {"raw_a", "b"}
    assert float(calibrator.a.detach()) > 0.0
    assert set(metrics) == {
        "initial_fit_loss", "final_fit_loss",
        "initial_internal_check_loss", "final_internal_check_loss",
    }


def test_calibration_curve_reports_all_preregistered_groups_even_when_empty():
    manifest = _manifest()
    frame = _frame(manifest)
    calibrator = MonotonicConfidenceCalibrator()
    curve = calibration_curve(frame, calibrator)
    expected = {
        "all", "active_target", "outlier", "severe_outlier",
        "ring", "nonring", "zero_target",
    }
    assert set(curve[curve.split.eq("fit")].group) == expected
    assert int(curve[(curve.split.eq("fit")) & (curve.group.eq("severe_outlier"))].bonds.iloc[0]) == 0


def test_stage_e0_config_preserves_d1b_and_contains_no_test_path():
    config = yaml.safe_load(Path("configs/ecir_mvr_stage_e0_confidence_calibration.yaml").read_text(encoding="utf-8"))
    assert config["checkpoint"]["sha256"] == "47189368db75c86f551a69cdbba5ef5f8c85a7e80929401aded309c246c5956d"
    assert config["checkpoint"]["neural_weights_frozen"] is True
    assert config["inference"]["torsion_gate_fixed_zero"] is True
    assert config["inference"]["diagnostic_control_label"] == DIAGNOSTIC_ALL_ONE
    assert not any("test" in str(value).lower() for value in config["data"].values())
