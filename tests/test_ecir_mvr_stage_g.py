from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch
import yaml

from etflow.ecir.bounded_residual_confidence import (
    BoundedResidualSignSafeCalibrator,
    STAGE_G_DEPLOYMENT_FEATURES,
    checkpoint_selection_priority,
    dataframe_stage_g_tensors,
    feature_view,
    select_stage_g_checkpoint,
    stage_g_loss,
    verify_stage_f_identity,
)
from etflow.ecir.confidence_calibration import file_sha256, strict_load_frozen_model
from etflow.ecir.mvr_model import MCVRModel
from scripts.build_ecir_mvr_stage_g_calibration_data import (
    build_parser as build_builder_parser,
    iter_builder_batches,
)
from scripts.evaluate_ecir_mvr_stage_g import infer_stage_g_mode
from scripts.fit_ecir_mvr_stage_g_calibrator import (
    build_parser as build_fitter_parser,
    resolve_dataset_residency,
)


def _features(count: int = 4, *, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    result = {
        name: torch.zeros(count, dtype=torch.float32, device=device)
        for name in STAGE_G_DEPLOYMENT_FEATURES
        if name not in {"bond_type_id", "element_pair_id", "ring", "aromatic"}
    }
    result.update(
        {
            "confidence_logit": torch.linspace(-1.0, 1.0, count, device=device),
            "original_confidence": torch.linspace(0.2, 0.8, count, device=device),
            "bond_type_id": torch.zeros(count, dtype=torch.long, device=device),
            "element_pair_id": torch.zeros(count, dtype=torch.long, device=device),
            "ring": torch.zeros(count, dtype=torch.bool, device=device),
            "aromatic": torch.zeros(count, dtype=torch.bool, device=device),
            "sign_safe_mask": torch.ones(count, dtype=torch.bool, device=device),
        }
    )
    return result


def _model_config() -> dict:
    return {
        "hidden_dim": 24,
        "edge_hidden_dim": 24,
        "time_embedding_dim": 8,
        "num_layers": 1,
        "encoder_num_layers": 1,
        "error_embedding_dim": 8,
        "torsion_scale": 0.0,
        "high_flex_torsion_scale": 0.0,
        "torsion_gate_fixed_zero": True,
        "bond_head_enabled": True,
        "max_abs_bond_residual": 0.05,
    }


def _config() -> dict:
    return yaml.safe_load(
        Path("configs/ecir_mvr_stage_g_bounded_residual.yaml").read_text(encoding="utf-8")
    )


def test_multiplier_initializes_exactly_at_identity():
    model = BoundedResidualSignSafeCalibrator()
    multiplier = model.multiplier(_features())
    assert torch.equal(multiplier, torch.ones_like(multiplier))


def test_multiplier_remains_within_configured_bounds():
    model = BoundedResidualSignSafeCalibrator(min_multiplier=0.5, max_multiplier=1.5)
    for parameter in model.parameters():
        parameter.data.fill_(100.0)
    multiplier = model.multiplier(_features())
    assert bool((multiplier >= 0.5).all())
    assert bool((multiplier <= 1.5).all())


def test_sign_safe_zero_forces_exact_zero_confidence():
    model = BoundedResidualSignSafeCalibrator()
    features = _features()
    features["sign_safe_mask"].zero_()
    assert torch.equal(model(features), torch.zeros(4))


def test_sign_safe_one_cannot_fully_disable_positive_original_confidence():
    model = BoundedResidualSignSafeCalibrator(min_multiplier=0.5, max_multiplier=1.5)
    features = _features()
    for parameter in model.feature_mlp.parameters():
        parameter.data.fill_(-100.0)
    confidence = model(features)
    assert bool((confidence >= 0.5 * features["original_confidence"]).all())
    assert bool((confidence > 0.0).all())


def test_zero_original_confidence_stays_zero_and_safe():
    model = BoundedResidualSignSafeCalibrator()
    features = _features()
    features["original_confidence"].zero_()
    assert torch.equal(model(features), torch.zeros(4))


def test_nan_and_infinite_inputs_produce_finite_outputs():
    model = BoundedResidualSignSafeCalibrator()
    features = _features()
    features["confidence_logit"][:] = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0])
    features["original_confidence"][:] = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.5])
    confidence, multiplier, base = model.forward_components(features)
    assert torch.isfinite(confidence).all()
    assert torch.isfinite(multiplier).all()
    assert torch.isfinite(base).all()


def test_d1b_strict_load_is_frozen(tmp_path: Path):
    model = MCVRModel(**_model_config())
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {"config": {"model": _model_config()}, "model_state_dict": model.state_dict()},
        checkpoint,
    )
    loaded, _ = strict_load_frozen_model(
        checkpoint, expected_sha256=file_sha256(checkpoint), device=torch.device("cpu")
    )
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    assert loaded.torsion_gate_fixed_zero is True


def test_stage_f_formal_artifacts_are_frozen_and_verified():
    config = _config()
    verify_stage_f_identity(config)
    expected = config["frozen_stage_f"]["sha256"]
    assert file_sha256(config["frozen_stage_f"]["validation_result"]) == expected["validation_result"]
    assert file_sha256(config["frozen_stage_f"]["calibrator"]) == expected["calibrator"]
    assert file_sha256(config["frozen_stage_f"]["training_history"]) == expected["training_history"]


def test_stage_g_output_is_isolated_from_stage_f():
    config = _config()
    assert Path(config["output_dir"]) == Path("diagnostics/ecir_mvr/stage_g")
    assert Path(config["output_dir"]) != Path("diagnostics/ecir_mvr/stage_f")


def test_batch_size_arguments_and_defaults_are_explicit():
    config = _config()
    assert config["builder"]["batch_size"] == 64
    assert config["training"]["batch_size"] == 65536
    assert config["linux_rtx5090"]["builder_batch_size"] == 128
    assert config["linux_rtx5090"]["calibrator_batch_size"] == 131072
    assert build_builder_parser().parse_args(["--builder-batch-size", "128"]).builder_batch_size == 128
    assert build_fitter_parser().parse_args(["--batch-size", "131072"]).batch_size == 131072


def test_builder_batch_16_and_64_preserve_record_order_and_identity():
    records = list(range(257))
    sixteen = [item for batch in iter_builder_batches(records, 16) for item in batch]
    sixty_four = [item for batch in iter_builder_batches(records, 64) for item in batch]
    assert sixteen == sixty_four == records


def test_calibrator_forward_is_batch_partition_invariant():
    model = BoundedResidualSignSafeCalibrator()
    features = _features(65536)
    whole = model(features)
    pieces = []
    for start in range(0, 65536, 4096):
        pieces.append(model({name: value[start : start + 4096] for name, value in features.items()}))
    assert torch.equal(whole, torch.cat(pieces))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_forward_agree_within_tolerance():
    cpu = BoundedResidualSignSafeCalibrator()
    cuda = BoundedResidualSignSafeCalibrator().cuda()
    cuda.load_state_dict(cpu.state_dict(), strict=True)
    expected = cpu(_features())
    actual = cuda(_features(device="cuda")).cpu()
    assert torch.allclose(expected, actual, atol=1.0e-6, rtol=1.0e-6)


def test_dataset_residency_validation_is_strict():
    tensors = {"value": torch.ones(8)}
    assert resolve_dataset_residency("cpu", device=torch.device("cpu"), tensors=tensors) == "cpu"
    assert resolve_dataset_residency("auto", device=torch.device("cpu"), tensors=tensors) == "cpu"
    with pytest.raises(ValueError, match="requires"):
        resolve_dataset_residency("cuda", device=torch.device("cpu"), tensors=tensors)


def test_profiling_is_default_off_and_configurable():
    config = _config()
    assert config["training"]["profile_cuda_memory"] is False
    assert build_fitter_parser().parse_args([]).profile_cuda_memory is None
    assert build_fitter_parser().parse_args(["--profile-cuda-memory"]).profile_cuda_memory is True


def test_collapsed_checkpoints_cannot_be_selected():
    collapsed = {
        "step": 100,
        "collapsed": True,
        "beneficial_repair_recall": 1.0,
        "optimal_scale_mae": 0.0,
        "wrong_sign_activation": 0.0,
        "false_positive_activation": 0.0,
        "multiplier_identity_error": 0.0,
        "cancellation_proxy": 0.0,
    }
    eligible = {**collapsed, "step": 200, "collapsed": False, "beneficial_repair_recall": 0.5}
    assert select_stage_g_checkpoint([collapsed, eligible])["step"] == 200
    assert select_stage_g_checkpoint([collapsed]) is None
    assert checkpoint_selection_priority(eligible) < checkpoint_selection_priority(collapsed)


def test_validation_inference_does_not_read_training_target_or_source_features():
    tree = ast.parse(inspect.getsource(infer_stage_g_mode))
    keys = {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    assert not {"minimal_target", "reference", "source", "severity"} & keys


def test_contiguous_tensor_conversion_separates_features_labels_and_masks():
    import pandas as pd

    row = {
        **{name: 0.0 for name in STAGE_G_DEPLOYMENT_FEATURES},
        "bond_type_id": 0,
        "element_pair_id": 0,
        "ring": False,
        "aromatic": False,
        "sign_safe_mask": True,
        "wrong_sign": False,
        "zero_target": False,
        "already_valid_unsafe": False,
        "beneficial": True,
        "molecule_code": 0,
        "optimal_scale": 0.5,
        "scale_weight": 1.0,
    }
    tensors = dataframe_stage_g_tensors(pd.DataFrame([row, row]))
    assert all(value.is_contiguous() for value in tensors.values())
    assert tensors["optimal_scale"].dtype == torch.float32
    assert tensors["sign_safe_mask"].dtype == torch.bool
    assert set(feature_view(tensors)) == set(STAGE_G_DEPLOYMENT_FEATURES) | {"sign_safe_mask"}


def test_beneficial_recall_and_identity_losses_are_active():
    confidence = torch.tensor([0.01, 0.5])
    multiplier = torch.tensor([0.5, 1.0])
    _, parts = stage_g_loss(
        confidence,
        multiplier,
        optimal_scale=torch.tensor([0.8, 0.5]),
        scale_weight=torch.ones(2),
        wrong_sign=torch.zeros(2, dtype=torch.bool),
        false_positive=torch.zeros(2, dtype=torch.bool),
        beneficial=torch.tensor([True, False]),
        molecule_ids=torch.tensor([0, 0]),
        lambda_wrong_sign=0.0,
        lambda_false_positive=0.0,
        lambda_overactivation=0.0,
        lambda_rank=0.0,
        lambda_beneficial_recall=1.0,
        lambda_multiplier_identity=1.0,
        beneficial_confidence_floor=0.1,
    )
    assert parts["beneficial_recall_loss"] > 0
    assert parts["multiplier_identity"] > 0


def test_windows_and_linux_launchers_are_portable_and_isolated():
    powershell = Path("scripts/run_ecir_mvr_stage_g.ps1").read_text(encoding="utf-8")
    linux = Path("scripts/run_ecir_mvr_stage_g.sh").read_text(encoding="utf-8")
    assert "BuilderBatchSize = 64" in powershell
    assert "BatchSize = 65536" in powershell
    assert "BUILDER_BATCH_SIZE=128" in linux
    assert "BATCH_SIZE=131072" in linux
    assert "set -euo pipefail" in linux
    assert "PYTHONUNBUFFERED=1" in linux
    assert "miniconda" not in linux.lower()
