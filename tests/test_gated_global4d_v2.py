from __future__ import annotations

from pathlib import Path

import lightning
import pytest
import torch
import yaml
from torch_geometric.data import Data

from etflow.data.flexbond_datamodule import FlexBondOptimizerDataModule
from etflow.models.global4d_checkpoint import (
    load_global4d_for_inference,
    warm_start_global4d,
)
from etflow.models.global_coupled_4d_flow import GlobalCoupled4DFlowLightningModule
from scripts.sample_global_coupled_4d_flow import _validate_completed_run_identity


def _batch(with_reference: bool = False):
    value = {
        "x_init": torch.tensor(
            [[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0.0], [2.0, 1, 0.5]]
        ),
        "node_attr": torch.randn(4, 10),
        "edge_index": torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]
        ),
        "edge_attr": torch.zeros(6, 1),
        "rotatable_bond_index": torch.tensor([[1, 2], [2, 3]]),
        "num_rotatable_bonds": torch.tensor([2]),
        "batch": torch.zeros(4, dtype=torch.long),
    }
    if with_reference:
        value["x_ref_aligned"] = value["x_init"] + torch.tensor(
            [[0.0, 0.1, 0], [0.0, 0, 0.1], [0.1, 0, 0], [0.0, -0.1, 0.1]]
        )
    return value


def _small_arguments(**overrides):
    arguments = {
        "hidden_dim": 24,
        "edge_hidden_dim": 24,
        "time_embedding_dim": 16,
        "num_layers": 2,
    }
    arguments.update(overrides)
    return arguments


def _legacy_checkpoint(path: Path) -> Path:
    model = GlobalCoupled4DFlowLightningModule(**_small_arguments())
    legacy_hparams = dict(model.hparams)
    for key in (
        "fusion_mode",
        "joint_mode",
        "internal_beta",
        "gate_hidden_dim",
        "gate_init_bias",
        "gate_regularization_weight",
        "cartesian_weight",
        "gate_use_flexibility_features",
        "data_loader_config",
        "training_runtime_config",
    ):
        legacy_hparams.pop(key, None)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": legacy_hparams,
            "pytorch-lightning_version": lightning.__version__,
            "global_step": 5000,
        },
        path,
    )
    return path


def _gated_config():
    return {
        "model": {
            **_small_arguments(),
            "fusion_mode": "gated_additive",
            "internal_beta": 1.0,
            "gate_hidden_dim": 12,
            "gate_init_bias": -2.0,
        },
        "loss": {},
        "optimizer": {"lr": 2.0e-4},
        "time_sampling": {"t_min": 0.0, "t_max": 0.25},
    }


def test_old_checkpoint_without_fusion_mode_loads_as_exact_strict(tmp_path):
    path = _legacy_checkpoint(tmp_path / "legacy.ckpt")
    loaded = GlobalCoupled4DFlowLightningModule.load_from_checkpoint(
        path, map_location="cpu"
    )
    assert loaded.hparams.fusion_mode == "strict_orthogonal"
    output = loaded(_batch())
    torch.testing.assert_close(
        output["v_final"], output["v_residual"] + output["v_internal"]
    )
    assert not hasattr(loaded, "gate_head")


def test_missing_gate_requires_explicit_initialization(tmp_path):
    path = _legacy_checkpoint(tmp_path / "legacy.ckpt")
    with pytest.raises(RuntimeError, match="initialize_missing_gate"):
        warm_start_global4d(
            path, _gated_config(), initialize_missing_gate=False
        )
    model, report = warm_start_global4d(
        path, _gated_config(), initialize_missing_gate=True
    )
    assert model.hparams.fusion_mode == "gated_additive"
    assert report["initialized_gate_keys"]
    assert all(key.startswith("gate_head.") for key in report["missing_keys"])


def test_sampler_loader_rejects_implicit_strict_to_gated_switch(tmp_path):
    path = _legacy_checkpoint(tmp_path / "legacy.ckpt")
    with pytest.raises(RuntimeError, match="fusion semantics differ"):
        load_global4d_for_inference(
            path,
            _gated_config(),
            map_location="cpu",
            initialize_missing_gate=False,
        )
    model, report = load_global4d_for_inference(
        path,
        _gated_config(),
        map_location="cpu",
        initialize_missing_gate=True,
    )
    assert model.hparams.fusion_mode == "gated_additive"
    assert report["warm_started_in_memory"] is True


def test_graph_gate_broadcast_and_all_head_gradients_are_finite():
    torch.manual_seed(9)
    first = _batch(with_reference=True)
    second = _batch(with_reference=True)
    combined = {
        "x_init": torch.cat((first["x_init"], second["x_init"] + 3), dim=0),
        "x_ref_aligned": torch.cat(
            (first["x_ref_aligned"], second["x_ref_aligned"] + 3), dim=0
        ),
        "node_attr": torch.cat((first["node_attr"], second["node_attr"]), dim=0),
        "edge_index": torch.cat((first["edge_index"], second["edge_index"] + 4), dim=1),
        "edge_attr": torch.cat((first["edge_attr"], second["edge_attr"]), dim=0),
        "rotatable_bond_index": torch.cat(
            (first["rotatable_bond_index"], second["rotatable_bond_index"] + 4), dim=1
        ),
        "num_rotatable_bonds": torch.tensor([2, 2]),
        "batch": torch.tensor([0] * 4 + [1] * 4),
    }
    model = GlobalCoupled4DFlowLightningModule(
        **_small_arguments(fusion_mode="gated_additive", gate_hidden_dim=12)
    )
    output = model(combined)
    assert output["graph_gate"].shape == (2, 1)
    torch.testing.assert_close(output["atom_gate"], output["graph_gate"][combined["batch"]])
    assert bool(((output["graph_gate"] >= 0) & (output["graph_gate"] <= 1)).all())
    model.log_dict = lambda *args, **kwargs: None
    loss = model._shared_step(combined, "train")
    loss.backward()
    parameter_groups = {
        "backbone": list(model.backbone.layers.parameters()),
        "cartesian_head": [model.backbone.cartesian_layer_weights],
        "joint_head": list(model.backbone.joint_head.parameters()),
        "gate_head": list(model.gate_head.parameters()),
    }
    for name, parameters in parameter_groups.items():
        gradients = [parameter.grad for parameter in parameters]
        assert any(gradient is not None for gradient in gradients), name
        assert all(
            gradient is None or torch.isfinite(gradient).all()
            for gradient in gradients
        ), name


def test_new_configs_and_entry_point_cannot_default_to_strict():
    root = Path(__file__).resolve().parents[1]
    for name in (
        "configs/gated_global4d_v2_pilot.yaml",
        "configs/formal_large_gated_global4d_v2_seed42_200k.yaml",
    ):
        config = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        assert config["model"]["fusion_mode"] == "gated_additive"
        assert config["data"]["batch_size"] == 8
        assert config["trainer"]["accumulate_grad_batches"] == 1
    entry = (root / "scripts/train_gated_global4d_v2.py").read_text(encoding="utf-8")
    assert 'required_fusion_mode="gated_additive"' in entry


def test_dataloader_num_workers_zero_omits_prefetch_and_disables_persistence():
    module = FlexBondOptimizerDataModule(
        cache_dir="unused",
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )
    item = Data(
        num_nodes=2,
        x_init=torch.zeros(2, 3),
        x_ref_aligned=torch.ones(2, 3),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
    )
    loader = module._loader([item], shuffle=False)
    assert loader.num_workers == 0
    assert loader.persistent_workers is False
    assert loader.prefetch_factor is None
    assert module.resolved_loader_config() == {
        "batch_size": 8,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
    }


def test_checkpoint_hparams_record_real_batch_configuration(tmp_path):
    runtime = {
        "batch_size": 8,
        "accumulate_grad_batches": 1,
        "effective_batch_size": 8,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }
    model = GlobalCoupled4DFlowLightningModule(
        **_small_arguments(fusion_mode="gated_additive"),
        data_loader_config={key: runtime[key] for key in (
            "batch_size", "num_workers", "pin_memory", "persistent_workers", "prefetch_factor"
        )},
        training_runtime_config=runtime,
    )
    assert dict(model.hparams.training_runtime_config) == runtime
    train_source = (
        Path(__file__).resolve().parents[1]
        / "scripts/train_global_coupled_4d_flow.py"
    ).read_text(encoding="utf-8")
    for option in (
        "--batch_size",
        "--accumulate_grad_batches",
        "--num_workers",
        "--pin_memory",
        "--persistent_workers",
        "--prefetch_factor",
    ):
        assert option in train_source


def test_capacity_benchmark_covers_required_batches_and_full_training_step():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/benchmark_gated_global4d_batch_capacity.py"
    ).read_text(encoding="utf-8")
    assert "DEFAULT_BATCH_SIZES = (4, 8, 16, 32, 48, 64, 96, 128)" in source
    for operation in (
        "model._shared_step(batch, \"train\")",
        "loss.backward()",
        "optimizer.step()",
        "optimizer.zero_grad(set_to_none=True)",
        "torch.cuda.max_memory_allocated()",
        "torch.cuda.max_memory_reserved()",
        "GPUUtilizationSampler",
        "fixed_records_seen",
    ):
        assert operation in source
    assert 'COMPOSITIONS = ("low_complexity", "mixed", "high_complexity")' in source


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("fusion_mode", "strict_orthogonal"),
        ("internal_beta", 0.5),
        ("gate_override", 1.0),
        ("joint_mode", "torsion_only"),
        ("refinement_steps", 11),
        ("alpha", 0.25),
    ],
)
def test_completed_sample_identity_rejects_all_fusion_and_sampling_mismatches(
    field, changed
):
    identity = {
        "checkpoint_inference_sha256": "checkpoint",
        "checkpoint_global_step": 20,
        "config_sha256": "config",
        "manifest_sha256": "manifest",
        "split": "test",
        "alpha": 0.5,
        "refinement_steps": 10,
        "max_molecules": 5,
        "max_displacement": 0.1,
        "max_coordinate_norm": 1000.0,
        "joint_mode": "full_4d",
        "fusion_mode": "gated_additive",
        "internal_beta": 1.0,
        "gate_override": None,
        "initialize_missing_gate": False,
        "missing_gate_seed": None,
    }
    payload = {"persistence": {"run_identity": identity}}
    changed_identity = {**identity, field: changed}
    with pytest.raises(ValueError, match="different sampling command"):
        _validate_completed_run_identity(payload, changed_identity)
