from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from scripts import profile_ecir_mvr_formal_step as profile


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ecir_mvr_formal_large_d1b_base.yaml"


def test_profile_matrix_supports_worker_and_loader_variants():
    rows = profile.profile_matrix(
        [0, 2, 4, 8, 12], [2, 4], [True, False], [True, False]
    )
    zero_worker = [row for row in rows if row["num_workers"] == 0]
    assert zero_worker == [
        {
            "num_workers": 0,
            "prefetch_factor": None,
            "persistent_workers": False,
            "pin_memory": True,
        },
        {
            "num_workers": 0,
            "prefetch_factor": None,
            "persistent_workers": False,
            "pin_memory": False,
        },
    ]
    assert any(
        row == {
            "num_workers": 12,
            "prefetch_factor": 4,
            "persistent_workers": False,
            "pin_memory": False,
        }
        for row in rows
    )


def test_warmup_is_excluded_and_records_per_second_uses_optimizer_steps():
    rows = []
    for step in range(35):
        value = 1000.0 if step < 5 else 0.5
        rows.append(
            {
                **{field: value for field in profile.TIMING_FIELDS},
                "loss": float(step),
                "cpu_rss_mib": 100.0 + step,
            }
        )
    summary = profile.summarize_steps(rows, warmup_steps=5, micro_batch=64)
    assert summary["measured_optimizer_steps"] == 30
    assert summary["total_optimizer_step_seconds_mean"] == 0.5
    assert summary["records_per_second"] == 128.0
    assert summary["loss_start"] == 5.0


def test_profile_uses_real_train_primitives_and_has_no_checkpoint_path():
    source = inspect.getsource(profile.run_profile_setting)
    assert "_dataset" in source
    assert "DataLoader" in source
    assert "_build_training_components" in source
    assert "_forward_loss" in source
    assert "_backward_loss" in source
    assert "optimizer.step()" in source
    assert "Collater" in inspect.getsource(profile._probe_data_pipeline)
    module_source = Path(profile.__file__).read_text(encoding="utf-8")
    assert "atomic_torch_save" not in module_source
    assert "train_ecir_mvr_medium_rescue_v2.py" not in module_source
    assert '"train"' in source
    assert '"val"' not in source


def test_profile_config_rejects_scientific_change_and_test_asset():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    profile.validate_profile_config(config)
    changed = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    changed["model"]["hidden_dim"] = 128
    with pytest.raises(ValueError, match="model or loss"):
        profile.validate_profile_config(changed)
    test_named = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    test_named["data"]["test_sources"] = "/forbidden/test.parquet"
    with pytest.raises(ValueError, match="may not name a test"):
        profile.validate_profile_config(test_named)
    windows_test_path = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    windows_test_path["data"]["train_sources"] = r"D:\assets\test\train.parquet"
    with pytest.raises(ValueError, match="may not name a test"):
        profile.validate_profile_config(windows_test_path)


def test_output_directory_must_not_overlap_formal_assets(tmp_path):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    root = tmp_path / "formal"
    config["data"].update(
        {
            "root": str(root),
            "train_sources": str(root / "real_sources/train.parquet"),
            "val_sources": str(root / "real_sources/val.parquet"),
            "train_targets": str(root / "minimal_targets/train.parquet"),
            "val_targets": str(root / "minimal_targets/val.parquet"),
        }
    )
    with pytest.raises(ValueError, match="outside formal"):
        profile.validate_output_directory(root / "profile", config)
    assert profile.validate_output_directory(tmp_path / "reports", config) == (
        tmp_path / "reports"
    ).resolve()


def test_shared_gpu_default_gate_is_reused():
    processes = [{"pid": 123, "used_gpu_memory_mib": 1000.0}]
    assert profile.should_block_shared_gpu(processes, allow_shared_gpu=False)
    assert not profile.should_block_shared_gpu(processes, allow_shared_gpu=True)


def test_profile_has_fixed_same_environment_baseline_and_optimized_variants():
    assert profile.RUNTIME_VARIANTS == (
        {
            "name": "baseline",
            "formal_adapter_lru_size": 0,
            "precompute_training_topology": False,
        },
        {
            "name": "optimized",
            "formal_adapter_lru_size": 512,
            "precompute_training_topology": True,
        },
    )
    baseline = {"status": "PASS", "records_per_second": 100.0}
    optimized = {"status": "PASS", "records_per_second": 125.0}
    comparison = profile.comparison_result(baseline, optimized)
    assert comparison["records_per_second_speedup_ratio"] == 1.25


def test_cache_statistics_are_aggregated_without_entering_batch_schema():
    summary = profile.summarize_cache_statistics(
        [
            {
                "worker_id": 0,
                "pid": 10,
                "cache_hits": 3,
                "cache_misses": 1,
                "rdkit_adapter_build_count": 1,
                "topology_build_count": 4,
            },
            {
                "worker_id": 1,
                "pid": 11,
                "cache_hits": 2,
                "cache_misses": 2,
                "rdkit_adapter_build_count": 2,
                "topology_build_count": 4,
            },
        ]
    )
    assert summary["cache_hits"] == 5
    assert summary["cache_misses"] == 3
    assert summary["cache_hit_rate"] == 0.625
    assert summary["rdkit_adapter_build_count"] == 3
    assert summary["topology_build_count"] == 8
    assert len(summary["workers"]) == 2
