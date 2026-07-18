from __future__ import annotations

import json
import inspect
from pathlib import Path

import pytest
import torch
import yaml

from scripts import preflight_ecir_mvr_formal_large as preflight
from scripts import train_ecir_mvr_medium_rescue_v2 as training


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ecir_mvr_formal_large_d1b_base.yaml"


def _gpu():
    return {
        "gpu_index": 0,
        "gpu_uuid": "GPU-test",
        "gpu_name": "RTX 4090",
        "memory_total_mib": 49152.0,
        "memory_used_mib": 22000.0,
        "memory_free_mib": 27152.0,
        "gpu_utilization_percent": 50.0,
        "power_draw_w": 300.0,
        "temperature_c": 60.0,
        "driver_version": "test",
    }


def _phase_result(records_per_second=100.0):
    return {
        "avg_optimizer_step_time_seconds": 0.64,
        "median_optimizer_step_time_seconds": 0.63,
        "p95_optimizer_step_time_seconds": 0.70,
        "avg_dataloader_time_seconds": 0.05,
        "avg_forward_time_seconds": 0.20,
        "avg_backward_time_seconds": 0.30,
        "avg_optimizer_time_seconds": 0.09,
        "loss_start": 1.0,
        "loss_end": 0.9,
        "loss_min": 0.9,
        "loss_max": 1.0,
        "gpu_utilization_mean": 80.0,
        "gpu_utilization_p50": 81.0,
        "gpu_utilization_p95": 90.0,
        "gpu_power_mean_w": 320.0,
        "gpu_power_peak_w": 350.0,
        "nvidia_smi_peak_memory_used_mib": 40000.0,
        "cpu_rss_peak_mib": 2048.0,
        "torch_peak_allocated_mib": 15000.0,
        "torch_peak_reserved_mib": 16000.0,
        "records_per_second": records_per_second,
        "total_elapsed_seconds": 64.0,
        "nan_or_inf": False,
    }


def _patch_gpu(monkeypatch):
    monkeypatch.setattr(preflight, "query_gpu", lambda index: _gpu())
    monkeypatch.setattr(preflight, "query_compute_processes", lambda uuid: [])
    monkeypatch.setattr(preflight, "_cleanup_cuda", lambda: None)


def test_base_config_is_formal_d1b_and_never_names_test_assets():
    config = yaml.safe_load(CONFIG.read_text())
    assert config["experiment_name"] == "ecir_mvr_formal_large_d1b_seed42"
    assert config["seed"] == 42
    assert config["model"]["bond_explicit_alpha"] == 1.0
    assert config["training"]["total_sample_exposures"] == 1_600_000
    paths = "\n".join(str(value) for value in config["data"].values())
    assert "/test" not in paths and "\\test" not in paths
    assert "test.parquet" not in inspect.getsource(preflight)


def test_shared_gpu_is_blocked_by_default_and_explicitly_allowed():
    processes = [{"pid": 123, "used_gpu_memory_mib": 1000.0}]
    assert preflight.should_block_shared_gpu(processes, allow_shared_gpu=False)
    assert not preflight.should_block_shared_gpu(
        processes, allow_shared_gpu=True
    )


def test_candidate_order_and_accumulation_for_64_and_128():
    pairs64 = preflight.candidate_pairs(64)
    assert [row["micro_batch_size"] for row in pairs64] == [64, 32, 16, 8, 4]
    assert [row["gradient_accumulation_steps"] for row in pairs64] == [1, 2, 4, 8, 16]
    assert all(row["effective_batch_size"] == 64 for row in pairs64)
    pairs128 = preflight.candidate_pairs(128)
    assert [row["micro_batch_size"] for row in pairs128] == [128, 64, 32, 16, 8]
    assert [row["gradient_accumulation_steps"] for row in pairs128] == [1, 2, 4, 8, 16]


def test_capacity_only_allows_256_and_512_without_relaxing_formal_mode():
    config = yaml.safe_load(CONFIG.read_text())
    with pytest.raises(ValueError, match="scientific configuration changed"):
        preflight._validate_base_config(config, 256)
    preflight._validate_base_config(config, 256, capacity_only=True)
    preflight._validate_base_config(config, 512, capacity_only=True)
    with pytest.raises(ValueError, match="scientific configuration changed"):
        preflight._validate_base_config(config, 128, capacity_only=True)
    assert [
        row["micro_batch_size"] for row in preflight.candidate_pairs(256)
    ] == [256, 128, 64, 32, 16]
    assert [
        row["micro_batch_size"] for row in preflight.candidate_pairs(512)
    ] == [512, 256, 128, 64, 32]


def test_capacity_budgets_are_expanded_and_not_scientifically_equivalent():
    budget256 = preflight.budget_definition(256, capacity_only=True)
    budget512 = preflight.budget_definition(512, capacity_only=True)
    assert budget256["optimizer_steps"] == budget512["optimizer_steps"] == 12_500
    assert budget256["total_sample_exposures"] == 3_200_000
    assert budget512["total_sample_exposures"] == 6_400_000
    assert budget256["formal_scientific_equivalence"] is False
    assert preflight.result_status(
        capacity_only=True, has_candidate=True
    ) == preflight.STATUS_CAPACITY_PASS
    assert preflight.result_status(
        capacity_only=True, has_candidate=False
    ) == preflight.STATUS_CAPACITY_FAILED


def test_cuda_visible_devices_maps_physical_and_logical_indices():
    assert preflight.resolve_gpu_selection(1, None) == {
        "physical_gpu_index": 1,
        "logical_cuda_index": 0,
    }
    assert preflight.resolve_gpu_selection(1, "1") == {
        "physical_gpu_index": 1,
        "logical_cuda_index": 0,
    }
    assert preflight.resolve_gpu_selection(0, "1") == {
        "physical_gpu_index": 1,
        "logical_cuda_index": 0,
    }
    assert preflight.resolve_gpu_selection(0, "1,0") == {
        "physical_gpu_index": 0,
        "logical_cuda_index": 1,
    }


def test_candidate_runs_two_step_smoke_then_reinitializes_for_100(monkeypatch):
    _patch_gpu(monkeypatch)
    calls = []

    def phase(config, pair, **kwargs):
        calls.append((kwargs["optimizer_steps"], kwargs["warmup_steps"]))
        return _phase_result()

    result = preflight.benchmark_candidate(
        {"training": {"optimizer_steps": 25000}},
        preflight.candidate_pairs(64)[0],
        device=torch.device("cpu"),
        gpu_index=0,
        preflight_steps=100,
        warmup_steps=20,
        phase_runner=phase,
    )
    assert calls == [(2, 0), (100, 20)]
    assert result["status"] == "PASS"
    assert result["optimizer_steps"] == 100
    assert result["total_sample_exposures"] == 6400


def test_oom_is_cleaned_and_next_smaller_candidate_can_continue(monkeypatch):
    _patch_gpu(monkeypatch)
    cleanup_calls = []
    monkeypatch.setattr(preflight, "_cleanup_cuda", lambda: cleanup_calls.append(1))

    def oom_phase(config, pair, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    def pass_phase(config, pair, **kwargs):
        return _phase_result()

    large, small = preflight.candidate_pairs(64)[:2]
    first = preflight.benchmark_candidate(
        {"training": {"optimizer_steps": 25000}},
        large,
        device=torch.device("cpu"),
        gpu_index=0,
        preflight_steps=100,
        warmup_steps=20,
        phase_runner=oom_phase,
    )
    second = preflight.benchmark_candidate(
        {"training": {"optimizer_steps": 25000}},
        small,
        device=torch.device("cpu"),
        gpu_index=0,
        preflight_steps=100,
        warmup_steps=20,
        phase_runner=pass_phase,
    )
    assert first["status"] == "OOM"
    assert second["status"] == "PASS"
    assert len(cleanup_calls) >= 2


def test_nan_is_reported_and_never_recommended(monkeypatch):
    _patch_gpu(monkeypatch)

    def nan_phase(config, pair, **kwargs):
        raise FloatingPointError("non-finite formal training loss")

    result = preflight.benchmark_candidate(
        {"training": {"optimizer_steps": 25000}},
        preflight.candidate_pairs(64)[0],
        device=torch.device("cpu"),
        gpu_index=0,
        preflight_steps=100,
        warmup_steps=20,
        phase_runner=nan_phase,
    )
    assert result["status"] == "NaN"
    assert result["nan_or_inf"] is True
    assert preflight.recommend_candidate([result]) is None


def test_first_20_warmup_steps_are_excluded_from_statistics():
    rows = []
    for index in range(100):
        measured = 1000.0 if index < 20 else 1.0
        rows.append(
            {
                "optimizer_step_time_seconds": measured,
                "dataloader_time_seconds": measured,
                "forward_time_seconds": measured,
                "backward_time_seconds": measured,
                "optimizer_time_seconds": measured,
                "loss": float(index),
                "gpu_utilization_percent": 50.0,
                "power_draw_w": 300.0,
                "memory_used_mib": 20000.0,
                "cpu_rss_mib": 1000.0,
            }
        )
    summary = preflight.summarize_measured_steps(rows, warmup_steps=20)
    assert summary["measured_optimizer_steps"] == 80
    assert summary["avg_optimizer_step_time_seconds"] == 1.0
    assert summary["loss_start"] == 20.0


def test_recommendation_uses_fastest_safe_candidate_and_strict_margin():
    candidates = [
        {"status": "PASS", "memory_safe": True, "nan_or_inf": False,
         "records_per_second": 90.0, "micro_batch_size": 64},
        {"status": "PASS", "memory_safe": False, "nan_or_inf": False,
         "records_per_second": 120.0, "micro_batch_size": 32},
        {"status": "PASS", "memory_safe": True, "nan_or_inf": False,
         "records_per_second": 100.0, "micro_batch_size": 16},
    ]
    assert preflight.recommend_candidate(candidates)["micro_batch_size"] == 16
    assert preflight.required_safety_margin_mib(20_000.0) == 4096.0
    assert preflight.required_safety_margin_mib(50_000.0) == 5000.0


def test_preflight_uses_real_training_primitives_without_checkpoint_or_launch():
    phase = inspect.getsource(preflight.run_training_phase)
    assert "_dataset" in phase
    assert "DataLoader" in phase
    assert "_build_training_components" in phase
    assert "_forward_loss" in phase
    assert "_backward_loss" in phase
    assert "optimizer.step()" in phase
    source = Path(preflight.__file__).read_text()
    assert "atomic_torch_save" not in source
    assert "subprocess.Popen" not in source
    assert "subprocess.run" not in source
    assert "formal_training_started\": False" in source
    shared_builder = inspect.getsource(training._build_training_components)
    assert "MCVRModel" in shared_builder
    assert "MCVRLoss" in shared_builder
    assert "torch.optim.AdamW" in shared_builder
    runner = (ROOT / "scripts/run_ecir_mvr_formal_preflight.sh").read_text()
    assert "CAPACITY_ONLY" in runner and "--capacity-only" in runner
    assert "train_ecir_mvr_medium_rescue_v2.py" not in runner


def test_preflight_pure_rules_do_not_modify_formal_target_directory(tmp_path):
    target = tmp_path / "minimal_targets" / "immutable.pt"
    target.parent.mkdir()
    target.write_bytes(b"immutable")
    before = (target.read_bytes(), target.stat().st_mtime_ns)
    preflight.candidate_pairs(64)
    preflight.recommend_candidate([])
    after = (target.read_bytes(), target.stat().st_mtime_ns)
    assert after == before


def test_recommended_config_pins_identity_budget_and_preflight_report(tmp_path):
    config = yaml.safe_load(CONFIG.read_text())
    report_path = tmp_path / "preflight.json"
    report = {
        "status": preflight.STATUS_PASS,
        "formal_training_started": False,
        "recommended": {
            "micro_batch_size": 32,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 128,
        },
    }
    report_path.write_text(json.dumps(report))
    recommended_path = tmp_path / "recommended.yaml"
    identities = {
        "formal_target_identity_sha256": "a" * 64,
        "formal_source_identity_sha256": "b" * 64,
    }
    preflight._write_recommended_config(
        config,
        report["recommended"],
        identities,
        recommended_path,
        report_path,
    )
    resolved = yaml.safe_load(recommended_path.read_text())
    assert resolved["training"]["batch_size"] == 32
    assert resolved["training"]["gradient_accumulation_steps"] == 4
    assert resolved["training"]["effective_batch_size"] == 128
    assert resolved["training"]["optimizer_steps"] == 12_500
    assert resolved["training"]["total_sample_exposures"] == 1_600_000
    assert resolved["frozen_identities"] == identities
    assert training._assert_formal_preflight(resolved) == report

    report_path.write_text(json.dumps({**report, "status": "tampered"}))
    with pytest.raises(RuntimeError, match="report identity changed"):
        training._assert_formal_preflight(resolved)


def test_capacity_artifacts_never_overwrite_formal_recommended_config(tmp_path):
    recommended = tmp_path / "D1B_FORMAL_RECOMMENDED_CONFIG.yaml"
    recommended.write_text("formal-128-recommendation\n")
    before = (recommended.read_bytes(), recommended.stat().st_mtime_ns)
    candidate = {
        "micro_batch_size": 128,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 256,
    }
    report = {
        "status": preflight.STATUS_CAPACITY_PASS,
        "mode": "capacity_only",
        "capacity_only": True,
        "shared_gpu": True,
        "gpu": {"gpu_index": 1, "memory_used_mib": 22000, "memory_free_mib": 26000},
        "target_effective_batch": 256,
        "formal_budget": preflight.budget_definition(256, capacity_only=True),
        "candidates": [],
        "recommended": candidate,
        "capacity_best_candidate": candidate,
        "recommendation_requires_stable_external_memory": True,
        "formal_training_command": "must-not-run",
    }
    preflight.write_report_artifacts(
        report,
        config=yaml.safe_load(CONFIG.read_text()),
        identities={},
        report_json=tmp_path / "capacity_effective256/report.json",
        report_md=tmp_path / "capacity_effective256/report.md",
        recommended_config=recommended,
        capacity_only=True,
    )
    after = (recommended.read_bytes(), recommended.stat().st_mtime_ns)
    assert after == before


def test_capacity_output_directories_are_isolated_from_formal_reports():
    paths256 = preflight.output_paths(
        capacity_only=True,
        target_effective_batch=256,
        report_json=None,
        report_md=None,
        recommended_config=None,
    )
    paths512 = preflight.output_paths(
        capacity_only=True,
        target_effective_batch=512,
        report_json=None,
        report_md=None,
        recommended_config=None,
    )
    assert paths256["report_json"].parent.name == "capacity_effective256"
    assert paths512["report_json"].parent.name == "capacity_effective512"
    assert paths256["report_json"] != paths512["report_json"]
    with pytest.raises(ValueError, match="output paths are fixed"):
        preflight.output_paths(
            capacity_only=True,
            target_effective_batch=256,
            report_json=preflight.DEFAULT_REPORT_JSON,
            report_md=None,
            recommended_config=None,
        )
