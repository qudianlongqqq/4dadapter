from __future__ import annotations

import json
from pathlib import Path

import yaml

from etflow.commons.run_timing import RunTiming, write_heartbeat


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/ecir_mvr_medium_5k_500_run_a_seed42_20k_rescue_v2.yaml"


def _config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_rescue_v2_preserves_scientific_training_budget():
    config = _config()
    training = config["training"]
    assert training["batch_size"] == 8
    assert training["gradient_accumulation_steps"] == 1
    assert training["effective_batch_size"] == 8
    assert training["learning_rate"] == 0.0002
    assert training["optimizer_steps"] == 20000


def test_rescue_v2_model_loss_data_and_inference_equal_v1():
    rescue = _config()
    v1 = yaml.safe_load((ROOT / "configs/ecir_mvr_medium_5k_500_run_a_seed42_20k.yaml").read_text(encoding="utf-8"))
    for key in ("data", "frozen_identities", "model", "run_a_mode", "loss", "inference", "noninferiority"):
        if key == "data":
            rescue_data = {name: value for name, value in rescue[key].items() if name != "target_audit"}
            assert rescue_data == v1[key]
        else:
            assert rescue[key] == v1[key]


def test_rescue_v2_starts_from_step_zero_and_has_one_controller_resume():
    config = _config()
    assert config["initialize_from_checkpoint"] is None
    assert config["resume_checkpoint"] is None
    assert config["recovery"]["max_automatic_resumes"] == 1
    assert config["recovery"]["resume_on_oom"] is False
    assert config["recovery"]["resume_on_model_safety_stop"] is False


def test_rescue_v2_velocity_growth_is_info_not_stop():
    config = _config()
    source = (ROOT / "scripts/train_ecir_mvr_medium_rescue_v2.py").read_text(encoding="utf-8")
    assert config["safety"]["sustained_velocity_growth_is_info_only"] is True
    assert 'stop_reason = "velocity_norm_sustained_growth"' not in source
    assert "INFO velocity_norm_sustained_growth_below_hard_limits" in source
    assert config["safety"]["max_velocity_graph_rms"] == 0.06
    assert config["safety"]["max_velocity_atom_norm"] == 0.12


def test_rescue_v2_checkpoint_and_validation_schedule():
    training = _config()["training"]
    required = [1000, 2000, 3000, 5000, 10000, 15000, 20000]
    assert training["checkpoint_interval"] == 1000
    assert training["checkpoint_steps"] == required
    assert training["checkpoint_validation_steps"] == required


def test_timing_and_heartbeat_are_atomic_artifacts(tmp_path):
    timing = RunTiming(tmp_path)
    timing.mark("pipeline_start")
    timing.mark("identity_audit_start")
    timing.mark("identity_audit_end")
    timing.mark("pipeline_end")
    summary = timing.finalize(
        completed_optimizer_steps=1000, batch_size=8,
        active_optimizer_seconds=100.0,
        interval_rows=[{
            "step_start": 0, "step_end": 1000, "interval_seconds": 110.0,
            "cumulative_training_seconds": 110.0, "active_optimizer_seconds": 100.0,
            "validation_seconds": 0.0, "steps_per_second": 10.0,
            "examples_per_second": 80.0, "ETA_seconds": 1900.0,
            "ETA_finish_time": "2026-01-01T00:00:00+08:00",
            "cuda_peak_allocated_mib": 1.0, "cuda_peak_reserved_mib": 2.0,
            "gpu_utilization_mean": 50.0, "gpu_utilization_p95": 75.0,
        }],
    )
    assert summary["mean_optimizer_steps_per_second"] == 10.0
    assert summary["mean_examples_per_second"] == 80.0
    assert (tmp_path / "run_timeline.log").is_file()
    assert (tmp_path / "timing.csv").is_file()
    write_heartbeat(tmp_path, status="RUNNING", current_step=1, target_step=20000)
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert heartbeat["status"] == "RUNNING" and heartbeat["current_step"] == 1


def test_rescue_v2_state_is_authorized_without_100k():
    state = json.loads((ROOT / "reports/ecir_mvr/progressive_state.json").read_text(encoding="utf-8"))
    assert state["medium_rescue_v2_permitted"] is True
    assert state["100k_permitted"] is False
    assert state["100k_started"] is False
