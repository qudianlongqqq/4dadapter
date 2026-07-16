from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ecir_mvr_medium_5k_500_run_a_seed42_20k.yaml"


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_medium_config_is_true_5000_500_scale():
    config = _config()
    assert config["data"]["train_molecules"] == 5000
    assert config["data"]["val_molecules"] == 500
    assert config["training"]["optimizer_steps"] == 20000


def test_historical_500_100_candidate_is_not_executable_medium():
    historical = yaml.safe_load((ROOT / "configs/ecir_mvr_medium_20k_run_a_selected.yaml").read_text(encoding="utf-8"))
    assert historical["scale_class"] == "extended_pilot_500_100_20k_candidate"
    assert historical["execution_permitted"] is False


def test_medium_is_from_scratch():
    config = _config()
    assert config["initialize_from_checkpoint"] is None
    assert config["resume_checkpoint"] is None
    assert config["provenance"]["training_from_scratch"] is True


def test_medium_run_a_torsion_is_strictly_disabled():
    config = _config()
    assert config["run_a_mode"]["enable_torsion_repair"] is False
    assert config["run_a_mode"]["torsion_gate_fixed_zero"] is True
    assert config["run_a_mode"]["torsion_velocity_scale"] == 0.0
    assert config["model"]["torsion_scale"] == 0.0
    assert config["model"]["high_flex_torsion_scale"] == 0.0


def test_medium_model_loss_inference_and_margins_equal_run_a():
    medium = _config()
    run_a = yaml.safe_load((ROOT / "configs/ecir_mvr_stage2b_run_a_rigid_only_500_100_5k.yaml").read_text(encoding="utf-8"))
    for key in ("model", "run_a_mode", "loss", "inference", "noninferiority"):
        assert medium[key] == run_a[key]


def test_medium_split_is_test_free_and_disjoint():
    train = json.loads((ROOT / "data/ecir_mvr/medium/split_train.json").read_text(encoding="utf-8"))
    val = json.loads((ROOT / "data/ecir_mvr/medium/split_val.json").read_text(encoding="utf-8"))
    train_ids = {item["molecule_id"] for item in train["records"]}
    val_ids = {item["molecule_id"] for item in val["records"]}
    assert len(train_ids) == 5000 and len(val_ids) == 500
    assert not train_ids & val_ids
    assert all("test" not in {part.lower() for part in Path(item["source_path"]).parts} for item in train["records"] + val["records"])


def test_medium_validation_has_registered_rare_groups():
    val = pd.read_parquet(ROOT / "data/ecir_mvr/medium/real_sources/val.parquet")
    assert {"normal", "mild", "medium", "severe"}.issubset(set(val.source_severity))
    assert {"ring", "non_ring"}.issubset(set(val.ring_group))
    assert int((val.rotatable_group == "rotatable_ge_6").sum()) >= 20


def test_medium_cartesian_protocol_is_safe_and_unseen():
    train = pd.read_parquet(ROOT / "data/ecir_mvr/medium/real_sources/train.parquet")
    val = pd.read_parquet(ROOT / "data/ecir_mvr/medium/real_sources/val.parquet")
    assert float(max(train.t_max.max(), val.t_max.max())) <= 0.25
    assert set(train.update_scale) == {0.0, 0.5}
    assert set(val.update_scale) == {0.0, 0.35}
    assert "out_of_domain_extreme" not in set(train.source_severity) | set(val.source_severity)


def test_medium_target_gate_and_identities_are_frozen():
    config = _config()
    source = json.loads((ROOT / config["data"]["source_metadata"]).read_text(encoding="utf-8"))
    target = json.loads((ROOT / config["data"]["target_metadata"]).read_text(encoding="utf-8"))
    assert target["decision"] == "PASS"
    assert source["medium_real_source_identity_sha256"] == config["frozen_identities"]["medium_real_source_identity_sha256"]
    assert target["medium_target_identity_sha256"] == config["frozen_identities"]["medium_target_identity_sha256"]
    assert source["test_paths_opened"] == target["test_paths_opened"] == 0


def test_medium_identity_fallbacks_are_exact_coordinates():
    for split in ("train", "val"):
        frame = pd.read_parquet(ROOT / f"data/ecir_mvr/medium/minimal_targets/{split}.parquet")
        fallback = frame[frame.target_status == "identity_fallback"]
        for path in fallback.target_cache_path.iloc[:10]:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            assert torch.equal(payload["x_input"], payload["x_target"])


def test_preflight_is_test_free_pass():
    audit = json.loads((ROOT / "diagnostics/ecir_mvr/medium/run_a_seed42_20k/preflight.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["test_records_read"] == 0
    assert all(item["pass"] for item in audit["checks"])


def test_run_a_checkpoint_remains_unchanged():
    path = ROOT / "logs_ecir_mvr/stage2b/run_a_rigid_only_seed42_5k/checkpoints/best_noninferior_validity.ckpt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "ac3e7e3b1fa4189e8ccdfeb45ea7c799a7130c213aeed017c301218b71487070"
