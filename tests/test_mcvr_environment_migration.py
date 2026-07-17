import json
from pathlib import Path
import pandas as pd
from scripts.compare_mcvr_environments import compare
from scripts.compare_stage_h0_smoke import compare as compare_smoke
from scripts.verify_mcvr_stage_h0_assets import verify

POLICY={"hard_match":{"python":"major_minor","torch":"major_minor","numpy":"major_minor"},"warn_only":["platform"]}
def env(**kw):
    value={"python":"3.11.9","torch":"2.5.1","packages":{"numpy":"1.26.4"},"platform":"Windows"};value.update(kw);return value
def test_environment_report_fields():
    source=Path("scripts/capture_mcvr_environment.py").read_text();assert "pip_freeze_sha256" in source and "compute_capability" in source and "tf32_matmul" in source
def test_hard_mismatch(): assert compare(env(),env(python="3.12.1"),POLICY)["status"]=="INCOMPATIBLE"
def test_warning_mismatch(): assert compare(env(),env(platform="Linux"),POLICY)["status"]=="COMPATIBLE_WITH_WARNINGS"
def test_smoke_identity_mismatch():
    base={"record_ids":["a"],"molecule_ids":["m"],"variants":["x"],"test_records":0,"methods":{},"clean_identity":{},"checkpoint_sha256":"x","validation_source_sha256":"y","validation_target_sha256":"z","validation_records":1}
    assert compare_smoke(base,{**base,"record_ids":["b"]})["status"]=="MISMATCH"
def test_smoke_float_tolerance():
    base={"record_ids":[],"molecule_ids":[],"variants":[],"test_records":0,"methods":{"x":{"v":1.}},"clean_identity":{},"checkpoint_sha256":"x","validation_source_sha256":"y","validation_target_sha256":"z","validation_records":1}
    assert compare_smoke(base,{**base,"methods":{"x":{"v":1+1e-7}}})["status"]=="MATCH_WITH_NUMERICAL_TOLERANCE"
def test_asset_missing(tmp_path):
    cfg=tmp_path/"c.yaml";cfg.write_text("data:\n  validity_statistics: missing\ncheckpoint:\n  path: missing\n  sha256: x\n");r=verify(cfg,tmp_path/"out.json");assert r["missing"]
def test_parquet_schema(tmp_path):
    path=tmp_path/"x.parquet";pd.DataFrame({"a":[1]}).to_parquet(path);assert str(pd.read_parquet(path).dtypes["a"])=="int64"
def test_checkpoint_sha_declared():
    import yaml;cfg=yaml.safe_load(Path("configs/ecir_mvr_stage_h0_conflict_fusion.yaml").read_text());assert len(cfg["checkpoint"]["sha256"])==64
def test_test_isolation():
    import yaml;cfg=yaml.safe_load(Path("configs/ecir_mvr_stage_h0_conflict_fusion.yaml").read_text());assert cfg["test_records_read"]==0
def test_dirty_worktree_check_present(): assert "status\",\"--porcelain" in Path("scripts/preflight_mcvr_stage_h0.py").read_text()
def test_protected_file_check(): assert "protected_file" in Path("configs/ecir_mvr_stage_h0_conflict_fusion.yaml").read_text()
