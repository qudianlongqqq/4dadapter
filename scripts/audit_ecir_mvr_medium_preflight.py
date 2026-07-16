#!/usr/bin/env python
"""Fail-closed preflight for the frozen MCVR medium seed42 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import pandas as pd
import rdkit
import torch
import yaml


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(name: str, value: bool, details: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(value), "details": details}


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("diagnostics/ecir_mvr/medium/run_a_seed42_20k/preflight.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/MCVR_MEDIUM_PREFLIGHT.md"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    split_meta = json.loads(Path(data["split_metadata"]).read_text(encoding="utf-8"))
    source_meta = json.loads(Path(data["source_metadata"]).read_text(encoding="utf-8"))
    target_meta = json.loads(Path(data["target_metadata"]).read_text(encoding="utf-8"))
    target_audit_path = Path(data.get("target_audit", Path(config["diagnostics_dir"]) / "target_audit.json"))
    target_audit = json.loads(target_audit_path.read_text(encoding="utf-8"))
    train_s, val_s = pd.read_parquet(data["train_sources"]), pd.read_parquet(data["val_sources"])
    train_t, val_t = pd.read_parquet(data["train_targets"]), pd.read_parquet(data["val_targets"])
    run_a = yaml.safe_load(Path("configs/ecir_mvr_stage2b_run_a_rigid_only_500_100_5k.yaml").read_text(encoding="utf-8"))
    train_molecules, val_molecules = set(train_s.molecule_id), set(val_s.molecule_id)
    identities = {
        "stage_c_decision": "PASS",
        "validity_statistics_identity_sha256": source_meta["validity_statistics_identity_sha256"],
        "parent_formal_split_identity_sha256": split_meta["parent_formal_split"]["identity_sha256"],
        "medium_split_identity_sha256": split_meta["identity_sha256"],
        "medium_real_source_identity_sha256": source_meta["medium_real_source_identity_sha256"],
        "medium_target_identity_sha256": target_meta["medium_target_identity_sha256"],
    }
    rescue_v2 = config["experiment_name"] == "ecir_mvr_medium_5k_500_run_a_seed42_20k_rescue_v2"
    training = config["training"]
    trainer_source = Path("scripts/train_ecir_mvr_medium_rescue_v2.py").read_text(encoding="utf-8") if rescue_v2 else ""
    state = json.loads(Path("reports/ecir_mvr/progressive_state.json").read_text(encoding="utf-8"))
    checks = [
        _check("01_molecule_counts", len(train_molecules) == 5000 and len(val_molecules) == 500, {"train": len(train_molecules), "val": len(val_molecules)}),
        _check("02_train_val_no_overlap", not (train_molecules & val_molecules), len(train_molecules & val_molecules)),
        _check("03_test_not_read", split_meta["test_paths_opened"] == source_meta["test_paths_opened"] == target_meta["test_paths_opened"] == 0, 0),
        _check("04_medium_real_source_identity", identities["medium_real_source_identity_sha256"] == config["frozen_identities"]["medium_real_source_identity_sha256"], identities["medium_real_source_identity_sha256"]),
        _check("05_medium_target_identity", target_meta["decision"] == "PASS" and identities["medium_target_identity_sha256"] == config["frozen_identities"]["medium_target_identity_sha256"], identities["medium_target_identity_sha256"]),
        _check("06_validity_statistics_identity", identities["validity_statistics_identity_sha256"] == "66dd6ab6cf290057d8dea725e0042ba5a5fcc69ad0ec9e0db8971910df377cd3", identities["validity_statistics_identity_sha256"]),
        _check("07_source_ratios", set(train_s.generator_name) == set(val_s.generator_name) == {"ETFlow_formal_upstream", "Cartesian_teacher_100k"}, {"train": train_s.generator_name.value_counts().to_dict(), "val": val_s.generator_name.value_counts().to_dict()}),
        _check("08_severity_coverage", {"normal", "mild", "medium", "severe"}.issubset(set(val_s.source_severity)), val_s.source_severity.value_counts().to_dict()),
        _check("09_high_flex_count", int((val_s.rotatable_group == "rotatable_ge_6").sum()) >= 20, int((val_s.rotatable_group == "rotatable_ge_6").sum())),
        _check("10_ring_non_ring_count", {"ring", "non_ring"}.issubset(set(val_s.ring_group)), val_s.ring_group.value_counts().to_dict()),
        _check("11_unseen_scale", set(train_s.update_scale) == {0.0, 0.5} and set(val_s.update_scale) == {0.0, 0.35}, {"train": sorted(train_s.update_scale.unique()), "val": sorted(val_s.update_scale.unique())}),
        _check("12_no_extreme_source", int((train_s.source_severity == "out_of_domain_extreme").sum() + (val_s.source_severity == "out_of_domain_extreme").sum()) == 0, 0),
        _check("13_clean_ratio", config["data"]["mixture"] == run_a["data"]["mixture"] and config["data"]["mixture"]["clean_identity"] == 0.25, config["data"]["mixture"]),
        _check("14_target_fallback_ratio", target_audit["fallback_records"] / target_audit["records"] <= 0.10, target_audit["fallback_records"] / target_audit["records"]),
        _check("15_frozen_run_a_parameters", all(config[key] == run_a[key] for key in ("model", "run_a_mode", "loss", "inference", "noninferiority")), "model/run_a_mode/loss/inference/noninferiority exact"),
        _check("16_torsion_fixed_zero", config["model"]["torsion_gate_fixed_zero"] and config["model"]["torsion_scale"] == config["model"]["high_flex_torsion_scale"] == config["run_a_mode"]["torsion_velocity_scale"] == 0.0 and not config["run_a_mode"]["enable_torsion_repair"], config["run_a_mode"]),
        _check("17_config_identity", config["frozen_identities"] == identities, _sha(args.config)),
        _check("18_git_commit", subprocess.run(["git", "merge-base", "--is-ancestor", "e29286944cad8c3f2cc2a60fb69773edc047dbaf", "HEAD"], cwd=ROOT).returncode == 0, subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()),
        _check("19_environment", torch.cuda.is_available() and torch.version.cuda == "12.8", {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "cuda": torch.version.cuda, "torch": torch.__version__, "rdkit": rdkit.__version__, "python": platform.python_version()}),
        _check("20_seed_and_scratch", config["seed"] == 42 and config["initialize_from_checkpoint"] is None and config["resume_checkpoint"] is None, {"seed": config["seed"], "initialize": config["initialize_from_checkpoint"], "resume": config["resume_checkpoint"]}),
    ]
    if rescue_v2:
        checks.extend([
            _check("21_rescue_v2_scientific_batch", training["batch_size"] == training["effective_batch_size"] == 8 and training["gradient_accumulation_steps"] == 1, {key: training[key] for key in ("batch_size", "effective_batch_size", "gradient_accumulation_steps")}),
            _check("22_rescue_v2_budget", training["optimizer_steps"] == 20000 and float(training["learning_rate"]) == 0.0002, {"optimizer_steps": training["optimizer_steps"], "learning_rate": training["learning_rate"]}),
            _check("23_velocity_growth_info_only", config["safety"]["sustained_velocity_growth_is_info_only"] is True and 'stop_reason = "velocity_norm_sustained_growth"' not in trainer_source, "sustained velocity growth cannot assign a stop reason"),
            _check("24_hard_velocity_limits", config["safety"]["max_velocity_graph_rms"] == config["model"]["max_velocity_graph_rms"] == 0.06 and config["safety"]["max_velocity_atom_norm"] == config["model"]["max_velocity_atom_norm"] == 0.12, config["safety"]),
            _check("25_checkpoint_and_validation_schedule", training["checkpoint_interval"] == 1000 and training["checkpoint_steps"] == [1000, 2000, 3000, 5000, 10000, 15000, 20000] and training["checkpoint_validation_steps"] == [1000, 2000, 3000, 5000, 10000, 15000, 20000], {"checkpoint_steps": training["checkpoint_steps"], "validation_steps": training["checkpoint_validation_steps"]}),
            _check("26_rescue_permission_boundary", bool(state.get("medium_rescue_v2_permitted")) and not state["100k_permitted"] and not state["100k_started"], {"medium_rescue_v2_permitted": state.get("medium_rescue_v2_permitted"), "100k_permitted": state["100k_permitted"], "100k_started": state["100k_started"]}),
        ])
    status = "PASS" if all(item["pass"] for item in checks) else "PREFLIGHT_FAIL"
    result = {
        "schema_version": "ecir-mvr-medium-preflight-v2" if rescue_v2 else "ecir-mvr-medium-preflight-v1", "status": status,
        "config": str(args.config), "config_sha256": _sha(args.config),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "seed": 42, "test_records_read": 0, "identities": identities,
        "train": {"records": len(train_s), "molecules": len(train_molecules), "targets": len(train_t)},
        "val": {"records": len(val_s), "molecules": len(val_molecules), "targets": len(val_t)},
        "checks": checks,
    }
    result["identity_sha256"] = _canonical_sha(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    lines = ["# MCVR Medium Preflight", "", f"Decision: **{status}**", "", f"Config SHA256: `{result['config_sha256']}`", "", "| Check | Result |", "|---|---|"]
    lines.extend(f"| {item['name']} | {'PASS' if item['pass'] else 'FAIL'} |" for item in checks)
    lines += ["", "The audit is validation-only, opened zero test paths, and authorizes no 100k run."]
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    if status != "PASS":
        raise SystemExit("PREFLIGHT_FAIL")


if __name__ == "__main__":
    main()
