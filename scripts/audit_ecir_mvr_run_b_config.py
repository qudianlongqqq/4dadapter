#!/usr/bin/env python
"""Fail-closed single-variable and frozen-identity audit for Stage 2b Run B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_a", type=Path, required=True)
    parser.add_argument("--run_b", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    args = parser.parse_args()
    a = yaml.safe_load(args.run_a.read_text(encoding="utf-8"))
    b = yaml.safe_load(args.run_b.read_text(encoding="utf-8"))
    checks = {
        "seed_identical": a["seed"] == b["seed"] == 42,
        "data_identical": a["data"] == b["data"],
        "frozen_identities_identical": a["frozen_identities"] == b["frozen_identities"],
        "training_identical": a["training"] == b["training"],
        "noninferiority_identical": a["noninferiority"] == b["noninferiority"],
        "inference_common_identical": all(a["inference"][key] == b["inference"][key]
                                          for key in a["inference"]),
        "loss_common_identical": all(a["loss"][key] == b["loss"][key] for key in a["loss"]),
        "model_nontorsion_identical": all(
            a["model"][key] == b["model"][key]
            for key in a["model"]
            if key not in {"torsion_scale", "high_flex_torsion_scale", "torsion_gate_fixed_zero"}
        ),
        "run_a_checkpoint_frozen": sha(b["run_a_checkpoint"]) == b["run_a_checkpoint_sha256"],
        "run_b_steps_5000": b["training"]["optimizer_steps"] == 5000,
        "conservative_scales_frozen": (
            b["model"]["rigid_scale"] == 1.0
            and b["model"]["torsion_scale"] == 0.10
            and b["model"]["high_flex_torsion_scale"] == 0.05
        ),
        "torsion_trust_frozen": (
            b["run_b_mode"]["max_torsion_change_rad"] == 0.035
            and b["run_b_mode"]["max_high_flex_torsion_change_rad"] == 0.020
        ),
        "long_runs_not_configured": b["training"]["optimizer_steps"] < 20000,
    }
    result = {
        "schema_version": "ecir-mvr-run-b-config-audit-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "run_a_config_sha256": sha(args.run_a),
        "run_b_config_sha256": sha(args.run_b), "test_records_read": 0,
        "run_a_checkpoint_sha256": sha(b["run_a_checkpoint"]),
        "frozen_identities": b["frozen_identities"],
    }
    if result["status"] != "PASS":
        raise RuntimeError(json.dumps(result, indent=2))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Run B single-variable audit", "", "Status: **PASS**", ""]
    lines += [f"- {name}: {value}" for name, value in checks.items()]
    lines += ["", f"- Run B config SHA256: `{result['run_b_config_sha256']}`",
              "- Test records read: 0", ""]
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
