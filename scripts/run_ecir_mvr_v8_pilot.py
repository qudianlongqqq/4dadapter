#!/usr/bin/env python
"""Fail-closed Full -> matched D1 -> no-Angle V8 pilot orchestrator."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

from scripts.train_ecir_mvr_v8 import ISOLATION, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ecir_mvr_v8_full_v1.yaml"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("diagnostics/ecir_mvr/v8_full_v1/pilots")
    )
    parser.add_argument(
        "--state", type=Path, default=Path("reports/ecir_mvr/MCVR_V8_FULL_V1_STATUS.json")
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    required = [
        Path(config["data"][name])
        for name in ("train_sources", "train_targets", "val_sources", "val_targets")
    ]
    required.extend(
        [
            Path(config["data"]["source_cache_root"]),
            Path(config["data"]["target_cache_root"]),
            Path(config["constraint_layer"]["frozen_scales"]),
            Path(config["sampler"]["manifest"]),
        ]
    )
    missing = [str(path.resolve()) for path in required if not path.exists()]
    state = {
        "schema_version": "mcvr-v8-full-v1-status-v1",
        "status": "MCVR_V8_FULL_V1_BLOCKED"
        if missing
        else "MCVR_V8_FULL_V1_IMPLEMENTED_PILOT_RUNNING",
        "pilot_order": ["full", "matched_d1_only", "no_angle"],
        "completed": [],
        "running": None,
        "missing_required_assets": missing,
        **ISOLATION,
    }
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    if missing:
        print(json.dumps(state, indent=2, sort_keys=True))
        raise SystemExit(2)
    if not args.execute:
        print(json.dumps(state, indent=2, sort_keys=True))
        return
    variants = (
        ("full", args.config),
        ("matched_d1_only", Path("configs/ecir_mvr_v8_d1_only_matched.yaml")),
        ("no_angle", Path("configs/ecir_mvr_v8_no_angle.yaml")),
    )
    for name, variant in variants:
        state["running"] = name
        args.state.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        command = [
            sys.executable,
            str(ROOT / "scripts/train_ecir_mvr_v8.py"),
            "--config",
            str(variant),
            "--output-dir",
            str(args.output_root / name),
            "--device",
            args.device,
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        state["completed"].append(name)
    state["running"] = None
    state["status"] = "MCVR_V8_FULL_V1_IMPLEMENTED_AND_PILOTED"
    args.state.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
