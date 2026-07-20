#!/usr/bin/env python
"""Summarize V8 checkpoint, solver, gradient, confidence, and isolation diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


ISOLATION = {
    "formal_test_records_read": 0,
    "formal_test_assets_opened": False,
    "minimal_validity_target_test_used": False,
    "frozen_holdout_records_read": 0,
    "parameter_selection_from_formal_test": False,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-log", type=Path)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "mcvr-v8-full-v1-checkpoint-v1":
        raise RuntimeError("not a V8 Full v1 checkpoint")
    latest_train = None
    if args.train_log and args.train_log.is_file():
        lines = args.train_log.read_text(encoding="utf-8").splitlines()
        latest_train = json.loads(lines[-1]) if lines else None
    validation = (
        json.loads(args.validation_report.read_text(encoding="utf-8"))
        if args.validation_report and args.validation_report.is_file()
        else None
    )
    result = {
        "schema_version": "mcvr-v8-diagnostics-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "step": int(checkpoint["step"]),
        "unroll_steps": int(checkpoint["unroll_steps"]),
        "resolved_config_sha256": checkpoint["resolved_config_sha256"],
        "residual_scales_identity_sha256": checkpoint["residual_scales_identity_sha256"],
        "latest_train": latest_train,
        "validation": validation,
        **ISOLATION,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
