#!/usr/bin/env python
"""Atomically request a future graceful stop from a compatible MCVR V8 runner."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from etflow.ecir.v8_validation_cache import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-step", type=int, required=True)
    parser.add_argument("--effective-batch", type=int, default=64)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    config_path = output_dir / "config.resolved.json"
    status_path = output_dir / "status.json"
    if not config_path.is_file() or not status_path.is_file():
        raise RuntimeError("graceful stop request requires an initialized V8 run")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    planned_total = int(config["steps_total"])
    current_step = int(status["training_step"])
    if not current_step < args.stop_step <= planned_total:
        raise RuntimeError("graceful stop step must be ahead of the current training step")
    total_exposure = int(args.stop_step) * int(args.effective_batch)
    request = {
        "schema_version": "mcvr-v8-graceful-stop-request-v1",
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "planned_original_total_steps": planned_total,
        "user_requested_stop_step": int(args.stop_step),
        "effective_batch": int(args.effective_batch),
        "total_record_exposure": total_exposure,
        "equivalent_old_batch8_steps": total_exposure // 8,
        "validation_mode": "FULL",
        "final_status": "MCVR_V8_FULL_V1_FORMAL_LARGE_12P5K_COMPLETED",
        "schedule_provenance": "checkpoint_from_original_200k_schedule",
        "formal_test_records_read": 0,
        "frozen_holdout_records_read": 0,
    }
    path = output_dir / "control" / "stop_request.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        stable_existing = {key: value for key, value in existing.items() if key != "requested_at"}
        stable_request = {key: value for key, value in request.items() if key != "requested_at"}
        if stable_existing != stable_request:
            raise RuntimeError("a different graceful stop request already exists")
        return
    atomic_json(path, request)
    print(json.dumps({"path": str(path), **request}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
