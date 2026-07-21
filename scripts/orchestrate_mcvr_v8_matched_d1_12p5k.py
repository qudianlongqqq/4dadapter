#!/usr/bin/env python
"""Launch and verify the formal matched D1-only 12.5K exposure-control run."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import torch

from etflow.ecir.v8_validation_cache import ISOLATION, atomic_json


FINAL_STATUS = "MCVR_V8_MATCHED_D1_FORMAL_LARGE_12P5K_COMPLETED"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, attempts: int = 40) -> dict[str, Any]:
    last_error: OSError | None = None
    for _ in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            last_error = error
            time.sleep(0.05)
    assert last_error is not None
    raise last_error


def write_status(path: Path, status: str, **values: Any) -> None:
    atomic_json(
        path,
        {
            "schema_version": "mcvr-v8-matched-d1-12p5k-orchestration-status-v1",
            "status": status,
            "last_update_time": datetime.now(timezone.utc).isoformat(),
            **values,
            **ISOLATION,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner-python", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("formal matched D1 output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    orchestration_status = output / "control/orchestration_status.json"
    orchestration_result = output / "matched_d1_12p5k_orchestration.json"
    command = [
        str(args.runner_python.resolve()),
        str(ROOT / "scripts/train_ecir_mvr_v8.py"),
        "--config",
        str(ROOT / "configs/ecir_mvr_v8_d1_only_matched_formal_large_200k.yaml"),
        "--output-dir",
        str(output),
        "--validation-batches",
        "625",
        "--device",
        args.device,
    ]
    creation_flags = 0x08000000 if os.name == "nt" else 0
    try:
        with (output / "stdout.log").open("a", encoding="utf-8") as stdout, (
            output / "stderr.log"
        ).open("a", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                creationflags=creation_flags,
            )
            (output / "pid.txt").write_text(f"{process.pid}\n", encoding="ascii")
            deadline = time.time() + 300
            while not (output / "status.json").is_file():
                if process.poll() is not None:
                    raise RuntimeError(
                        f"matched D1 runner exited during initialization: {process.returncode}"
                    )
                if time.time() > deadline:
                    raise RuntimeError("matched D1 runner initialization timed out")
                time.sleep(0.25)
            subprocess.run(
                [
                    str(args.runner_python.resolve()),
                    str(ROOT / "scripts/request_mcvr_v8_graceful_stop.py"),
                    "--output-dir",
                    str(output),
                    "--stop-step",
                    "12500",
                    "--effective-batch",
                    "64",
                    "--final-status",
                    FINAL_STATUS,
                    "--schedule-provenance",
                    "checkpoint_from_original_200k_schedule",
                ],
                cwd=ROOT,
                check=True,
                timeout=60,
                stdout=stdout,
                stderr=stderr,
            )
            initialized = read_json(output / "status.json")
            assets = read_json(output / "asset_hashes.json")
            write_status(
                orchestration_status,
                "MCVR_V8_MATCHED_D1_FORMAL_LARGE_12P5K_RUNNING",
                pid=process.pid,
                actual_launch_command=command,
                branch=assets["git_branch"],
                head=assets["git_head"],
                d1_checkpoint_sha256=assets["d1_checkpoint_sha256"],
                resolved_config_sha256=assets["resolved_config_sha256"],
                planned_original_total_steps=200000,
                user_requested_stop_step=12500,
                effective_batch=64,
                total_record_exposure=800000,
                initial_training_step=initialized["training_step"],
            )
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"matched D1 runner exited with code {return_code}")
        final = read_json(output / "status.json")
        if final.get("status") != FINAL_STATUS:
            raise RuntimeError("matched D1 final status changed")
        if int(final.get("actual_completed_step", -1)) != 12500:
            raise RuntimeError("matched D1 did not complete exact step12500")
        step_checkpoint = output / "checkpoints/step012500.ckpt"
        last_checkpoint = output / "checkpoints/last.ckpt"
        step_sha, last_sha = sha256(step_checkpoint), sha256(last_checkpoint)
        if step_sha != last_sha:
            raise RuntimeError("matched D1 step12500 and last checkpoint differ")
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        for key in (
            "optimizer_state_dict",
            "scheduler_state_dict",
            "rng_states",
            "sampler_state",
        ):
            if key not in checkpoint:
                raise RuntimeError(f"matched D1 checkpoint missing {key}")
        if checkpoint["sampler_state"]["records_exposed"] != 800000:
            raise RuntimeError("matched D1 sampler exposure changed")
        subprocess.run(
            [
                str(args.runner_python.resolve()),
                str(ROOT / "scripts/report_mcvr_v8_vs_matched_d1_12p5k.py"),
            ],
            cwd=ROOT,
            check=True,
            timeout=1800,
        )
        full_report = read_json(
            output / "validation_cache/step012500/full/evaluation.json"
        )
        result = {
            "schema_version": "mcvr-v8-matched-d1-12p5k-orchestration-result-v1",
            "status": FINAL_STATUS,
            "pid": process.pid,
            "actual_launch_command": command,
            "process_return_code": return_code,
            "process_normal_exit": True,
            "force_kill_used": False,
            "planned_original_total_steps": 200000,
            "user_requested_stop_step": 12500,
            "actual_completed_step": 12500,
            "effective_batch": 64,
            "total_record_exposure": 800000,
            "equivalent_old_batch8_steps": 100000,
            "schedule_provenance": "checkpoint_from_original_200k_schedule",
            "step012500_checkpoint_sha256": step_sha,
            "last_checkpoint_sha256": last_sha,
            "full_validation": {
                "status": full_report["status"],
                "metrics": full_report["metrics"],
                "set_metrics": full_report["set_metrics"],
            },
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **ISOLATION,
        }
        atomic_json(orchestration_result, result)
        write_status(
            orchestration_status,
            "COMPLETED",
            final_run_status=FINAL_STATUS,
            pid=process.pid,
            process_return_code=0,
            process_normal_exit=True,
            checkpoint_sha256=step_sha,
        )
    except BaseException as error:
        write_status(orchestration_status, "FAILED_CLOSED", error=str(error))
        raise


if __name__ == "__main__":
    main()
