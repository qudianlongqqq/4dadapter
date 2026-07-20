#!/usr/bin/env python
"""Safely bridge an already-running legacy V8 process to a 12.5K graceful stop."""

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

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import torch
import psutil

from etflow.ecir.v8_validation_cache import atomic_json


ISOLATION = {
    "formal_test_records_read": 0,
    "formal_test_assets_opened": False,
    "minimal_validity_target_test_used": False,
    "frozen_holdout_records_read": 0,
    "parameter_selection_from_formal_test": False,
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, *, attempts: int = 40, delay_seconds: float = 0.05) -> dict:
    """Read a JSON artifact robustly across transient Windows replace locks."""
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(delay_seconds)
    assert last_error is not None
    raise last_error


def _alive(pid: int) -> bool:
    if not psutil.pid_exists(pid):
        return False
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _status(path: Path, **updates) -> None:
    payload = {
        "schema_version": "mcvr-v8-12p5k-stop-orchestration-status-v1",
        "last_update_time": datetime.now(timezone.utc).isoformat(),
        **updates,
        **ISOLATION,
    }
    atomic_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-pid", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    orchestration_status = output / "control" / "orchestration_status.json"
    orchestration_result = output / "graceful_stop_orchestration.json"
    checkpoint_10k = output / "checkpoints" / "step010000.ckpt"
    fast_report = output / "validation_cache" / "step010000" / "fast" / "evaluation.json"
    stop_request = output / "control" / "stop_request.json"
    if not stop_request.is_file():
        raise RuntimeError("12.5K orchestration requires an atomic stop request")
    request = _json(stop_request)
    if int(request["user_requested_stop_step"]) != 12500:
        raise RuntimeError("12.5K orchestration stop request changed")
    if not _alive(args.original_pid):
        raise RuntimeError("original V8 process is not alive")
    _atomic_text(output / "original_pid.txt", f"{args.original_pid}\n")
    _status(
        orchestration_status,
        status="WAITING_FOR_STEP10000_FAST",
        original_pid=args.original_pid,
        planned_original_total_steps=200000,
        user_requested_stop_step=12500,
    )
    while True:
        if not _alive(args.original_pid):
            raise RuntimeError("original V8 process exited before step10000 FAST completed")
        live = _json(output / "status.json")
        evaluator_completed = (
            live.get("status") == "COMPLETED"
            and live.get("validation_mode") == "FAST"
            and int(live.get("training_step", -1)) == 10000
        )
        trainer_acknowledged = (
            live.get("status") == "TRAINING"
            and int(live.get("training_step", -1)) >= 10000
            and live.get("latest_validation", {}).get("mode") == "FAST"
        )
        fast_complete = fast_report.is_file() and (evaluator_completed or trainer_acknowledged)
        if fast_complete:
            break
        time.sleep(max(args.poll_seconds, 0.05))
    checkpoint_sha = _sha(checkpoint_10k)
    checkpoint = torch.load(checkpoint_10k, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) != 10000:
        raise RuntimeError("step10000 checkpoint payload is incomplete")
    if "optimizer_state_dict" not in checkpoint or "rng_states" not in checkpoint:
        raise RuntimeError("step10000 checkpoint lacks optimizer or RNG state")
    _status(
        orchestration_status,
        status="SENDING_NORMAL_CTRL_C_AFTER_FAST",
        original_pid=args.original_pid,
        step10000_checkpoint_sha256=checkpoint_sha,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "send_windows_ctrl_c.py"),
            "--pid",
            str(args.original_pid),
            "--settle-seconds",
            "2",
        ],
        cwd=ROOT,
        check=True,
        timeout=30,
    )
    deadline = time.time() + 300
    while _alive(args.original_pid):
        if time.time() > deadline:
            raise RuntimeError("original V8 process did not exit after normal Ctrl+C")
        time.sleep(0.5)
    interrupted_status = _json(output / "status.json")
    interruption_evidence = {
        "schema_version": "mcvr-v8-normal-interruption-evidence-v1",
        "original_pid": args.original_pid,
        "delivery": "Windows CTRL_C_EVENT",
        "force_kill_used": False,
        "step10000_checkpoint_sha256": checkpoint_sha,
        "fast_validation_report": str(fast_report),
        "fast_validation_status": "COMPLETED",
        "status_after_interrupt": interrupted_status,
        "interrupted_at": datetime.now(timezone.utc).isoformat(),
        **ISOLATION,
    }
    atomic_json(output / "control" / "normal_interruption_evidence.json", interruption_evidence)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_ecir_mvr_v8.py"),
        "--config",
        str(ROOT / "configs" / "ecir_mvr_v8_full_v1_formal_large_200k.yaml"),
        "--output-dir",
        str(output),
        "--steps",
        "200000",
        "--resume",
        str(checkpoint_10k),
        "--validation-batches",
        "625",
        "--device",
        args.device,
    ]
    creation_flags = 0x08000000 if os.name == "nt" else 0
    with (output / "stdout.log").open("a", encoding="utf-8") as stdout, (
        output / "stderr.log"
    ).open("a", encoding="utf-8") as stderr:
        resumed = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            creationflags=creation_flags,
        )
        _atomic_text(output / "pid.txt", f"{resumed.pid}\n")
        _atomic_text(output / "resume_to_12p5k_pid.txt", f"{resumed.pid}\n")
        _status(
            orchestration_status,
            status="RESUMED_FROM_STEP10000_TO_12500",
            original_pid=args.original_pid,
            resumed_pid=resumed.pid,
            step10000_checkpoint_sha256=checkpoint_sha,
        )
        return_code = resumed.wait()
    if return_code != 0:
        raise RuntimeError(f"12.5K graceful-stop runner exited with code {return_code}")
    final_status = _json(output / "status.json")
    expected_status = "MCVR_V8_FULL_V1_FORMAL_LARGE_12P5K_COMPLETED"
    if final_status.get("status") != expected_status:
        raise RuntimeError("12.5K final status is not the requested completion status")
    if int(final_status.get("actual_completed_step", -1)) != 12500:
        raise RuntimeError("12.5K runner did not stop at the exact requested step")
    step_checkpoint = output / "checkpoints" / "step012500.ckpt"
    last_checkpoint = output / "checkpoints" / "last.ckpt"
    step_sha = _sha(step_checkpoint)
    last_sha = _sha(last_checkpoint)
    if step_sha != last_sha:
        raise RuntimeError("step012500 and last checkpoint bytes differ")
    final_checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
    if int(final_checkpoint.get("step", -1)) != 12500:
        raise RuntimeError("final checkpoint step changed")
    for key in ("optimizer_state_dict", "rng_states", "sampler_state", "scheduler_state_dict"):
        if key not in final_checkpoint:
            raise RuntimeError(f"final checkpoint missing state: {key}")
    full_report_path = output / "validation_cache" / "step012500" / "full" / "evaluation.json"
    paired_path = (
        output
        / "validation_cache"
        / "step012500"
        / "full"
        / "paired_baseline_comparison.json"
    )
    full_report = _json(full_report_path)
    paired = _json(paired_path)
    result = {
        "schema_version": "mcvr-v8-12p5k-stop-orchestration-result-v1",
        "status": expected_status,
        "original_pid": args.original_pid,
        "resumed_pid": resumed.pid,
        "original_process_interruption": "Windows CTRL_C_EVENT after completed step10000 FAST",
        "force_kill_used": False,
        "resumed_process_return_code": return_code,
        "resumed_process_normal_exit": True,
        "planned_original_total_steps": 200000,
        "user_requested_stop_step": 12500,
        "actual_completed_step": 12500,
        "effective_batch": 64,
        "total_record_exposure": 800000,
        "equivalent_old_batch8_steps": 100000,
        "schedule_provenance": "checkpoint_from_original_200k_schedule",
        "step10000_checkpoint_sha256": checkpoint_sha,
        "step012500_checkpoint_sha256": step_sha,
        "last_checkpoint_sha256": last_sha,
        "full_validation": {
            "status": full_report["status"],
            "metrics": full_report["metrics"],
            "set_metrics": full_report.get("set_metrics"),
            "paired_baseline_status": paired["status"],
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **ISOLATION,
    }
    atomic_json(orchestration_result, result)
    _status(
        orchestration_status,
        status="COMPLETED",
        final_run_status=expected_status,
        actual_completed_step=12500,
        step012500_checkpoint_sha256=step_sha,
        resumed_process_normal_exit=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        try:
            if "--output-dir" in sys.argv:
                output_arg = Path(sys.argv[sys.argv.index("--output-dir") + 1]).resolve()
                _status(
                    output_arg / "control" / "orchestration_status.json",
                    status="FAILED_CLOSED",
                    error=str(error),
                )
        except BaseException:
            pass
        raise
