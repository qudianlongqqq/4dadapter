#!/usr/bin/env python3
"""Resume the frozen final multiseed order after restricted seed331 recovery.

The scientific runners are invoked unchanged.  Restricted runs use the
orchestration-only posttrain recovery entry point after the exact train stage,
because the frozen restricted pipeline's initial-evaluation control flow calls
the existing-coordinate preflight too early.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import torch

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()
import scripts.run_sixs_final_multiseed_replication as original

REPORT = original.REPORT
ARTIFACT = original.ARTIFACT
STATUS = original.STATUS
CUDA_PYTHON = original.CUDA_PYTHON
RECOVERY_RUNS = (("unrestricted", 331), ("restricted", 353), ("unrestricted", 353))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def update(stage: str, state: str = "RUNNING", **extra: Any) -> None:
    previous = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    previous.update({
        "schema_version": "sixs-final-multiseed-recovery-supervisor-v1",
        "status": state,
        "stage": stage,
        "pid": os.getpid(),
        "updated_at_epoch": time.time(),
        "cuda_python": str(CUDA_PYTHON),
        "one_gpu_training_at_a_time": True,
        "MULTISEED_RESUMED": "YES",
        "NO_MODEL_CHANGE": "YES",
        "NO_RETRAINING_OF_RESTRICTED_SEED331": "YES",
        "NO_OUTCOME_BASED_SELECTION": "YES",
        "formal_outcome_read": False,
        "large_holdout_outcome_read": False,
        **extra,
    })
    atomic_json(STATUS, previous)


def env_for(formulation: str, seed: int, config: Path, report: Path, artifact: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "PYTHONUTF8": "1",
        "SIXS_FULL_JOINT_DEVICE": "cuda",
        "SIXS_MULTISEED_REPLICATION": "1",
        "SIXS_MULTISEED_CONFIG": str(config),
        "SIXS_MULTISEED_REPORT": str(report),
        "SIXS_MULTISEED_ARTIFACT": str(artifact),
        "SIXS_MULTISEED_RUN_NAME": f"sixs_final_multiseed/{formulation}_seed{seed}",
        "SIXS_MULTISEED_RUN_ROOT": str(report.parent),
        "SIXS_MULTISEED_EXPERIMENT_ID": f"SIXS_FINAL_{formulation.upper()}_SEED{seed}_STEP17500",
    })
    return environment


def run_command(command: list[str], environment: dict[str, str], stdout_path: Path, stderr_path: Path) -> None:
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"worker failed with exit code {completed.returncode}: {' '.join(command)}")


def run_one(formulation: str, seed: int) -> None:
    report = REPORT / f"{formulation}_seed{seed}"
    artifact = ARTIFACT / f"{formulation}_seed{seed}"
    config = original.make_config(formulation, seed)
    report.mkdir(parents=True, exist_ok=True)
    artifact.mkdir(parents=True, exist_ok=True)
    if original.child_complete(formulation, report):
        update(f"SKIP_COMPLETE_{formulation.upper()}_SEED{seed}", run=f"{formulation}_seed{seed}")
        return
    environment = env_for(formulation, seed, config, report, artifact)
    stdout_path = report / "PIPELINE_STDOUT.log"
    stderr_path = report / "PIPELINE_STDERR.log"
    if formulation == "unrestricted":
        runner = ROOT / "scripts/run_sixs_j1r1_full_joint_unrestricted_movement.py"
        update(f"RUN_UNRESTRICTED_SEED{seed}", run=f"unrestricted_seed{seed}", runner_sha256=sha(runner), config_sha256=sha(config))
        run_command([str(CUDA_PYTHON), str(runner), "--stage", "pipeline"], environment, stdout_path, stderr_path)
    else:
        runner = ROOT / "scripts/run_sixs_j1r1_full_joint_adaptive_ba_movement.py"
        recovery = ROOT / "scripts/recover_sixs_restricted_multiseed_posttrain.py"
        update(f"TRAIN_RESTRICTED_SEED{seed}", run=f"restricted_seed{seed}", runner_sha256=sha(runner), config_sha256=sha(config))
        run_command([str(CUDA_PYTHON), str(runner), "--stage", "train"], environment, stdout_path, stderr_path)
        update(f"POSTTRAIN_RECOVERY_RESTRICTED_SEED{seed}", run=f"restricted_seed{seed}", recovery_wrapper_sha256=sha(recovery))
        run_command([str(CUDA_PYTHON), str(recovery), "--seed", str(seed), "--skip-xtb"], environment, stdout_path, stderr_path)
    if not original.child_complete(formulation, report):
        raise RuntimeError(f"{formulation} seed{seed} returned without frozen DEV completion")


def main() -> int:
    recovered = REPORT / "restricted_seed331/POSTTRAIN_RECOVERY_AUDIT.json"
    if not recovered.is_file():
        raise RuntimeError("restricted seed331 recovery audit is missing")
    recovery_status = json.loads(recovered.read_text(encoding="utf-8"))
    if recovery_status.get("status") != "PASS" or recovery_status.get("XTB_COMPLETE") != "YES":
        raise RuntimeError("restricted seed331 recovery is not complete")
    if not original.child_complete("restricted", REPORT / "restricted_seed331"):
        raise RuntimeError("restricted seed331 frozen DEV status is incomplete")
    gate = original.gpu_gate()
    update("RECOVERY_SUPERVISOR_STARTED", "RUNNING", GPU_TRAINING_READY="YES", restricted_seed331_recovery_sha256=sha(recovered), **gate)
    try:
        for formulation, seed in RECOVERY_RUNS:
            run_one(formulation, seed)
        update("ALL_NEW_RUNS_TRAINING_AND_FROZEN_DEV_COMPLETE", "PASS", completed_runs=["restricted_seed331", *(f"{form}_seed{seed}" for form, seed in RECOVERY_RUNS)])
        finalizer = ROOT / "scripts/finalize_sixs_final_multiseed_replication.py"
        final_stdout = REPORT / "FINALIZER_STDOUT.log"
        final_stderr = REPORT / "FINALIZER_STDERR.log"
        update("MULTISEED_FINALIZER", "RUNNING", finalizer_sha256=sha(finalizer))
        run_command([str(CUDA_PYTHON), str(finalizer)], os.environ.copy(), final_stdout, final_stderr)
        final_status_path = REPORT / "FINAL_STATUS.json"
        final_status = json.loads(final_status_path.read_text(encoding="utf-8"))
        if final_status.get("status") != "PASS" or final_status.get("MULTISEED_STATUS") != "COMPLETE":
            raise RuntimeError("multiseed finalizer did not freeze a PASS result")
        update("COMPLETE", "PASS", MULTISEED_STATUS="COMPLETE", MULTISEED_INTEGRITY=final_status.get("MULTISEED_INTEGRITY"),
               FINAL_FORMULATION_CLASSIFICATION=final_status.get("FINAL_FORMULATION_CLASSIFICATION"))
        return 0
    except Exception as error:
        (REPORT / "RECOVERY_SUPERVISOR_TRACEBACK.txt").write_text(traceback.format_exc(), encoding="utf-8")
        update("STOPPED_ENGINEERING_FAILURE", "FAIL", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
