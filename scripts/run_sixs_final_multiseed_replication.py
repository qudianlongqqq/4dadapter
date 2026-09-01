#!/usr/bin/env python3
"""Serial GPU-only supervisor for the frozen SIXS restricted/unrestricted multiseed runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
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
REPORT = ROOT / "reports/ecir_mvr/sixs_final_restricted_vs_unrestricted_multiseed"
ARTIFACT = ROOT / "artifacts/ecir_mvr/sixs_final_restricted_vs_unrestricted_multiseed"
STATUS = REPORT / "RUN_STATUS.json"
CUDA_PYTHON = Path(r"E:\miniconda\envs\etflow-5080-v2\python.exe")
RUNS = (("restricted", 331), ("unrestricted", 331), ("restricted", 353), ("unrestricted", 353))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def update(stage: str, state: str = "RUNNING", **extra: Any) -> None:
    old = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    old.update({
        "schema_version": "sixs-final-multiseed-supervisor-v1", "status": state, "stage": stage,
        "pid": os.getpid(), "updated_at_epoch": time.time(), "cuda_python": str(CUDA_PYTHON),
        "one_gpu_training_at_a_time": True, "formal_outcome_read": False,
        "large_holdout_outcome_read": False, **extra,
    })
    atomic_json(STATUS, old)


def make_config(formulation: str, seed: int) -> Path:
    source = ROOT / "configs" / (
        "sixs_j1r1_full_joint_adaptive_ba_movement.json" if formulation == "restricted"
        else "sixs_j1r1_full_joint_unrestricted_movement.json"
    )
    data = json.loads(source.read_text(encoding="utf-8"))
    data["seed"] = seed
    data["experiment_id"] = f"SIXS_FINAL_{formulation.upper()}_SEED{seed}_STEP17500"
    if int(data["training"]["optimizer_steps"]) != 17500:
        raise RuntimeError("frozen endpoint is not 17500")
    if float(data["model"]["beta_nll_beta"]) != 0.5:
        raise RuntimeError("frozen beta changed")
    target = REPORT / f"{formulation}_seed{seed}" / "FROZEN_CONFIG.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if target.is_file() and target.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"frozen config identity changed: {target}")
    if not target.is_file():
        target.write_text(encoded, encoding="utf-8")
    return target


def child_complete(formulation: str, report: Path) -> bool:
    status_path = report / "FINAL_STATUS.json"
    final = report / "FINAL_CHECKPOINT.pt"
    if not status_path.is_file() or not final.is_file():
        return False
    value = json.loads(status_path.read_text(encoding="utf-8"))
    stage = value.get("CURRENT_STAGE")
    if formulation == "restricted":
        return value.get("PIPELINE_STATUS") == "PASS" and stage == "COMPLETE" and int(value.get("CURRENT_STEP", 0)) == 17500
    return value.get("PIPELINE_STATUS") == "PASS" and stage == "UNRESTRICTED_VALIDITY_COMPLETE" and int(value.get("CURRENT_STEP", 17500)) == 17500


def gpu_gate() -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    result = {
        "CUDA_AVAILABLE": available, "CUDA_DEVICE_COUNT": int(torch.cuda.device_count()) if available else 0,
        "CUDA_DEVICE_NAME": torch.cuda.get_device_name(0) if available else None,
        "PYTORCH_CUDA_VERSION": torch.version.cuda, "MODEL_DEVICE": "cuda" if available else None,
        "TRAINING_DEVICE": "cuda" if available else None,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<UNSET>"),
    }
    if not available:
        update("STOP_BEFORE_TRAINING", "FAIL", GPU_TRAINING_READY="NO", **result)
        raise RuntimeError("CUDA unavailable; CPU fallback forbidden")
    return result


def run_child(formulation: str, seed: int) -> None:
    run_report = REPORT / f"{formulation}_seed{seed}"
    run_artifact = ARTIFACT / f"{formulation}_seed{seed}"
    config = make_config(formulation, seed)
    run_report.mkdir(parents=True, exist_ok=True)
    run_artifact.mkdir(parents=True, exist_ok=True)
    if child_complete(formulation, run_report):
        update(f"SKIP_COMPLETE_{formulation.upper()}_SEED{seed}", run=f"{formulation}_seed{seed}")
        return
    script = ROOT / "scripts" / (
        "run_sixs_j1r1_full_joint_adaptive_ba_movement.py" if formulation == "restricted"
        else "run_sixs_j1r1_full_joint_unrestricted_movement.py"
    )
    env = os.environ.copy()
    env.update({
        "PYTHONUTF8": "1", "SIXS_FULL_JOINT_DEVICE": "cuda", "SIXS_MULTISEED_REPLICATION": "1",
        "SIXS_MULTISEED_CONFIG": str(config), "SIXS_MULTISEED_REPORT": str(run_report),
        "SIXS_MULTISEED_ARTIFACT": str(run_artifact),
        "SIXS_MULTISEED_RUN_NAME": f"sixs_final_multiseed/{formulation}_seed{seed}",
        "SIXS_MULTISEED_RUN_ROOT": str(run_report.parent),
        "SIXS_MULTISEED_EXPERIMENT_ID": f"SIXS_FINAL_{formulation.upper()}_SEED{seed}_STEP17500",
    })
    stdout_path, stderr_path = run_report / "PIPELINE_STDOUT.log", run_report / "PIPELINE_STDERR.log"
    update(f"RUN_{formulation.upper()}_SEED{seed}", run=f"{formulation}_seed{seed}", config_sha256=sha(config))
    with stdout_path.open("a", encoding="utf-8") as out, stderr_path.open("a", encoding="utf-8") as err:
        completed = subprocess.run([str(CUDA_PYTHON), str(script), "--stage", "pipeline"], cwd=ROOT, env=env, stdout=out, stderr=err)
    if completed.returncode != 0:
        raise RuntimeError(f"{formulation} seed{seed} worker failed with exit code {completed.returncode}")
    if not child_complete(formulation, run_report):
        raise RuntimeError(f"{formulation} seed{seed} returned without frozen completion")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", choices=("training-evaluation",), default="training-evaluation")
    parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True); ARTIFACT.mkdir(parents=True, exist_ok=True)
    gate = gpu_gate(); atomic_json(REPORT / "GPU_PREFLIGHT.json", {**gate, "GPU_TRAINING_READY": "YES"})
    update("SUPERVISOR_STARTED", GPU_TRAINING_READY="YES", **gate)
    try:
        for formulation, seed in RUNS:
            run_child(formulation, seed)
        update("ALL_NEW_RUNS_TRAINING_AND_FROZEN_DEV_COMPLETE", "PASS", completed_runs=[f"{f}_seed{s}" for f, s in RUNS])
        return 0
    except Exception as error:
        trace = traceback.format_exc()
        (REPORT / "SUPERVISOR_TRACEBACK.txt").write_text(trace, encoding="utf-8")
        update("STOPPED_ENGINEERING_FAILURE", "FAIL", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
