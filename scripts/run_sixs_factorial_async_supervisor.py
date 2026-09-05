"""Three-process async supervisor for the frozen six-arm factorial."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(r"E:\3dconformergenerationcode\4dadapter-lsgoba-musigma-reliability-factorial")
REPORT = ROOT / "reports/ecir_mvr/sixs_musigma_reliability_factorial_cuda"
MAIN = ROOT / "scripts/run_sixs_musigma_reliability_factorial.py"
CUDA_PYTHON = Path(r"E:\miniconda\envs\etflow-5080-v2\python.exe")
CPU_PYTHON = Path(r"E:\python\python.exe")
STATUS = REPORT / "ASYNC_SUPERVISOR_STATUS.json"
ARMS = ["J0-R0", "J0-R1", "J1-R0", "J1-R1", "J2-R0", "J2-R1"]


def write_status(status: str, **extra) -> None:
    payload = {
        "schema_version": "sixs-factorial-async-supervisor-v1", "status": status,
        "pid": os.getpid(), "updated_at_epoch": time.time(), "frozen_arm_order": ARMS,
        "FORMAL_READ": "NO", "LARGE_HOLDOUT_READ": "NO", "XTB_STARTED": "NO",
        "SIGMA_V2_MODIFIED": "NO", "SCIENTIFIC_METRICS_CHANGED": "NO",
        "COHORT_CHANGED": "NO", "MODEL_CHANGED": "NO", **extra,
    }
    tmp = STATUS.with_name(STATUS.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, STATUS)


def env(device: str) -> dict[str, str]:
    value = os.environ.copy()
    value["SIXS_FACTORIAL_RUN_NAMESPACE"] = "sixs_musigma_reliability_factorial_cuda"
    value["SIXS_FACTORIAL_DEVICE"] = device
    value.setdefault("OMP_NUM_THREADS", "1")
    value.setdefault("MKL_NUM_THREADS", "1")
    return value


def open_log(name: str):
    return (REPORT / name).open("a", encoding="utf-8", buffering=1)


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    evaluator_out = open_log("ASYNC_CPU_EVALUATOR_STDOUT.log")
    evaluator_err = open_log("ASYNC_CPU_EVALUATOR_STDERR.log")
    trainer_out = open_log("ASYNC_GPU_TRAINER_STDOUT.log")
    trainer_err = open_log("ASYNC_GPU_TRAINER_STDERR.log")
    finalizer_out = open_log("ASYNC_FINALIZER_STDOUT.log")
    finalizer_err = open_log("ASYNC_FINALIZER_STDERR.log")
    evaluator = trainer = None
    try:
        write_status("STARTING")
        evaluator = subprocess.Popen(
            [str(CPU_PYTHON), "-u", str(MAIN), "evaluate-queue"], cwd=ROOT,
            env=env("cpu"), stdout=evaluator_out, stderr=evaluator_err,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        trainer = subprocess.Popen(
            [str(CUDA_PYTHON), "-u", str(MAIN), "train-queue"], cwd=ROOT,
            env=env("cuda"), stdout=trainer_out, stderr=trainer_err,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        write_status("RUNNING", cpu_evaluator_pid=evaluator.pid, gpu_trainer_pid=trainer.pid)
        trainer_code = trainer.wait()
        evaluator_code = evaluator.wait()
        if trainer_code != 0 or evaluator_code != 0:
            write_status("FAIL_CLOSED", gpu_trainer_exit_code=trainer_code, cpu_evaluator_exit_code=evaluator_code)
            return 1
        write_status("FINALIZING", gpu_trainer_exit_code=trainer_code, cpu_evaluator_exit_code=evaluator_code)
        finalizer = subprocess.run(
            [str(CPU_PYTHON), "-u", str(MAIN), "finalize"], cwd=ROOT, env=env("cpu"),
            stdout=finalizer_out, stderr=finalizer_err, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if finalizer.returncode != 0:
            write_status("FAIL_CLOSED", finalizer_exit_code=finalizer.returncode)
            return 1
        write_status("COMPLETE", gpu_trainer_exit_code=trainer_code,
                     cpu_evaluator_exit_code=evaluator_code, finalizer_exit_code=finalizer.returncode)
        return 0
    except Exception as exc:
        write_status("FAIL_CLOSED", error_type=type(exc).__name__, error=str(exc),
                     cpu_evaluator_pid=None if evaluator is None else evaluator.pid,
                     gpu_trainer_pid=None if trainer is None else trainer.pid)
        return 1
    finally:
        for handle in (evaluator_out, evaluator_err, trainer_out, trainer_err, finalizer_out, finalizer_err):
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
