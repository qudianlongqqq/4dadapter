#!/usr/bin/env python
"""Local resumable supervisor for frozen SIXS-U cross-upstream evaluation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/ecir_mvr/sixs_final_cross_upstream_unrestricted"
ASSET = Path(r"E:\3dconformergenerationcode\dataset\sixs_final_cross_upstream_unrestricted")
STATUS = REPORT / "SUPERVISOR_STATUS.json"
PROTOCOL = ROOT / "configs/sixs_final_cross_upstream_unrestricted.json"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def status(stage: str, state: str = "RUNNING", **extra) -> None:
    current = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    current.update({
        "schema_version": "sixs-final-cross-upstream-unrestricted-supervisor-v1",
        "status": state, "stage": stage, "supervisor_pid": os.getpid(),
        "updated_at_epoch": time.time(), "zero_shot": True,
        "model_training_performed": False, "model_changed": False,
        "one_gpu_heavy_job_at_a_time": True, "repeated_polling": False,
        "mmff94s_repair_in_this_run": False, **extra,
    })
    atomic_json(STATUS, current)


def json_pass(path: Path, accepted=("PASS", "COMPLETE", "COMPLETED", "FROZEN_BEFORE_ENDPOINTS")) -> bool:
    if not path.is_file():
        return False
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status")) in accepted
    except Exception:
        return False


def run(name: str, executable: Path, arguments: list[str], marker: Path | None = None) -> None:
    if marker is not None and json_pass(marker):
        status(name, reused_completed=True)
        return
    logs = REPORT / "stage_logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout = logs / f"{name}.stdout.log"
    stderr = logs / f"{name}.stderr.log"
    status(name)
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        completed = subprocess.run([str(executable), *arguments], cwd=ROOT, stdout=out, stderr=err, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}; see {stderr}")
    if marker is not None and not json_pass(marker):
        raise RuntimeError(f"{name} completion marker missing or invalid: {marker}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-python", type=Path, default=Path(r"E:\miniconda\envs\etflow-5080-v2\python.exe"))
    parser.add_argument("--external-python", type=Path, default=Path(r"E:\miniconda\envs\external-validity\python.exe"))
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    driver = ROOT / "scripts/run_sixs_final_cross_upstream_unrestricted.py"
    external = ROOT / "scripts/evaluate_sixs_final_cross_upstream_external.py"
    xtb = ROOT / "scripts/run_sixs_final_cross_upstream_xtb.py"
    try:
        run("PREFLIGHT", args.cuda_python, [str(driver), "preflight"], REPORT / "01_PREFLIGHT.json")
        run("SMOKE", args.cuda_python, [str(driver), "smoke"], REPORT / "02_ENGINEERING_SMOKE_STATUS.json")
        for upstream in ("avgflow", "ditmc"):
            branch_report = REPORT / upstream
            branch_asset = ASSET / upstream
            run(f"{upstream.upper()}_GENERATE", args.cuda_python, [str(driver), "generate", "--upstream", upstream], branch_asset / "COORDINATE_FREEZE.json")
            run(f"{upstream.upper()}_EXTERNAL", args.external_python, [str(external), "--upstream", upstream], branch_asset / "ENDPOINT_COMPLETION.json")
            run(f"{upstream.upper()}_FIDELITY", args.cuda_python, [str(driver), "fidelity", "--upstream", upstream], branch_asset / "FIDELITY_COMPLETION.json")
        for upstream in ("avgflow", "ditmc"):
            branch_report = REPORT / upstream
            branch_asset = ASSET / upstream
            # The frozen xTB worker imports the scientific checkpoint module for
            # protocol hashing/identity checks, so it requires the project CUDA
            # environment (which contains torch).  xTB itself remains the same
            # external 6.7.1 executable and performs no neural computation here.
            run(
                f"{upstream.upper()}_XTB", args.cuda_python,
                [str(xtb), "--protocol", str(PROTOCOL), "--coordinate-dir", str(branch_asset), "--output-dir", str(branch_report / "xtb_singlepoint"), "--report-dir", str(branch_report)],
                branch_report / "XTB_FINAL_STATUS.json",
            )
            run(f"{upstream.upper()}_SUMMARIZE", args.cuda_python, [str(driver), "summarize", "--upstream", upstream], branch_report / "RESULT.json")
        run("COMBINED", args.cuda_python, [str(driver), "combined"], REPORT / "FINAL_STATUS.json")
        status("COMPLETE", "PASS", avgflow_status="PASS", ditmc_status="PASS", ablation_status="PASS_REUSED_EXISTING_ARTIFACTS_WITH_SCOPE_GUARDS")
        return 0
    except Exception as error:
        status("FAILED", "FAIL", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
