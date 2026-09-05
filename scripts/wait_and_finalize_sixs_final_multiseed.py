#!/usr/bin/env python3
"""Event-wait for the serial training supervisor, then run the frozen finalizer."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/ecir_mvr/sixs_final_restricted_vs_unrestricted_multiseed"
CUDA_PYTHON = Path(r"E:\miniconda\envs\etflow-5080-v2\python.exe")


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--supervisor-pid",type=int,required=True); args=parser.parse_args()
    psutil.Process(args.supervisor_pid).wait()
    state=json.loads((REPORT/"RUN_STATUS.json").read_text(encoding="utf-8"))
    if state.get("status")!="PASS" or state.get("stage")!="ALL_NEW_RUNS_TRAINING_AND_FROZEN_DEV_COMPLETE":
        raise RuntimeError(f"training supervisor did not complete cleanly: {state}")
    with (REPORT/"FINALIZER_STDOUT.log").open("a",encoding="utf-8") as out, (REPORT/"FINALIZER_STDERR.log").open("a",encoding="utf-8") as err:
        done=subprocess.run([str(CUDA_PYTHON),str(ROOT/"scripts/finalize_sixs_final_multiseed_replication.py")],cwd=ROOT,stdout=out,stderr=err)
    if done.returncode!=0: raise RuntimeError(f"multiseed finalizer failed: {done.returncode}")
    return 0


if __name__=="__main__": raise SystemExit(main())
