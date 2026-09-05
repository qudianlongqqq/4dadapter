#!/usr/bin/env python3
"""Blocking, non-polling supervisor for the frozen two-branch interaction run."""
from __future__ import annotations
import json, os, subprocess, time
from pathlib import Path

ROOT=Path(r"E:\3dconformergenerationcode\4dadapter-lsgoba-musigma-reliability-factorial")
REPORT=ROOT/"reports/ecir_mvr/sixs_j1r1_joint_magnitude_interaction_seed307"
MAIN=ROOT/"scripts/run_sixs_j1r1_joint_magnitude_interaction.py"
CUDA=Path(r"E:\miniconda\envs\etflow-5080-v2\python.exe")
CPU=Path(r"E:\python\python.exe")
STATUS=REPORT/"SUPERVISOR_STATUS.json"
ARMS=("J1-R0-JOINT","J1-R1-JOINT")

def write(state,**extra):
    payload={"schema_version":"sixs-joint-magnitude-supervisor-v1","status":state,"pid":os.getpid(),"updated_at_epoch":time.time(),"execution":"BLOCKING_SEQUENTIAL_NO_POLLING","arms":ARMS,"FORMAL_READ":"NO","LARGE_HOLDOUT_READ":"NO","XTB_STARTED":"NO",**extra}
    tmp=STATUS.with_name(STATUS.name+f".tmp.{os.getpid()}");tmp.write_text(json.dumps(payload,indent=2),encoding="utf-8");os.replace(tmp,STATUS)

def env(mode):
    value=os.environ.copy();value["SIXS_JOINT_DEVICE"]=mode;value["SIXS_FACTORIAL_RUN_NAMESPACE"]="sixs_musigma_reliability_factorial_cuda";value.setdefault("OMP_NUM_THREADS","1");value.setdefault("MKL_NUM_THREADS","1");return value

def run(python,args,mode,out,err):
    with (REPORT/out).open("a",encoding="utf-8",buffering=1) as stdout,(REPORT/err).open("a",encoding="utf-8",buffering=1) as stderr:
        result=subprocess.run([str(python),"-u",str(MAIN),*args],cwd=ROOT,env=env(mode),stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode: raise RuntimeError(f"stage failed {args}: {result.returncode}")

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    try:
        write("RUNNING",current_stage="J1_R0_TRAIN_COORDINATES")
        run(CUDA,["train-coordinate",ARMS[0]],"cuda","J1_R0_GPU_STDOUT.log","J1_R0_GPU_STDERR.log")
        write("RUNNING",current_stage="J1_R0_EXTERNAL_EVALUATION")
        run(CPU,["evaluate",ARMS[0]],"cpu","J1_R0_EVAL_STDOUT.log","J1_R0_EVAL_STDERR.log")
        write("RUNNING",current_stage="J1_R1_TRAIN_COORDINATES")
        run(CUDA,["train-coordinate",ARMS[1]],"cuda","J1_R1_GPU_STDOUT.log","J1_R1_GPU_STDERR.log")
        write("RUNNING",current_stage="J1_R1_EXTERNAL_EVALUATION")
        run(CPU,["evaluate",ARMS[1]],"cpu","J1_R1_EVAL_STDOUT.log","J1_R1_EVAL_STDERR.log")
        write("RUNNING",current_stage="BOOTSTRAP_INTERACTION_FINALIZER")
        run(CPU,["finalize"],"cpu","FINALIZER_STDOUT.log","FINALIZER_STDERR.log")
        write("COMPLETE",current_stage="COMPLETE");return 0
    except Exception as exc:
        write("FAIL_CLOSED",error_type=type(exc).__name__,error=str(exc));return 1

if __name__=="__main__": raise SystemExit(main())
