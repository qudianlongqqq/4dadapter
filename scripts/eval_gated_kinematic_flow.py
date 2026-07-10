#!/usr/bin/env python
"""Fair-cohort evaluator entry point for Gated Kinematic samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

try:
    from eval_flexbond_optimizer import main as shared_evaluator
except ModuleNotFoundError:
    from scripts.eval_flexbond_optimizer import main as shared_evaluator
from etflow.commons.run_state import update_run_state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True); parser.add_argument("--inference_cache",required=True)
    parser.add_argument("--reference_cache",required=True); parser.add_argument("--split",default="test")
    parser.add_argument("--gated_samples",required=True); parser.add_argument("--output_dir",required=True)
    parser.add_argument("--threshold",type=float,default=1.25)
    args=parser.parse_args()
    output_dir=Path(args.output_dir)
    if (output_dir/"summary.csv").exists():
        raise FileExistsError("Refusing to overwrite an existing evaluation summary")
    update_run_state(output_dir,"started",stage="evaluation",expected_outputs=["summary.csv","summary.json"])
    sys.argv=[sys.argv[0],"--manifest",args.manifest,"--inference_cache",args.inference_cache,
        "--reference_cache",args.reference_cache,"--split",args.split,"--gated_samples",args.gated_samples,
        "--output_dir",args.output_dir,"--threshold",str(args.threshold)]
    try:
        shared_evaluator()
        if not (output_dir/"summary.csv").is_file() or not (output_dir/"summary.json").is_file():
            raise RuntimeError("Evaluation ended without expected outputs")
        update_run_state(output_dir,"completed",stage="evaluation")
    except KeyboardInterrupt:
        (output_dir/"STOPPED_REASON.txt").write_text("KeyboardInterrupt\n",encoding="utf-8")
        update_run_state(output_dir,"stopped",stage="evaluation");raise
    except Exception as exc:
        update_run_state(output_dir,"failed",stage="evaluation",error=repr(exc));raise


if __name__=="__main__": main()
