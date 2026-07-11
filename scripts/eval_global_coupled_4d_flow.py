#!/usr/bin/env python
"""Fair-cohort evaluator entry point for Global Coupled 4D samples."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--inference_cache", required=True)
    parser.add_argument("--reference_cache", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=1.25)
    args = parser.parse_args()
    summary = args.output_dir / "summary.csv"
    if summary.exists() and summary.stat().st_size:
        raise FileExistsError("refusing to overwrite an existing evaluation")
    update_run_state(args.output_dir, "started", stage="evaluation")
    sys.argv = [sys.argv[0], "--manifest", args.manifest,
                "--inference_cache", args.inference_cache,
                "--reference_cache", args.reference_cache, "--split", args.split,
                "--global_coupled_4d_samples", args.samples,
                "--output_dir", str(args.output_dir), "--threshold", str(args.threshold)]
    try:
        shared_evaluator()
        expected = [args.output_dir / "summary.csv", args.output_dir / "summary.json"]
        if not all(path.is_file() and path.stat().st_size for path in expected):
            raise RuntimeError("evaluation ended without complete summaries")
        update_run_state(args.output_dir, "completed", stage="evaluation")
    except Exception as exc:
        update_run_state(args.output_dir, "failed", stage="evaluation", error=repr(exc))
        raise


if __name__ == "__main__":
    main()

