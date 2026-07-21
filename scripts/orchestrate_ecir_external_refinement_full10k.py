#!/usr/bin/env python
"""Wait for xTB FULL10K and finalize all frozen external-baseline reports."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

import psutil

from etflow.ecir.external_refinement_baselines import ISOLATION
from etflow.ecir.v8_validation_cache import atomic_json, file_sha256, utc_now


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    diagnostics = ROOT / "diagnostics/ecir_mvr/external_refinement_baselines/formal_large_seed43"
    status_path = diagnostics / "full10k_orchestration.json"
    runner_status_path = diagnostics / "gfn2_xtb/status.json"
    report_dir = ROOT / "reports/ecir_mvr/external_refinement_baselines"
    started = time.perf_counter()

    def update(status: str, phase: str, **values) -> None:
        atomic_json(status_path, {
            "schema_version": "mcvr-external-refinement-full10k-orchestration-v1",
            "status": status,
            "phase": phase,
            "runner_pid": args.runner_pid,
            "runner_process_alive": psutil.pid_exists(args.runner_pid),
            "elapsed_seconds": time.perf_counter() - started,
            "last_update_time": utc_now(),
            **ISOLATION,
            **values,
        })

    try:
        update("FULL10K_RUNNING", "WAITING_FOR_GFN2_XTB")
        while psutil.pid_exists(args.runner_pid):
            runner = json.loads(runner_status_path.read_text(encoding="utf-8"))
            update(
                "FULL10K_RUNNING", "WAITING_FOR_GFN2_XTB",
                completed_records=runner.get("completed_records"),
                successful_records=runner.get("successful_records"),
                fallback_records=runner.get("fallback_records"),
                estimated_remaining_seconds=runner.get("estimated_remaining_seconds"),
            )
            time.sleep(max(1.0, args.poll_seconds))
        runner = json.loads(runner_status_path.read_text(encoding="utf-8"))
        if runner.get("status") != "FULL10K_COMPLETED" or runner.get("completed_records") != 10000:
            raise RuntimeError(f"xTB runner did not complete safely: {runner.get('status')}")
        manifest = diagnostics / "gfn2_xtb/full10k/prediction_manifest.json"
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        if manifest_payload.get("status") != "COMPLETED" or manifest_payload.get("records_written") != 10000:
            raise RuntimeError("xTB FULL10K prediction cache is incomplete")

        py = sys.executable
        source = ROOT / "diagnostics/ecir_mvr/validation_cache/formal_large_seed43/source/prediction_manifest.json"
        validity = ROOT / "data/ecir_mvr/validity_reference_stats.json"
        xtb_evaluation = diagnostics / "gfn2_xtb/full10k/evaluation.json"
        update("FULL10K_RUNNING", "EVALUATING_GFN2_XTB")
        run([py, "scripts/evaluate_ecir_external_refinement_baselines.py", "--prediction-manifest", str(manifest), "--source-cache-manifest", str(source), "--validity-statistics", str(validity), "--output", str(xtb_evaluation), "--mode", "FULL", "--bootstrap-draws", "10000"])
        xtb_augmented = diagnostics / "augmented/gfn2_xtb/evaluation.json"
        run([py, "scripts/augment_ecir_evaluation_diversity.py", "--evaluation", str(xtb_evaluation), "--prediction-manifest", str(manifest), "--output", str(xtb_augmented)])

        update("FULL10K_RUNNING", "BUILDING_PAIRED_REPORT")
        evaluations = {
            "RAW": diagnostics / "augmented/raw/evaluation.json",
            "MMFF94S": diagnostics / "augmented/mmff94s/evaluation.json",
            "GFN2_XTB": xtb_augmented,
            "MATCHED_D1_12P5K": diagnostics / "augmented/matched_d1_12p5k/evaluation.json",
            "V8_FULL_12P5K": diagnostics / "augmented/v8_full_12p5k/evaluation.json",
            "FROZEN_D1": diagnostics / "augmented/frozen_d1/evaluation.json",
            "V5_B": diagnostics / "augmented/v5_b/evaluation.json",
            "V7": diagnostics / "augmented/v7/evaluation.json",
        }
        command = [py, "scripts/compare_ecir_external_refinement_baselines.py", "--phase", "FULL10K", "--source-cache-manifest", str(source)]
        for method, path in evaluations.items():
            command.extend(["--evaluation", f"{method}={path}"])
        command.extend(["--prediction", f"MMFF94S={diagnostics / 'mmff94s/full10k/prediction_manifest.json'}", "--prediction", f"GFN2_XTB={manifest}", "--output-dir", str(report_dir), "--bootstrap-draws", "10000"])
        run(command)
        run([py, "scripts/finalize_ecir_external_refinement_reports.py", "--report-dir", str(report_dir), "--diagnostics-root", str(diagnostics)])

        update("FULL10K_RUNNING", "VERIFYING_TESTS")
        tests = sorted(str(path) for path in (ROOT / "tests").glob("test_external_refinement_*.py"))
        run([py, "-m", "pytest", "-q", *tests])
        report = json.loads((report_dir / "FULL10K_RESULTS.json").read_text(encoding="utf-8"))
        if report.get("status") != "MCVR_EXTERNAL_REFINEMENT_FULL10K_COMPLETED":
            raise RuntimeError("final FULL10K report status changed")
        for key, expected in ISOLATION.items():
            if report.get(key) != expected:
                raise RuntimeError(f"final report isolation changed: {key}")

        update("FULL10K_RUNNING", "COMMITTING_REPORTS")
        run(["git", "add", "--", "reports/ecir_mvr/external_refinement_baselines"])
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
        if diff.returncode != 0:
            run(["git", "commit", "-m", "eval(ecir): report external refinement FULL10K"])
        run(["git", "push", "origin", "eval/mcvr-v8-external-refinement-baselines"])
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        update(
            "MCVR_EXTERNAL_REFINEMENT_FULL10K_COMPLETED", "COMPLETED",
            completed_records=10000,
            successful_records=runner["successful_records"],
            fallback_records=runner["fallback_records"],
            runner_normal_exit=True,
            prediction_manifest_sha256=file_sha256(manifest),
            full_report_sha256=file_sha256(report_dir / "FULL10K_RESULTS.json"),
            git_branch="eval/mcvr-v8-external-refinement-baselines",
            git_head=head,
            git_push_completed=True,
        )
    except BaseException as error:
        update("FAILED_CLOSED", "FAILED_CLOSED", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
