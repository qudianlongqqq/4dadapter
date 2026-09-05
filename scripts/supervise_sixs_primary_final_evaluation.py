#!/usr/bin/env python
"""Local resumable supervisor for the frozen SIXS primary-final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_RECORDS = 5000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def status(path: Path, stage: str, state: str = "RUNNING", **extra: Any) -> None:
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    current.update({"schema_version": "sixs-primary-final-supervisor-v1", "status": state, "stage": stage, "pid": os.getpid(), "repeated_polling": False, **extra})
    atomic_json(path, current)


def run(command: list[str], cwd: Path) -> None:
    print("SUPERVISOR_EXEC " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def complete_external(path: Path, ids: list[str]) -> bool:
    if not path.is_file():
        return False
    frame = pd.read_parquet(path)
    return len(frame) == EXPECTED_RECORDS and frame.record_id.astype(str).tolist() == ids


def main(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    report = args.report_dir.resolve()
    output = args.output_dir.resolve()
    status_path = report / "SUPERVISOR_STATUS.json"
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    methods = ["source"] + [method["id"] for method in protocol["model_methods"]] + ["mmff94s"]
    try:
        status(status_path, "COORDINATES")
        run([
            str(args.cuda_python), str(repo / "scripts/run_sixs_primary_final_coordinates.py"),
            "--stage", "pipeline", "--protocol", str(args.protocol), "--primary", str(args.primary),
            "--source-manifest", str(args.source_manifest), "--source-asset-freeze", str(args.source_asset_freeze),
            "--output-dir", str(output), "--report-dir", str(report),
        ], repo)

        source_records = pd.read_parquet(output / "methods/source/PER_RECORD.parquet")
        ids = source_records.record_id.astype(str).tolist()
        if len(ids) != EXPECTED_RECORDS or len(set(ids)) != EXPECTED_RECORDS:
            raise RuntimeError("coordinate record denominator mismatch")

        for index, method in enumerate(methods, start=1):
            target = output / "methods" / method
            pb = target / "POSEBUSTERS.parquet"
            v3d = target / "VALIDITY3D.parquet"
            if complete_external(pb, ids) and complete_external(v3d, ids):
                continue
            status(status_path, "EXTERNAL_VALIDITY", method=method, method_index=index, method_total=len(methods))
            run([
                str(args.external_python), str(repo / "scripts/evaluate_sixs_primary_final_external.py"),
                "--arm", method, "--sdf", str(target / "COORDINATES.sdf"),
                "--records", str(target / "PER_RECORD.parquet"), "--pb", str(pb), "--v3d", str(v3d),
            ], repo)

        status(status_path, "XTB_SINGLE_POINT")
        run([
            str(args.cuda_python), str(repo / "scripts/run_sixs_primary_final_xtb.py"),
            "--protocol", str(args.protocol), "--coordinate-dir", str(output),
            "--output-dir", str(output / "xtb_single_point"), "--report-dir", str(report),
        ], repo)
        status(status_path, "FINAL_STATISTICAL_ANALYSIS")
        run([
            str(args.cuda_python), str(repo / "scripts/finalize_sixs_primary_final_evaluation.py"),
            "--protocol", str(args.protocol), "--output-dir", str(output),
            "--report-dir", str(report),
        ], repo)
        status(
            status_path, "RAW_SCIENTIFIC_EVALUATION_COMPLETE", state="PASS",
            methods=len(methods), records_per_method=EXPECTED_RECORDS,
            protocol_sha256=sha256_file(args.protocol), protected_final_outcome_opened=True,
            model_training_performed=False, outcome_dependent_rerun=False,
        )
        return 0
    except BaseException as exc:
        (report / "SUPERVISOR_TRACEBACK.txt").write_text(traceback.format_exc(), encoding="utf-8")
        status(status_path, "ENGINEERING_FAILURE", state="FAIL", error_type=type(exc).__name__, error=str(exc))
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-asset-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--cuda-python", type=Path, required=True)
    parser.add_argument("--external-python", type=Path, required=True)
    args = parser.parse_args()
    for name in ("repo", "protocol", "primary", "source_manifest", "source_asset_freeze", "output_dir", "report_dir", "cuda_python", "external_python"):
        setattr(args, name, getattr(args, name).resolve())
    return args


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
