#!/usr/bin/env python
"""Evaluate an external-refinement cache with the frozen V8 evaluator."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

ROOT = bootstrap()

from etflow.ecir.external_refinement_baselines import ISOLATION
from etflow.ecir.v8_validation_cache import atomic_json, iter_prediction_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--source-cache-manifest", type=Path, required=True)
    parser.add_argument("--validity-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("FAST", "FULL"), required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    command = [
        sys.executable, str(ROOT / "scripts/evaluate_ecir_mvr_v8_prediction_cache.py"),
        "--prediction-manifest", str(args.prediction_manifest.resolve()),
        "--source-cache-manifest", str(args.source_cache_manifest.resolve()),
        "--validity-statistics", str(args.validity_statistics.resolve()),
        "--output", str(args.output.resolve()), "--mode", args.mode,
        "--bootstrap-draws", str(args.bootstrap_draws),
    ]
    subprocess.run(command, check=True)
    records = list(iter_prediction_records(args.prediction_manifest.resolve()))
    result = json.loads(args.output.resolve().read_text(encoding="utf-8"))
    reasons = Counter(str(row["failure_reason"]) for row in records if row.get("failure_reason"))
    runtimes = [float(row["runtime_seconds"]) for row in records]
    result["external_refinement"] = {
        "method": records[0]["method"] if records else None,
        "records": len(records),
        "successful_records": sum(int(row["success"]) for row in records),
        "fallback_records": sum(int(row["fallback_to_source"]) for row in records),
        "timeout_records": sum(int(row.get("timeout", False)) for row in records),
        "unsupported_records": sum(int(row.get("unsupported", False)) for row in records),
        "convergence_failed_records": sum(int(not row["converged"] and not row.get("timeout", False) and not row.get("unsupported", False)) for row in records),
        "success_rate": sum(int(row["success"]) for row in records) / len(records),
        "fallback_rate": sum(int(row["fallback_to_source"]) for row in records) / len(records),
        "seconds_per_record_mean": sum(runtimes) / len(runtimes),
        "native_energy_delta_mean_success_only": sum(float(row["native_energy_delta"]) for row in records if row.get("success") and row.get("native_energy_delta") is not None) / max(1, sum(int(row.get("success") and row.get("native_energy_delta") is not None) for row in records)),
        "failure_reasons": dict(reasons),
        "all_record_deployment_semantics": True,
    }
    result.update(ISOLATION)
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"status": "COMPLETED", **result["external_refinement"]}, indent=2))


if __name__ == "__main__":
    main()
