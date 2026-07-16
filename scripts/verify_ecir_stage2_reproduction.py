#!/usr/bin/env python
"""Attach deterministic identity and drift checks to an ECIR reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from etflow.ecir.audit import file_sha256


REQUIRED = (
    "bond_violation", "angle_violation", "torsion_circular_error",
    "ring_invalidity", "aligned_RMSD", "MAT_P", "MAT_R",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--baseline_result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--absolute_tolerance", type=float, default=5.0e-7)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    deltas = {}
    for metric in REQUIRED:
        old = float(baseline["bootstrap_delta_candidate_minus_upstream"][metric]["mean"])
        new = float(result["bootstrap_delta_candidate_minus_upstream"][metric]["mean"])
        deltas[metric] = {"baseline": old, "reproduced": new, "absolute_delta": abs(new - old)}
    passed = all(value["absolute_delta"] <= args.absolute_tolerance for value in deltas.values())
    result["reproduction"] = {
        "status": "PASS" if passed else "FAIL",
        "absolute_tolerance": args.absolute_tolerance,
        "metrics": deltas,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "atlas_sha256": file_sha256(args.atlas),
        "baseline_result_sha256": file_sha256(args.baseline_result),
        "test_used": False,
    }
    args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not passed:
        raise SystemExit("Stage A reproduction failed")
    print(json.dumps(result["reproduction"], indent=2))


if __name__ == "__main__":
    main()
