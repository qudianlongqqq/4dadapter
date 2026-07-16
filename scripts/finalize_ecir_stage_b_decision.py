#!/usr/bin/env python
"""Recompute the Stage B schedule check from completed sweep outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import pandas as pd

from etflow.ecir.stage_b_decision import compare_train_range_to_legacy


def _json_finite(value):
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.results)
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    passed, diagnostics = compare_train_range_to_legacy(frame)
    check = "train_range_nonworse_than_legacy_full"
    passed_checks = set(decision["passed_checks"])
    failed_checks = set(decision["failed_checks"])
    (passed_checks if passed else failed_checks).add(check)
    (failed_checks if passed else passed_checks).discard(check)
    decision["passed_checks"] = sorted(passed_checks)
    decision["failed_checks"] = sorted(failed_checks)
    decision["schedule_comparison"] = diagnostics
    if not failed_checks:
        decision["decision"] = "EXISTING_CKPT_RESCUED"
    elif decision["true_validity_metrics_with_ci_improvement"]:
        decision["decision"] = "EXISTING_CKPT_DIAGNOSTIC_ONLY"
    else:
        decision["decision"] = "EXISTING_CKPT_NOT_RESCUED"
    decision = _json_finite(decision)
    args.decision.write_text(
        json.dumps(decision, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": decision["decision"], **diagnostics}, indent=2))


if __name__ == "__main__":
    main()
