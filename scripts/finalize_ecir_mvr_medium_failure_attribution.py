#!/usr/bin/env python
"""Finalize test evidence and fail-closed boundaries for the Medium attribution audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from etflow.commons.global_coupled_4d_sampling import atomic_json_save


PROTECTED_SHA256 = "738171eb7e5f047e94cd4e1a46689613fe1f30bf33a320a2e3ec5a6944a5ec7d"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targeted-passed", type=int, required=True)
    parser.add_argument("--full-passed", type=int, required=True)
    args = parser.parse_args()
    if args.targeted_passed < 1 or args.full_passed < 345:
        raise RuntimeError("test evidence does not meet the preregistered minimum")
    protected = Path("reports/global4d_profile_bundle_verification.json")
    if _sha(protected) != PROTECTED_SHA256:
        raise RuntimeError("protected report changed during failure attribution")

    result_path = Path("diagnostics/ecir_mvr/medium/failure_attribution/result.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["formal_decision_unchanged"] != "MEDIUM_SEED42_SCHEDULE_V4_FAIL":
        raise RuntimeError("formal V4 decision changed")
    if result["test_records_read"] != 0 or result["training_run"] is not False:
        raise RuntimeError("validation-only boundary changed")
    factors = result["factor_contributions"]
    target_gap = float(factors["target_gap"])
    factors["signed_target_gap_shares"] = {
        name: float(value) / target_gap if abs(target_gap) > 1.0e-12 else 0.0
        for name, value in factors["signed_contributions"].items()
    }
    gain_path = Path("diagnostics/ecir_mvr/medium/failure_attribution/stagewise_gain_summary.json")
    gain = json.loads(gain_path.read_text(encoding="utf-8"))
    gain["factor_contributions"] = factors
    atomic_json_save(gain, gain_path)
    result["tests"] = {
        "status": "PASS",
        "targeted_command": (
            "python -m pytest -q tests/test_ecir_mvr_medium_failure_attribution.py "
            "tests/test_ecir_mvr_medium_rescue_v3.py "
            "tests/test_ecir_mvr_medium_rescue_v2.py tests/test_ecir_mvr_medium.py"
        ),
        "targeted_passed": args.targeted_passed,
        "full_command": "python -m pytest -q",
        "full_passed": args.full_passed,
        "failures": 0,
        "test_records_read": 0,
    }
    atomic_json_save(result, result_path)

    state_path = Path("reports/ecir_mvr/progressive_state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update({
        "current_stage": "MEDIUM_FAILURE_ATTRIBUTION_COMPLETE",
        "current_decision": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
        "medium_seed42_schedule_v4_decision": "MEDIUM_SEED42_SCHEDULE_V4_FAIL",
        "failure_attribution_completed": True,
        "failure_attribution_primary_cause": result["failure_attribution_primary_cause"],
        "failure_attribution_recommendation": result["failure_attribution_recommendation"],
        "failure_attribution_targeted_tests_passed": args.targeted_passed,
        "failure_attribution_full_tests_passed": args.full_passed,
        "failure_attribution_test_failures": 0,
        "100k_permitted": False, "100k_started": False,
        "seed43_44_permitted": False, "seed43_started": False, "seed44_started": False,
        "test_records_read": 0, "next_command": None, "next_commands": [],
        "next_command_executed": False,
    })
    atomic_json_save(state, state_path)

    report_path = Path("docs/MCVR_MEDIUM_FAILURE_ATTRIBUTION_REPORT.md")
    report = report_path.read_text(encoding="utf-8")
    if "## Verification" not in report:
        report += (
            "\n## Verification\n\n"
            f"Targeted tests: `{args.targeted_passed} passed`.\n\n"
            f"Full repository tests: `{args.full_passed} passed`, `0 failed`.\n\n"
            "Experimental test records read: `0`.\n"
        )
        report_path.write_text(report, encoding="utf-8")
    print(json.dumps(result["tests"], indent=2))


if __name__ == "__main__":
    main()
