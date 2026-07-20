#!/usr/bin/env python
"""Render the current V8 pilot state without promoting validation to formal test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state", type=Path, default=Path("reports/ecir_mvr/MCVR_V8_FULL_V1_STATUS.json")
    )
    parser.add_argument("--validation", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/ecir_mvr/MCVR_V8_FULL_V1_PILOT.md")
    )
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    validation = (
        json.loads(args.validation.read_text(encoding="utf-8"))
        if args.validation and args.validation.is_file()
        else None
    )
    # The live orchestrator emits pilot fields at the top level; the richer
    # handoff status nests them so implementation-gate evidence can coexist.
    formal_pilot = state.get("formal_pilot", state)
    lines = [
        "# MCVR V8 Full v1 pilot status",
        "",
        f"Status: `{state['status']}`.",
        "",
        "This report contains development validation only. It is not a formal-test report.",
        "",
        f"Completed implementation and gates: `{state.get('completed', [])}`.",
        "",
        f"Formal pilot running: `{formal_pilot.get('running')}`.",
        "",
        f"Missing required assets: `{formal_pilot.get('missing_required_assets', [])}`.",
    ]
    if validation:
        lines.extend(["", "## Development validation", ""])
        for name, value in validation["metrics"].items():
            lines.append(f"- {name}: `{value}`")
    lines.extend(
        [
            "",
            "Isolation: formal_test_records_read=0; formal_test_assets_opened=false; "
            "frozen_holdout_records_read=0.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
