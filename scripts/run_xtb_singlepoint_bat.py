#!/usr/bin/env python3
"""Reuse the audited resumable LSGO xTB runner on BAT's frozen coordinates."""

from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()
from scripts import run_xtb_singlepoint_lsgo as implementation


def main() -> int:
    implementation.OUT = Path(__file__).resolve().parents[1] / "reports/ecir_mvr/bat_refinement"
    result = implementation.main()
    if result == 0:
        print("BAT_XTB_SINGLE_POINT_COMPLETED")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
