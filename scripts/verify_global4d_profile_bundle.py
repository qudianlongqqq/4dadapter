#!/usr/bin/env python
"""Cross-platform integrity and loadability verification for profile bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from etflow.commons.global4d_performance import compact_json
from etflow.commons.global4d_profile_bundle import verify_bundle_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle_dir", required=True, type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/global4d_profile_bundle_verification.json"),
    )
    args = parser.parse_args()
    result = verify_bundle_directory(args.bundle_dir, verify_model=True)
    compact_json(result, args.report)
    print(result["status"])
    if result["status"] != "VALID":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
