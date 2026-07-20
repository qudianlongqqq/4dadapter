#!/usr/bin/env python
"""Build a train-only V8 rare-error stratification manifest."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from etflow.ecir.v8_sampler import build_stratified_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_stratified_payload(args.train_sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "identity_sha256": payload["identity_sha256"],
                "cohort_counts": payload["cohort_counts"],
                "formal_test_records_read": 0,
                "formal_test_assets_opened": False,
                "frozen_holdout_records_read": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
