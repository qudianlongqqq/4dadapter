#!/usr/bin/env python
"""Build a train-only V8 rare-error stratification manifest."""

# ruff: noqa: E402

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

from etflow.ecir.v8_sampler import build_stratified_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sources", type=Path, required=True)
    parser.add_argument("--train-targets", type=Path)
    parser.add_argument("--target-cache-root", type=Path)
    parser.add_argument("--molecule-exposure-cap", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_stratified_payload(
        args.train_sources,
        target_manifest=args.train_targets,
        target_cache_root=args.target_cache_root,
        molecule_exposure_cap=args.molecule_exposure_cap,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    file_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "identity_sha256": payload["identity_sha256"],
                "file_sha256": file_sha256,
                "cohort_counts": payload["cohort_counts"],
                "overlap_counts": payload["overlap_counts"],
                "molecule_exposure_cap": payload["molecule_exposure_cap"],
                "records_scanned": payload["records_scanned"],
                "validation_records_read": 0,
                "formal_test_records_read": 0,
                "formal_test_assets_opened": False,
                "minimal_validity_target_test_used": False,
                "frozen_holdout_records_read": 0,
                "parameter_selection_from_formal_test": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
