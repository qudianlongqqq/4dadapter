#!/usr/bin/env python3
"""Run local STEP 2D extraction and identity freeze as one supervised job."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType


def import_file(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--full-extract", required=True, type=Path)
    parser.add_argument("--identity-extract", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--base-historical-union", required=True, type=Path)
    parser.add_argument("--base-historical-provenance", required=True, type=Path)
    parser.add_argument("--historical-completeness-audit", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scripts = args.repo_root.resolve() / "scripts"
    extractor = import_file(
        scripts / "extract_sixs_step2d_full_archive.py", "sixs_step2d_extractor"
    )
    extraction = extractor.extract(
        args.archive,
        args.full_extract,
        args.identity_extract,
        expected_bytes=extractor.EXPECTED_ARCHIVE_BYTES,
        expected_sha256=extractor.EXPECTED_ARCHIVE_SHA256,
    )
    builder = import_file(
        scripts / "build_sixs_step2d_primary_final_cohort.py",
        "sixs_step2d_primary_builder",
    )
    build_args = argparse.Namespace(
        standardized_dir=args.full_extract / "DRUGS" / "standardized_pickles",
        split=args.full_extract / "DRUGS" / "split.npy",
        raw_member_map=args.full_extract / extractor.MEMBER_MAP_NAME,
        extraction_status=args.full_extract / extractor.STATUS_NAME,
        base_historical_union=args.base_historical_union,
        base_historical_provenance=args.base_historical_provenance,
        historical_completeness_audit=args.historical_completeness_audit,
        source_builder=scripts / "build_sixs_step2d_source_universe.py",
        selector=scripts / "freeze_sixs_prospective_final_cohort.py",
        work_root=args.work_root,
        report_dir=args.report_dir,
        reuse_source_index=False,
    )
    status = builder.build(build_args)
    print(
        json.dumps(
            {
                "FULL_EXTRACTION_STATUS": extraction["status"],
                "STEP_2D_STATUS": status["STEP_2D_STATUS"],
                "PRIMARY_FINAL_MEMBERSHIP_FROZEN": status[
                    "PRIMARY_FINAL_MEMBERSHIP_FROZEN"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
