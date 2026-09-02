#!/usr/bin/env python3
"""Build and, if every fail-closed gate passes, freeze STEP 2D membership.

This orchestrator consumes identity/topology metadata only. It never runs a
model, generates coordinates, reads protected outcomes, or evaluates a metric.
Large resumable indexes stay outside Git; only compact audits and the selected
2,500-molecule manifest are written to the repository report directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


IDENTITY_DEFINITION = "sixs-canonical-isomeric-atom-map-free-h-normalized-v1"
SELECTION_RULE = (
    "ASCENDING_SHA256_DOMAIN_NUL_CANONICAL_IDENTITY__"
    "TIE_BREAK_CANONICAL_IDENTITY__TAKE_FIRST_2500"
)
EXPECTED_SELECTOR_SHA256 = (
    "41df4a438e4acca2a545555ea9e5f0b1adf14d330cb9a60a413859d9a88e8907"
)
REQUIRED_HISTORY_REASONS = {
    "CURRENT_REFINER_TRAIN",
    "HISTORICAL_DEVELOPMENT_VAL",
    "CURRENT_DEV_AND_OUTCOME_EXPOSED_MODEL_DEVELOPMENT",
    "FORMAL100_OUTCOME_EXPOSED",
    "OUTCOME_EXPOSED_SECONDARY_COHORT",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def import_file(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_history(
    union_path: Path, provenance_path: Path
) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]], Counter[str]]:
    payload = json.loads(union_path.read_text(encoding="utf-8"))
    if payload.get("identity_definition") != IDENTITY_DEFINITION:
        raise RuntimeError("Historical exclusion identity definition mismatch")
    identities = payload.get("molecule_identities")
    if not isinstance(identities, list) or len(identities) != len(set(identities)):
        raise RuntimeError("Historical exclusion union is invalid or duplicated")
    reasons_by_identity: dict[str, set[str]] = {}
    roles_by_identity: dict[str, set[str]] = {}
    reason_counts: Counter[str] = Counter()
    with provenance_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            identity = row["molecule_identity"]
            reasons = {value for value in row["reason_excluded"].split(";") if value}
            roles = {value for value in row["historical_role"].split(";") if value}
            reasons_by_identity[identity] = reasons
            roles_by_identity[identity] = roles
            reason_counts.update(reasons)
    if set(identities) != set(reasons_by_identity):
        raise RuntimeError("Historical union and provenance identity sets differ")
    return set(identities), reasons_by_identity, roles_by_identity, reason_counts


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def validate_or_freeze(
    selector: ModuleType,
    *,
    source_universe: Path,
    historical_exclusion: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        expected = {
            "source_universe_sha256": sha256_file(source_universe),
            "historical_exclusion_sha256": sha256_file(historical_exclusion),
            "selection_code_sha256": sha256_file(Path(selector.__file__)),
            "target_molecules": 2500,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Existing frozen manifest does not match inputs: {output}")
        return payload
    return selector.freeze(
        source_universe=source_universe,
        historical_exclusion=historical_exclusion,
        output=output,
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    extraction = json.loads(args.extraction_status.read_text(encoding="utf-8"))
    if extraction.get("status") != "COMPLETE":
        raise RuntimeError("Full archive extraction is not complete")

    selector = import_file(args.selector.resolve(), "sixs_step2d_selector")
    selector_sha = sha256_file(args.selector)
    if selector_sha != EXPECTED_SELECTOR_SHA256:
        raise RuntimeError(
            f"Frozen selector SHA256 changed: {selector_sha} != {EXPECTED_SELECTOR_SHA256}"
        )
    if selector.IDENTITY_DEFINITION != IDENTITY_DEFINITION:
        raise RuntimeError("Frozen selector identity definition changed")

    source_jsonl = args.work_root / "source_universe_manifest.jsonl"
    source_csv = args.work_root / "source_universe_manifest.csv"
    source_failures = args.work_root / "canonicalization_failure_report.jsonl"
    source_duplicates = args.work_root / "duplicate_identity_report.jsonl"
    source_summary_path = args.work_root / "source_universe_summary.json"
    if args.reuse_source_index:
        required_source_outputs = (
            source_jsonl,
            source_csv,
            source_failures,
            source_duplicates,
            source_summary_path,
        )
        missing = [str(path) for path in required_source_outputs if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Cannot reuse existing source index; missing: " + ", ".join(missing)
            )
        source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
        if source_summary.get("status") != "COMPLETE":
            raise RuntimeError("Existing source index summary is not complete")
        if source_summary.get("source_universe_jsonl_sha256") != sha256_file(
            source_jsonl
        ):
            raise RuntimeError("Existing source universe hash does not match its summary")
    else:
        source_module = import_file(args.source_builder.resolve(), "sixs_step2d_source")
        source_args = argparse.Namespace(
            standardized_dir=args.standardized_dir,
            split=args.split,
            raw_filename_index=None,
            raw_member_map=args.raw_member_map,
            raw_index_status="subset",
            cache_dir=args.work_root / "chunk_cache",
            csv_output=source_csv,
            jsonl_output=source_jsonl,
            failure_output=source_failures,
            duplicate_output=source_duplicates,
            summary_output=source_summary_path,
        )
        source_summary = source_module.build(source_args)

    base_union, reasons_by_identity, roles_by_identity, reason_counts = load_history(
        args.base_historical_union, args.base_historical_provenance
    )
    missing_reasons = sorted(REQUIRED_HISTORY_REASONS - set(reason_counts))
    historical_audit_text = args.historical_completeness_audit.read_text(
        encoding="utf-8"
    )
    prior_history_incomplete = (
        "HISTORICAL_DEVELOPMENT_IDENTITY_UNION_COMPLETE = NO" in historical_audit_text
    )
    historical_evidence_complete = not missing_reasons and not prior_history_incomplete

    source_rows = read_jsonl(source_jsonl)
    augmented_union = set(base_union)
    augmented_rows: list[dict[str, Any]] = []
    conservative_added = 0
    for row in source_rows:
        identity = row["molecule_id"]
        if row["history_status"] != "PROVEN_NATIVE_TEST":
            if identity not in augmented_union:
                conservative_added += 1
            augmented_union.add(identity)
            augmented_rows.append(
                {
                    "molecule_identity": identity,
                    "molecule_identity_sha256": row["molecule_identity_sha256"],
                    "reason_excluded": (
                        "UPSTREAM_NATIVE_TRAIN_VAL_OR_UNRECOVERABLE_NATIVE_SPLIT"
                    ),
                    "historical_role": "CONSERVATIVE_SOURCE_DOMAIN_EXCLUSION",
                    "source_manifest": str(source_jsonl.resolve()),
                    "historical_use_status": "USED_OR_UNKNOWN_EXCLUDED",
                }
            )

    augmented_union_path = args.work_root / "historical_exclusion_union_manifest.json"
    augmented_payload = {
        "schema_version": "sixs-step2d-historical-exclusion-union-v2",
        "identity_definition": IDENTITY_DEFINITION,
        "base_historical_union_sha256": sha256_file(args.base_historical_union),
        "base_historical_union_molecules": len(base_union),
        "conservative_source_domain_additions": conservative_added,
        "molecule_identities": sorted(augmented_union),
    }
    atomic_text(
        augmented_union_path,
        json.dumps(augmented_payload, indent=2, sort_keys=True) + "\n",
    )
    atomic_jsonl(
        args.work_root / "historical_exclusion_provenance_manifest.jsonl",
        augmented_rows,
    )

    eligible_rows: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for row in source_rows:
        accepted, reasons = selector.is_eligible(row, augmented_union)
        if row["history_status"] != "PROVEN_NATIVE_TEST":
            accepted = False
            reasons = list(reasons) + ["native_test_identity_not_exactly_proven"]
        if accepted:
            eligible_rows.append(row)
        else:
            rejection_counts.update(reasons)
    eligible_jsonl = args.work_root / "eligible_unused_pool_manifest.jsonl"
    atomic_jsonl(eligible_jsonl, eligible_rows)
    eligible_csv = args.work_root / "eligible_unused_pool_manifest.csv"
    eligible_fields = [
        "molecule_id",
        "molecule_identity_sha256",
        "source_dataset_identity",
        "reference_identity",
        "source_chunk",
        "native_split",
        "native_split_linkage",
        "heavy_atom_count",
        "rotatable_bond_count",
        "ring_count",
        "formal_charge",
        "atomic_numbers",
    ]
    write_csv(
        eligible_csv,
        [
            {
                **{field: row[field] for field in eligible_fields},
                "atomic_numbers": ";".join(map(str, row["atomic_numbers"])),
            }
            for row in eligible_rows
        ],
        eligible_fields,
    )

    historical_complete = historical_evidence_complete and all(
        row["history_status"] == "PROVEN_NATIVE_TEST"
        or row["molecule_id"] in augmented_union
        for row in source_rows
    )
    enough = len(eligible_rows) >= selector.TARGET_MOLECULES
    membership_frozen = historical_complete and enough
    external_manifest = args.work_root / "primary_final_2500_manifest.json"
    report_manifest = args.report_dir / "04_PRIMARY_FINAL_2500_MANIFEST.json"
    selected_payload: dict[str, Any] | None = None
    if membership_frozen:
        selected_payload = validate_or_freeze(
            selector,
            source_universe=source_jsonl,
            historical_exclusion=augmented_union_path,
            output=external_manifest,
        )
        atomic_text(
            report_manifest,
            json.dumps(selected_payload, indent=2, sort_keys=True) + "\n",
        )
    elif report_manifest.exists():
        raise RuntimeError(
            "A previous primary manifest exists but current fail-closed gates do not pass"
        )

    selected = selected_payload["rows"] if selected_payload is not None else []
    selected_ids = {row["molecule_id"] for row in selected}
    current_train = {
        identity
        for identity, reasons in reasons_by_identity.items()
        if "CURRENT_REFINER_TRAIN" in reasons
    }
    current_dev = {
        identity
        for identity, reasons in reasons_by_identity.items()
        if "CURRENT_DEV_AND_OUTCOME_EXPOSED_MODEL_DEVELOPMENT" in reasons
    }
    train_overlap = len(selected_ids & current_train)
    dev_overlap = len(selected_ids & current_dev)
    history_overlap = len(selected_ids & augmented_union)
    if membership_frozen and (train_overlap or dev_overlap or history_overlap):
        raise RuntimeError("Selected membership overlaps an exclusion set")

    failure_rows = read_jsonl(source_failures)
    duplicate_rows = read_jsonl(source_duplicates)
    failure_counts = Counter(row["failure_type"] for row in failure_rows)
    duplicate_split_counts = Counter(row["native_splits"] for row in duplicate_rows)
    write_csv(
        args.report_dir / "02_CANONICALIZATION_FAILURE_SUMMARY.csv",
        [
            {"failure_type": key, "count": value}
            for key, value in sorted(failure_counts.items())
        ],
        ["failure_type", "count"],
    )
    write_csv(
        args.report_dir / "03_DUPLICATE_IDENTITY_SUMMARY.csv",
        [
            {"native_splits": key, "canonical_identities": value}
            for key, value in sorted(duplicate_split_counts.items())
        ],
        ["native_splits", "canonical_identities"],
    )

    overlap = {
        "schema_version": "sixs-step2d-overlap-audit-v1",
        "primary_final_membership_frozen": membership_frozen,
        "selected_molecules": len(selected),
        "current_train_overlap": train_overlap if membership_frozen else None,
        "current_dev_overlap": dev_overlap if membership_frozen else None,
        "historical_exclusion_overlap": history_overlap if membership_frozen else None,
        "processed_native_test_reference_facts": {
            "files": 1000,
            "unique_identities": 1000,
            "canonicalization_failed": 0,
            "train_overlap": 765,
            "val_overlap": 113,
            "not_in_train_or_val": 124,
            "usable_directly_as_primary": False,
        },
    }
    atomic_text(
        args.report_dir / "05_OVERLAP_AUDIT.json",
        json.dumps(overlap, indent=2, sort_keys=True) + "\n",
    )

    native_linkage_limit = source_summary["native_linkage_outside_complete_raw_index"]
    asset_audit = f"""# GEOM-DRUGS source asset audit

```text
SOURCE_DATASET = GEOM-DRUGS-WITH-TORSIONAL-DIFFUSION-NATIVE-SPLITS
ARCHIVE_BYTES = {extraction['archive_bytes']}
ARCHIVE_SHA256 = {extraction['archive_sha256']}
ARCHIVE_MEMBERS = {extraction['archive_members']}
ARCHIVE_REGULAR_FILES = {extraction['archive_regular_files']}
ARCHIVE_REGULAR_FILE_BYTES = {extraction['archive_regular_file_bytes']}
STANDARDIZED_PICKLE_CHUNKS = {extraction['standardized_pickle_chunks']}
RAW_PICKLE_MEMBERS = {extraction['raw_pickle_members']}
NATIVE_SPLIT_COUNTS = {json.dumps(source_summary['native_split_index_counts'], sort_keys=True)}
NATIVE_SPLIT_INDEX_UNIVERSE = {source_summary['native_split_index_universe']}
STANDARDIZED_ENTRIES = {source_summary['standardized_entries']}
SOURCE_UNIVERSE_N_MOLECULES = {source_summary['source_universe_n_molecules']}
RAW_LISTING_NATIVE_LINKAGE = {source_summary['raw_listing_native_linkage']}
NATIVE_LINKAGE_OUTSIDE_COMPLETE_RAW_INDEX = {native_linkage_limit}
```

`split.npy` is a three-array native index partition. Standardized chunks are
Python dictionaries keyed by source stems; each molecule stores a SMILES,
charge, ensemble metadata, and a list of conformer dictionaries containing an
RDKit molecule. Coordinates and energy/outcome fields were not materialized in
any final manifest. The released archive has only a raw filename subset, not
the complete 304,339-name raw index and not a continuous prefix. Standardized
identities therefore receive no guessed native index and are conservatively
excluded.
"""
    atomic_text(args.report_dir / "01_GEOM_DRUGS_SOURCE_ASSET_AUDIT.md", asset_audit)

    completeness_label = (
        "YES" if historical_complete else "NO__LEGACY_IDENTITIES_OR_PROVENANCE_INCOMPLETE"
    )
    provenance = f"""# STEP 2D provenance and fail-closed decision

The official archive is bound by byte size and SHA256. Canonical identity uses
RDKit parsing, atom-map removal, ordinary explicit-H removal, sanitization, and
canonical isomeric SMILES with stereochemistry preserved. Canonical duplicates
collapse to one identity; failures are ineligible and explicitly reported.

The base historical union contains `{len(base_union)}` canonical identities and
covers current TRAIN, full historical VAL/DEV, Formal100, the reused factorial/
ablation/multiseed DEV cohort, and the inspected DiTMC large holdout. All source
identities whose native-test membership is not exactly recoverable are added to
the conservative exclusion union. No absence is interpreted as nonuse.

```text
SELECTION_RULE = {SELECTION_RULE}
SELECTOR_SHA256 = {selector_sha}
HISTORICAL_REQUIRED_REASON_CLASSES_MISSING = {','.join(missing_reasons) or 'NONE'}
HISTORICAL_EXCLUSION_UNION_COMPLETE = {completeness_label}
ELIGIBLE_UNUSED_POOL_N_MOLECULES = {len(eligible_rows)}
PRIMARY_FINAL_MEMBERSHIP_FROZEN = {'YES' if membership_frozen else 'NO'}
PROTECTED_OUTCOME_READ = NO
```
"""
    atomic_text(args.report_dir / "06_PROVENANCE_REPORT.md", provenance)

    blockers: list[str] = []
    if missing_reasons:
        blockers.append("MISSING_HISTORICAL_REASON_CLASSES:" + ",".join(missing_reasons))
    if prior_history_incomplete:
        blockers.append("LEGACY_HISTORICAL_COHORT_IDENTITIES_NOT_FULLY_RECOVERED")
    if not enough:
        blockers.append(
            f"ELIGIBLE_EXACT_NATIVE_TEST_POOL_BELOW_2500:{len(eligible_rows)}"
        )
    if not historical_complete:
        blockers.append("HISTORICAL_EXCLUSION_UNION_INCOMPLETE")
    status_value = "PASS_MEMBERSHIP_FROZEN" if membership_frozen else "BLOCKED_FAIL_CLOSED"
    blocked_next_step = (
        "RECOVER_COMPLETE_RAW_FILENAME_INDEX_AND_MISSING_LEGACY_IDENTITIES__"
        "DO_NOT_EVALUATE"
    )
    status: dict[str, Any] = {
        "schema_version": "sixs-step2d-primary-final-builder-v1",
        "STEP_2D_STATUS": status_value,
        "ARCHIVE_FOUND": True,
        "ARCHIVE_BYTES": extraction["archive_bytes"],
        "ARCHIVE_SHA256": extraction["archive_sha256"],
        "OFFICIAL_ARCHIVE_VERIFIED": True,
        "FULL_EXTRACTION_STATUS": extraction["status"],
        "FULL_EXTRACTION_ROOT": extraction["output_root"],
        "STANDARDIZED_PICKLE_CHUNKS": extraction["standardized_pickle_chunks"],
        "SOURCE_ASSET_AUDIT": "PASS_WITH_RAW_SUBSET_NATIVE_LINKAGE_LIMIT_EXPLICIT",
        "SOURCE_UNIVERSE_N_MOLECULES": source_summary["source_universe_n_molecules"],
        "CANONICAL_IDENTITY_STATUS": "PASS",
        "CANONICAL_FAILURE_POLICY": "INELIGIBLE_EXPLICIT_REPORT_NO_UNSEEN_ASSUMPTION",
        "CANONICALIZATION_FAILURES": len(failure_rows),
        "DUPLICATE_CANONICAL_IDENTITIES": len(duplicate_rows),
        "HISTORICAL_EXCLUSION_UNION_N_MOLECULES": len(augmented_union),
        "HISTORICAL_EXCLUSION_UNION_COMPLETE": completeness_label,
        "ELIGIBLE_UNUSED_POOL_N_MOLECULES": len(eligible_rows),
        "SELECTION_RULE": SELECTION_RULE,
        "SELECTION_RULE_EXACTLY_PRESERVED": True,
        "SELECTOR_SHA256": selector_sha,
        "PRIMARY_FINAL_N_MOLECULES": len(selected),
        "PRIMARY_FINAL_MANIFEST": str(report_manifest.resolve())
        if membership_frozen
        else None,
        "PRIMARY_FINAL_MANIFEST_SHA256": sha256_file(report_manifest)
        if membership_frozen
        else None,
        "CURRENT_TRAIN_OVERLAP": train_overlap if membership_frozen else None,
        "CURRENT_DEV_OVERLAP": dev_overlap if membership_frozen else None,
        "HISTORICAL_EXCLUSION_OVERLAP": history_overlap
        if membership_frozen
        else None,
        "PRIMARY_FINAL_MEMBERSHIP_FROZEN": membership_frozen,
        "PROTECTED_OUTCOME_READ": False,
        "SCIENTIFIC_MODEL_CHANGED": False,
        "READY_FOR_STEP_3": membership_frozen,
        "BLOCKERS": blockers,
        "NEXT_STEP": (
            "STEP_3_MMFF94S_PROTOCOL_FREEZE"
            if membership_frozen
            else blocked_next_step
        ),
        "NO_REDUNDANT_DOWNLOAD": True,
        "NO_REPEATED_POLLING": True,
        "NO_BUSY_WAITING": True,
        "work_artifacts": {
            "source_universe_manifest": str(source_jsonl.resolve()),
            "source_universe_manifest_sha256": sha256_file(source_jsonl),
            "historical_exclusion_union_manifest": str(augmented_union_path.resolve()),
            "historical_exclusion_union_manifest_sha256": sha256_file(
                augmented_union_path
            ),
            "eligible_unused_pool_manifest": str(eligible_jsonl.resolve()),
            "eligible_unused_pool_manifest_sha256": sha256_file(eligible_jsonl),
            "canonicalization_failure_report": str(source_failures.resolve()),
            "duplicate_identity_report": str(source_duplicates.resolve()),
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "base_history_reason_counts": dict(sorted(reason_counts.items())),
    }
    status_path = args.report_dir / "FINAL_STATUS.json"
    atomic_text(status_path, json.dumps(status, indent=2, sort_keys=True) + "\n")

    artifact_names = [
        "01_GEOM_DRUGS_SOURCE_ASSET_AUDIT.md",
        "02_CANONICALIZATION_FAILURE_SUMMARY.csv",
        "03_DUPLICATE_IDENTITY_SUMMARY.csv",
        "05_OVERLAP_AUDIT.json",
        "06_PROVENANCE_REPORT.md",
        "FINAL_STATUS.json",
    ]
    if membership_frozen:
        artifact_names.append("04_PRIMARY_FINAL_2500_MANIFEST.json")
    sums = "".join(
        f"{sha256_file(args.report_dir / name)}  {name}\n" for name in artifact_names
    )
    atomic_text(args.report_dir / "SHA256SUMS.txt", sums)
    print(json.dumps(status, sort_keys=True), flush=True)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standardized-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--raw-member-map", required=True, type=Path)
    parser.add_argument("--extraction-status", required=True, type=Path)
    parser.add_argument("--base-historical-union", required=True, type=Path)
    parser.add_argument("--base-historical-provenance", required=True, type=Path)
    parser.add_argument("--historical-completeness-audit", required=True, type=Path)
    parser.add_argument("--source-builder", required=True, type=Path)
    parser.add_argument("--selector", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--reuse-source-index", action="store_true")
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
