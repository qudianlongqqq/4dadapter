#!/usr/bin/env python3
"""Finalize STEP 2D after source and historical identity assets are complete."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from typing import Any


IDENTITY_DEFINITION = "sixs-canonical-isomeric-atom-map-free-h-normalized-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def add_field(existing: str, value: str) -> str:
    values = {item for item in existing.split(";") if item}
    values.add(value)
    return ";".join(sorted(values))


def load_freezer(path: Path):
    spec = importlib.util.spec_from_file_location("sixs_frozen_selector", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen selector {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--source-summary", required=True, type=Path)
    parser.add_argument("--historical-json", required=True, type=Path)
    parser.add_argument("--historical-csv", required=True, type=Path)
    parser.add_argument("--selector", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--test-mols", required=True, type=Path)
    args = parser.parse_args()

    report = args.report.resolve()
    source_rows = [
        json.loads(line)
        for line in args.source_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_summary = json.loads(args.source_summary.read_text(encoding="utf-8"))
    extraction = json.loads(
        (args.asset_root / "IDENTITY_EXTRACTION_STATUS.json").read_text(
            encoding="utf-8"
        )
    )
    if source_summary.get("status") != "COMPLETE" or extraction.get("status") != "COMPLETE":
        raise RuntimeError("Source identity assets are not complete")

    union_payload = json.loads(args.historical_json.read_text(encoding="utf-8"))
    union = set(union_payload["molecule_identities"])
    provenance: dict[str, dict[str, str]] = {}
    with args.historical_csv.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            provenance[row["molecule_identity"]] = row

    for row in source_rows:
        identity = row["molecule_id"]
        if row["history_status"] == "PROVEN_UNUSED_NATIVE_TEST":
            continue
        union.add(identity)
        current = provenance.setdefault(
            identity,
            {
                "molecule_identity": identity,
                "molecule_identity_sha256": row["molecule_identity_sha256"],
                "reason_excluded": "",
                "source_manifest": "",
                "historical_role": "",
                "historical_use_status": "",
            },
        )
        unknown = "unassigned" in row["native_split"].split("|")
        current["reason_excluded"] = add_field(
            current["reason_excluded"],
            "SOURCE_HISTORY_UNKNOWN"
            if unknown
            else "UPSTREAM_NATIVE_TRAIN_OR_VALIDATION_SPLIT",
        )
        current["source_manifest"] = add_field(
            current["source_manifest"], str(args.split.resolve())
        )
        current["historical_role"] = add_field(
            current["historical_role"],
            "CONSERVATIVE_UNKNOWN_HISTORY_EXCLUSION"
            if unknown
            else "ETFLOW_UPSTREAM_TRAIN_OR_DEVELOPMENT_EXCLUSION",
        )
        current["historical_use_status"] = (
            "UNKNOWN_EXCLUDED" if unknown else "PROVEN_USED"
        )

    union_payload = {
        "schema_version": "sixs-step2d-historical-exclusion-union-v1",
        "identity_definition": IDENTITY_DEFINITION,
        "molecule_identities": sorted(union),
    }
    atomic_text(
        args.historical_json,
        json.dumps(union_payload, indent=2, sort_keys=True) + "\n",
    )
    fields = [
        "molecule_identity",
        "molecule_identity_sha256",
        "reason_excluded",
        "source_manifest",
        "historical_role",
        "historical_use_status",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for identity in sorted(provenance):
        writer.writerow(provenance[identity])
    atomic_text(args.historical_csv, buffer.getvalue())

    freezer = load_freezer(args.selector.resolve())
    eligible = []
    for row in source_rows:
        accepted, _ = freezer.is_eligible(row, union)
        if accepted:
            eligible.append(row)
    if len(eligible) < 2_500:
        raise RuntimeError(f"Eligible unused pool is only {len(eligible)}")
    if any(row["native_split"] != "test" for row in eligible):
        raise RuntimeError("Eligible pool contains a non-native-test molecule")

    manifest_json = report / "PRIMARY_FINAL_MOLECULE_MANIFEST.json"
    payload = freezer.freeze(
        source_universe=args.source_jsonl.resolve(),
        historical_exclusion=args.historical_json.resolve(),
        output=manifest_json,
    )
    source_by_id = {row["molecule_id"]: row for row in source_rows}
    manifest_csv = report / "16_PRIMARY_MOLECULE_MANIFEST.csv"
    manifest_fields = [
        "molecule_identity",
        "molecule_identity_sha256",
        "selection_rank",
        "native_split",
        "historical_exclusion_overlap",
        "reference_identity",
        "source_dataset_identity",
        "heavy_atom_count",
        "rotatable_bond_count",
        "ring_count",
        "formal_charge",
        "atomic_numbers",
    ]
    selected_buffer = io.StringIO(newline="")
    selected_writer = csv.DictWriter(
        selected_buffer, fieldnames=manifest_fields, lineterminator="\n"
    )
    selected_writer.writeheader()
    for selected in payload["rows"]:
        source = source_by_id[selected["molecule_id"]]
        selected_writer.writerow(
            {
                "molecule_identity": selected["molecule_id"],
                "molecule_identity_sha256": selected["molecule_identity_sha256"],
                "selection_rank": selected["selection_rank"],
                "native_split": source["native_split"],
                "historical_exclusion_overlap": "NO",
                "reference_identity": selected["reference_identity"],
                "source_dataset_identity": selected["source_dataset_identity"],
                "heavy_atom_count": selected["heavy_atom_count"],
                "rotatable_bond_count": selected["rotatable_bond_count"],
                "ring_count": selected["ring_count"],
                "formal_charge": selected["formal_charge"],
                "atomic_numbers": ";".join(map(str, selected["atomic_numbers"])),
            }
        )
    atomic_text(manifest_csv, selected_buffer.getvalue())

    selected_ids = {row["molecule_id"] for row in payload["rows"]}
    train_ids = {
        identity
        for identity, row in provenance.items()
        if "CURRENT_REFINER_TRAIN" in row["reason_excluded"]
    }
    dev_ids = {
        identity
        for identity, row in provenance.items()
        if "CURRENT_DEV_AND_OUTCOME_EXPOSED_MODEL_DEVELOPMENT"
        in row["reason_excluded"]
    }
    train_overlap = len(selected_ids & train_ids)
    dev_overlap = len(selected_ids & dev_ids)
    historical_overlap = len(selected_ids & union)
    if train_overlap or dev_overlap or historical_overlap:
        raise RuntimeError("Frozen primary membership overlaps historical identities")

    archive_sha = sha256_file(args.archive)
    split_sha = sha256_file(args.split)
    test_mols_sha = sha256_file(args.test_mols)
    union_sha = sha256_file(args.historical_json)
    source_csv_sha = sha256_file(args.source_csv)
    source_jsonl_sha = sha256_file(args.source_jsonl)
    manifest_sha = sha256_file(manifest_csv)

    source_doc = f"""# STEP 2D source asset provenance

```text
GEOM_SOURCE_ORIGIN = https://drive.google.com/drive/folders/1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7 (official torsional-diffusion release linked by repository README; standardized GEOM-DRUGS derivative)
GEOM_DRIVE_FILE_ID = 1j_Pj6XZjJLf1tVQzjGWo2EFct2Nj_BvR
GEOM_PRIMARY_PUBLICATION_ORIGIN = Harvard Dataverse doi:10.7910/DVN/JNGTDF datafile 4327252 (raw rdkit_folder.tar.gz; 50137579520 bytes; published MD5 e8f2168b7050652db22c976be25c450e; not duplicated locally)
GEOM_ASSET_FILENAME = drugs.tar.gz
GEOM_ASSET_SIZE = {args.archive.stat().st_size}
GEOM_ASSET_SHA256 = {archive_sha}
TORSIONAL_DIFFUSION_REPOSITORY_COMMIT = 5f713b42d7000307655f272471014c6127ea59be
SPLIT_SOURCE_ORIGIN = official torsional-diffusion drugs.tar.gz
SPLIT_FILES = DRUGS/split.npy; DRUGS/test_mols.pkl; DRUGS/test_smiles.csv; DRUGS/standardized_pickles/*.pickle
SPLIT_FILE_SHA256 = {split_sha}
TEST_MOLS_SHA256 = {test_mols_sha}
VERSION_PROVENANCE = PARTIAL__OFFICIAL_RELEASE_AND_CONTENT_HASH_BOUND__UPSTREAM_DRIVE_DID_NOT_PUBLISH_SHA256
SOURCE_UNIVERSE_N_MOLECULES = {len(source_rows)}
SOURCE_UNIVERSE_CSV_SHA256 = {source_csv_sha}
SOURCE_UNIVERSE_JSONL_SHA256 = {source_jsonl_sha}
```

Only molecular identity, native split, reference availability, graph composition,
charge and input-compatibility metadata were materialized. No model or scientific
outcome was evaluated.
"""
    atomic_text(report / "11_SOURCE_ASSET_PROVENANCE.md", source_doc)

    eligible_doc = f"""# Eligible unused pool audit

The complete standardized source universe contains `{len(source_rows)}` unique
canonical molecular identities. Native train/validation identities and every
unknown-split identity were conservatively placed in the historical exclusion
union. The candidate pool therefore consists only of official native-test
identities with all frozen input-only compatibility gates passing.

```text
HISTORICAL_EXCLUSION_UNION_N_MOLECULES = {len(union)}
HISTORICAL_EXCLUSION_UNION_SHA256 = {union_sha}
HISTORICAL_EXCLUSION_UNION_COMPLETE = YES
ELIGIBLE_UNUSED_POOL_N_MOLECULES = {len(eligible)}
ELIGIBLE_UNUSED_POOL_NATIVE_SPLIT = test
UNKNOWN_HISTORY_ALLOWED_IN_POOL = NO
ELIGIBILITY_RULE_CHANGED = NO
```
"""
    atomic_text(report / "14_ELIGIBLE_UNUSED_POOL_AUDIT.md", eligible_doc)

    protocol_doc = """# ETFlow NFE10 versus NFE50 protocol decision

The historical materialized source tables label generation as NFE10, but their
source cache records do not bind NFE, config hash, runner hash, ETFlow commit or
environment. The historical WSL runner/config environment is absent. The current
official ETFlow checkout and `drugs-o3.yaml` consistently expose NFE50, and the
checkpoint itself contains no recoverable inference hyperparameters. Therefore
historical NFE10 is not exactly reproducible.

The outcome-blind prospective decision is frozen before any coordinate generation:

```text
HISTORICAL_NFE10_EXACTLY_RECOVERABLE = NO
HISTORICAL_NFE10_CODE = NOT_IMMUTABLY_BOUND
HISTORICAL_NFE10_CHECKPOINT = PATH_AND_CURRENT_COPY_HASH_AVAILABLE__HISTORICAL_COPY_IDENTITY_NOT_FULLY_BOUND
HISTORICAL_NFE10_CONFIG = NOT_RECOVERED
HISTORICAL_NFE10_RUNNER = NOT_RECOVERED
HISTORICAL_NFE10_ENVIRONMENT = NOT_RECOVERED
PROSPECTIVE_ETFLOW_PROTOCOL = FROZEN_NFE50_COMPATIBLE_BUT_NOT_IDENTICAL
PROSPECTIVE_ETFLOW_NFE = 50
ETFLOW_PROTOCOL_MATCH_STATUS = COMPATIBLE_BUT_NOT_IDENTICAL__HISTORICAL_NFE10_VS_FROZEN_PROSPECTIVE_NFE50
ETFLOW_PROTOCOL_FROZEN = YES
FUTURE_OUTCOME_DEPENDENT_PROTOCOL_SWITCH = PROHIBITED
```
"""
    atomic_text(report / "15_ETFLOW_NFE10_NFE50_PROTOCOL_DECISION.md", protocol_doc)

    freeze_doc = f"""# Prospective primary molecule membership freeze

The unchanged selector sorted all eligible unused identities by ascending
`SHA256("sixs-prospective-primary-final-v1" + NUL + canonical_identity)`, used
canonical identity as the tie-break, and froze the first 2,500 molecules.

```text
PRIMARY_FINAL_N_MOLECULES = 2500
PRIMARY_FINAL_N_RECORDS_PLANNED = 5000
PRIMARY_FINAL_MOLECULE_MANIFEST = {manifest_csv.as_posix()}
PRIMARY_FINAL_MOLECULE_MANIFEST_SHA256 = {manifest_sha}
PRIMARY_FINAL_TRAIN_OVERLAP = {train_overlap}
PRIMARY_FINAL_DEV_OVERLAP = {dev_overlap}
PRIMARY_FINAL_HISTORICAL_EXCLUSION_OVERLAP = {historical_overlap}
PRIMARY_FINAL_MOLECULE_MEMBERSHIP_FROZEN = YES
SCIENTIFIC_COORDINATES_GENERATED = NO
PROTECTED_OUTCOME_READ = NO
```
"""
    atomic_text(report / "17_PRIMARY_MOLECULE_FREEZE.md", freeze_doc)

    status: dict[str, Any] = {
        "schema_version": "sixs-step2d-source-universe-and-membership-freeze-v1",
        "STEP_2D_STATUS": "PASS",
        "GEOM_SOURCE_ACQUIRED": "YES__OFFICIAL_STANDARDIZED_GEOM_DRUGS_RELEASE",
        "GEOM_SOURCE_SHA256": archive_sha,
        "TORSIONAL_DIFFUSION_SPLITS_ACQUIRED": "YES",
        "SPLIT_SHA256": split_sha,
        "SOURCE_UNIVERSE_N_MOLECULES": len(source_rows),
        "SOURCE_UNIVERSE_CSV_SHA256": source_csv_sha,
        "HISTORICAL_EXCLUSION_UNION_N_MOLECULES": len(union),
        "HISTORICAL_EXCLUSION_UNION_COMPLETE": "YES",
        "HISTORICAL_EXCLUSION_UNION_SHA256": union_sha,
        "ELIGIBLE_UNUSED_POOL_N_MOLECULES": len(eligible),
        "HISTORICAL_NFE10_EXACTLY_RECOVERABLE": "NO",
        "PROSPECTIVE_ETFLOW_PROTOCOL": "FROZEN_NFE50_COMPATIBLE_BUT_NOT_IDENTICAL",
        "PROSPECTIVE_ETFLOW_NFE": 50,
        "ETFLOW_PROTOCOL_MATCH_STATUS": "COMPATIBLE_BUT_NOT_IDENTICAL__HISTORICAL_NFE10_VS_FROZEN_PROSPECTIVE_NFE50",
        "ETFLOW_PROTOCOL_FROZEN": "YES",
        "PRIMARY_FINAL_N_MOLECULES": 2500,
        "PRIMARY_FINAL_MOLECULE_MANIFEST": str(manifest_csv),
        "PRIMARY_FINAL_MOLECULE_MANIFEST_SHA256": manifest_sha,
        "PRIMARY_FINAL_TRAIN_OVERLAP": train_overlap,
        "PRIMARY_FINAL_DEV_OVERLAP": dev_overlap,
        "PRIMARY_FINAL_HISTORICAL_EXCLUSION_OVERLAP": historical_overlap,
        "PRIMARY_FINAL_MOLECULE_MEMBERSHIP_FROZEN": "YES",
        "PROTECTED_OUTCOME_READ": "NO",
        "READY_TO_GENERATE_FROZEN_ETFLOW_SOURCE_RECORDS": "YES",
        "BLOCKERS": "NONE_FOR_IDENTITY_FREEZE__PROSPECTIVE_SOURCE_GENERATION_NOT_STARTED",
        "NEXT_STEP": "SEPARATELY_AUTHORIZE_AND_RUN_FROZEN_NFE50_ETFLOW_SOURCE_GENERATION_FOR_THE_FROZEN_2500_MOLECULE_MANIFEST",
        "NO_SCIENTIFIC_EVALUATION": "YES",
        "NO_PROTECTED_OUTCOME_READ": "YES",
        "NO_REPEATED_POLLING": "YES",
        "STOP_AFTER_STEP_2D": "YES",
    }
    status["artifact_sha256"] = {
        name: sha256_file(report / name)
        for name in (
            "11_SOURCE_ASSET_PROVENANCE.md",
            "12_SOURCE_UNIVERSE_IDENTITY_INDEX.csv",
            "13_HISTORICAL_EXCLUSION_PROVENANCE.csv",
            "14_ELIGIBLE_UNUSED_POOL_AUDIT.md",
            "15_ETFLOW_NFE10_NFE50_PROTOCOL_DECISION.md",
            "16_PRIMARY_MOLECULE_MANIFEST.csv",
            "17_PRIMARY_MOLECULE_FREEZE.md",
            "HISTORICAL_EXCLUSION_UNION.json",
            "SOURCE_UNIVERSE_IDENTITY_INDEX.jsonl",
            "SOURCE_UNIVERSE_SUMMARY.json",
            "PRIMARY_FINAL_MOLECULE_MANIFEST.json",
        )
    }
    status_path = report / "STEP2D_STATUS.json"
    atomic_text(status_path, json.dumps(status, indent=2, sort_keys=True) + "\n")

    names = [
        "11_SOURCE_ASSET_PROVENANCE.md",
        "12_SOURCE_UNIVERSE_IDENTITY_INDEX.csv",
        "13_HISTORICAL_EXCLUSION_PROVENANCE.csv",
        "14_ELIGIBLE_UNUSED_POOL_AUDIT.md",
        "15_ETFLOW_NFE10_NFE50_PROTOCOL_DECISION.md",
        "16_PRIMARY_MOLECULE_MANIFEST.csv",
        "17_PRIMARY_MOLECULE_FREEZE.md",
        "STEP2D_STATUS.json",
        "HISTORICAL_EXCLUSION_UNION.json",
        "SOURCE_UNIVERSE_IDENTITY_INDEX.jsonl",
        "SOURCE_UNIVERSE_SUMMARY.json",
        "PRIMARY_FINAL_MOLECULE_MANIFEST.json",
    ]
    sums = "".join(f"{sha256_file(report / name)}  {name}\n" for name in names)
    atomic_text(report / "STEP2D_SHA256SUMS.txt", sums)
    print(json.dumps(status, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
