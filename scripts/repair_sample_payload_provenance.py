#!/usr/bin/env python
"""Safely validate or repair metadata on an existing sample payload."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch

from etflow.data.flexbond_eval_manifest import (
    build_manifest_aware_sample_payload,
    limit_manifest_molecules,
    load_eval_manifest,
    validate_dataset_against_manifest,
    validate_sample_payload_provenance,
)
from etflow.data.flexbond_inference_dataset import FlexBondInferenceDataset


def _validate_core_records(
    records: object, selected_manifest: dict, expected_method: str
) -> list[dict]:
    if not isinstance(records, list):
        raise ValueError("Sample payload records must be a list.")
    rows = selected_manifest["records"]
    expected_ids = [str(row["sample_id"]) for row in rows]
    actual_ids = [str(record.get("sample_id")) for record in records]
    if actual_ids != expected_ids:
        raise ValueError("ordered sample IDs differ from the requested sampling subset")
    for record, row in zip(records, rows):
        if str(record.get("method_name")) != expected_method:
            raise ValueError(f"method_name mismatch for {row['sample_id']!r}")
        molecule_id = record.get("source_mol_id", record.get("mol_id"))
        if str(molecule_id) != str(row["mol_id"]):
            raise ValueError(f"molecule ID mismatch for {row['sample_id']!r}")
        if str(record.get("x_init_hash")) != str(row["x_init_hash"]):
            raise ValueError(f"x_init_hash mismatch for {row['sample_id']!r}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--inference_cache", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_molecules", type=int)
    parser.add_argument("--expected_method", required=True)
    args = parser.parse_args()

    payload = torch.load(args.payload, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Sample payload must be a mapping.")
    manifest = load_eval_manifest(args.manifest)
    selected_manifest = (
        limit_manifest_molecules(manifest, args.max_molecules)
        if args.max_molecules is not None
        else manifest
    )
    dataset = FlexBondInferenceDataset(args.inference_cache, args.split)
    inference = validate_dataset_against_manifest(dataset, selected_manifest)
    records = _validate_core_records(
        payload.get("records"), selected_manifest, args.expected_method
    )

    try:
        validate_sample_payload_provenance(
            payload,
            manifest=manifest,
            manifest_path=args.manifest,
            split=args.split,
            inference_cache_path=args.inference_cache,
            inference_by_id=inference,
        )
    except ValueError:
        extra = {
            key: value
            for key, value in payload.items()
            if key not in {"records", "manifest", "manifest_provenance"}
        }
        repaired = build_manifest_aware_sample_payload(
            records=records,
            manifest=manifest,
            manifest_path=args.manifest,
            selected_manifest=selected_manifest,
            split=args.split,
            inference_cache_path=args.inference_cache,
            inference_by_id=inference,
            extra=extra,
        )
        temporary = args.payload.with_name(args.payload.name + ".provenance.tmp")
        torch.save(repaired, temporary)
        os.replace(temporary, args.payload)
        print(f"REPAIRED provenance metadata: {args.payload}")
    else:
        print(f"VALID provenance metadata: {args.payload}")


if __name__ == "__main__":
    main()
