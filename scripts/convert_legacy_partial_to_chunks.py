#!/usr/bin/env python
"""Safely convert one validated legacy Global4D partial into durable chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import torch

from etflow.commons.global4d_chunked_persistence import (
    compact_sampling_state,
    convert_records_to_chunks,
    ordered_sample_ids_sha256,
    utc_now,
)
from etflow.commons.global_coupled_4d_sampling import (
    atomic_json_save,
    checkpoint_inference_identity,
    file_sha256,
)
from etflow.data.flexbond_eval_manifest import (
    build_sample_payload_provenance,
    limit_manifest_molecules,
    load_eval_manifest,
    manifest_content_sha256,
    validate_dataset_against_manifest,
)
from etflow.data.flexbond_inference_dataset import FlexBondInferenceDataset


def _provenance_identity(value: dict) -> dict:
    value = dict(value)
    manifest = dict(value.get("manifest") or {})
    cache = dict(value.get("inference_cache") or {})
    manifest.pop("path", None)
    cache.pop("path", None)
    value["manifest"] = manifest
    value["inference_cache"] = cache
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy_partial", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_molecules", type=int)
    parser.add_argument("--save_every_records", type=int, default=50)
    parser.add_argument("--chunks_dir", type=Path)
    parser.add_argument("--state", type=Path)
    args = parser.parse_args()
    if args.save_every_records < 1:
        parser.error("--save_every_records must be positive")
    if not args.legacy_partial.is_file() or args.legacy_partial.is_symlink():
        parser.error("--legacy_partial must be a regular non-symlink file")

    chunks_dir = args.chunks_dir or args.legacy_partial.parent / "partial_chunks"
    state_path = args.state or args.legacy_partial.parent / "sampling_state.json"
    payload = torch.load(args.legacy_partial, map_location="cpu", weights_only=False)
    if payload.get("partial") is not True:
        raise ValueError("Legacy payload is not marked partial")
    records = list(payload.get("records") or [])
    run_identity = dict(payload.get("run_identity") or {})
    if not run_identity:
        raise ValueError("Legacy payload is missing run_identity")

    manifest = load_eval_manifest(args.manifest)
    selected = (
        limit_manifest_molecules(manifest, args.max_molecules)
        if args.max_molecules is not None
        else manifest
    )
    selected_rows = selected["records"]
    if len(records) > len(selected_rows):
        raise ValueError("Legacy payload has more records than the selected manifest")
    completed = {**selected, "records": selected_rows[: len(records)]}
    dataset = FlexBondInferenceDataset(args.cache_dir, args.split)
    inference = validate_dataset_against_manifest(dataset, selected)
    expected_provenance = build_sample_payload_provenance(
        manifest=manifest,
        manifest_path=args.manifest,
        selected_manifest=completed,
        split=args.split,
        inference_cache_path=args.cache_dir,
        inference_by_id=inference,
        records=records,
    )
    if _provenance_identity(dict(payload.get("manifest_provenance") or {})) != (
        _provenance_identity(expected_provenance)
    ):
        raise ValueError("Legacy partial manifest provenance validation failed")

    checkpoint = checkpoint_inference_identity(args.checkpoint)
    required_identity = {
        "checkpoint_inference_sha256": checkpoint["inference_sha256"],
        "config_sha256": file_sha256(args.config),
        "manifest_sha256": manifest_content_sha256(manifest),
        "split": args.split,
    }
    for key, expected in required_identity.items():
        if run_identity.get(key) != expected:
            raise ValueError(f"Legacy partial {key} does not match the requested input")

    final_scan = convert_records_to_chunks(
        records,
        chunks_dir,
        selected_rows=selected_rows,
        run_identity=run_identity,
        save_every_records=args.save_every_records,
    )
    atomic_json_save(
        compact_sampling_state(
            status="PARTIAL",
            completed_count=len(records),
            total_count=len(selected_rows),
            completed_chunk_count=final_scan.chunk_count,
            current_chunk_size=0,
            save_every_records=args.save_every_records,
            run_identity=run_identity,
            ordered_sample_ids_hash=ordered_sample_ids_sha256(
                [str(row["sample_id"]) for row in selected_rows]
            ),
            output=args.legacy_partial.parent / "samples.pt",
            device="conversion",
            started_at=utc_now(),
            latest_chunk_sha256=final_scan.latest_chunk_sha256,
            legacy_source_sha256=file_sha256(args.legacy_partial),
        ),
        state_path,
    )
    print(
        f"Converted {len(records)} records into {final_scan.chunk_count} chunks; "
        f"legacy source retained at {args.legacy_partial}"
    )


if __name__ == "__main__":
    main()
