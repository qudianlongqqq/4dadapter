#!/usr/bin/env python
"""Materialize the immutable formal-large validation source/topology cache."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from etflow.ecir.chemical_validity import ChemicalValidity
from etflow.ecir.v8_validation_cache import (
    ValidationStatus,
    append_prediction_chunk,
    atomic_json,
    canonical_sha256,
    completed_chunk_ranges,
    file_sha256,
    finish_prediction_manifest,
    iter_prediction_records,
    new_prediction_manifest,
    prediction_identity,
    validate_manifest_identity,
)
from scripts.train_ecir_mvr_v8 import _assert_manifest, _real_dataset


EVALUATOR_SHA = "47f123b59933d9f856bb65133082594451273b587010a25328756735d5e73bc1"
SAFETY_SHA = "1664147678a65086c23cd8df57af79a2a573da1df25ba9f762c5edf4b7b6614a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-sources", type=Path, required=True)
    parser.add_argument("--val-targets", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--validity-statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.chunk_size < 1:
        raise ValueError("source cache chunk size must be positive")
    sources = args.val_sources.resolve()
    targets = args.val_targets.resolve()
    frame = _assert_manifest(sources, "val")
    _assert_manifest(targets, "val")
    record_ids = frame.sample_id.astype(str).tolist()
    if args.max_records is not None:
        if args.max_records < 1:
            raise ValueError("max-records must be positive")
        record_ids = record_ids[: args.max_records]
    identity = prediction_identity(
        method="Source",
        checkpoint_sha256="SOURCE_HAS_NO_CHECKPOINT",
        config_sha256=canonical_sha256({"canonical_constraints": True}),
        validation_sources_sha256=file_sha256(sources),
        validation_targets_sha256=file_sha256(targets),
        record_ids=record_ids,
        evaluator_semantics_sha256=EVALUATOR_SHA,
        safety_semantics_sha256=SAFETY_SHA,
    )
    output = args.output_dir.resolve()
    manifest_path = output / "prediction_manifest.json"
    status = ValidationStatus.start(
        output / "status.json",
        phase="FULL_PREDICTING",
        training_step=0,
        validation_mode="SOURCE_CACHE",
    )
    started = time.perf_counter()
    try:
        if manifest_path.exists():
            if not args.resume:
                raise RuntimeError("source cache exists; use --resume after identity verification")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_manifest_identity(manifest, identity)
            list(iter_prediction_records(manifest_path, require_completed=False))
        else:
            manifest = new_prediction_manifest(identity, chunk_size=args.chunk_size, output_dir=output)
            atomic_json(manifest_path, manifest)
        completed = completed_chunk_ranges(manifest)
        validity = ChemicalValidity(args.validity_statistics.resolve())
        dataset = _real_dataset(
            sources,
            targets,
            validity,
            source_cache_root=args.source_cache_root.resolve(),
            target_cache_root=args.target_cache_root.resolve(),
            source_identity="validation-only-not-used-for-training-scales",
        )
        count = len(record_ids)
        for start in range(0, count, args.chunk_size):
            end = min(start + args.chunk_size, count)
            if (start, end) in completed:
                continue
            rows = []
            for index in range(start, end):
                item, record = dataset.get_item_and_record(index)
                if str(item.sample_id) != record_ids[index]:
                    raise RuntimeError("source cache record order changed")
                rows.append(
                    {
                        "record_index": index,
                        "sample_id": str(item.sample_id),
                        "molecule_id": str(item.molecule_id),
                        "item": item,
                        "record": record,
                    }
                )
            append_prediction_chunk(
                manifest_path,
                manifest,
                record_start=start,
                record_end=end,
                records=rows,
            )
            elapsed = time.perf_counter() - started
            status.update(
                current_validation_record=end,
                prediction_chunks_completed=len(manifest["chunks"]),
                records_per_second=end / max(elapsed, 1.0e-9),
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=(count - end) * elapsed / max(end, 1),
            )
        finish_prediction_manifest(manifest_path, manifest)
        elapsed = time.perf_counter() - started
        status.update(
            status="COMPLETED",
            phase="COMPLETED",
            current_validation_record=count,
            prediction_chunks_completed=len(manifest["chunks"]),
            records_per_second=count / max(elapsed, 1.0e-9),
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=0.0,
        )
    except BaseException as error:
        status.fail(error, elapsed_seconds=time.perf_counter() - started)
        raise


if __name__ == "__main__":
    main()
