#!/usr/bin/env python
"""Create an immutable ordered subset of an existing prediction cache."""

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

from etflow.ecir.v8_validation_cache import (
    append_prediction_chunk, atomic_json, canonical_sha256, finish_prediction_manifest,
    iter_prediction_records, new_prediction_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=250)
    args = parser.parse_args()
    source_manifest = json.loads(args.input.resolve().read_text(encoding="utf-8"))
    selection = json.loads(args.selection_manifest.resolve().read_text(encoding="utf-8"))
    by_index = {int(row["record_index"]): row for row in iter_prediction_records(args.input.resolve())}
    records = [by_index[int(index)] for index in selection["record_indices"]]
    if [str(row["sample_id"]) for row in records] != [str(value) for value in selection["record_ids"]]:
        raise RuntimeError("prediction subset identity changed")
    identity = dict(source_manifest["identity"])
    identity.update({
        "record_count": len(records),
        "record_identity_sha256": canonical_sha256([str(row["sample_id"]) for row in records]),
        "parent_prediction_manifest_sha256": canonical_sha256(source_manifest),
    })
    identity["identity_sha256"] = canonical_sha256({key: value for key, value in identity.items() if key != "identity_sha256"})
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "prediction_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETED" and existing.get("identity") == identity:
            print(json.dumps({"status": "COMPLETED", "resumed": True, "records": len(records)}))
            return
        raise RuntimeError("output subset cache exists with different identity")
    manifest = new_prediction_manifest(identity, chunk_size=args.chunk_size, output_dir=output)
    manifest["subset_selection_manifest"] = str(args.selection_manifest.resolve())
    atomic_json(path, manifest)
    for start in range(0, len(records), args.chunk_size):
        end = min(start + args.chunk_size, len(records))
        append_prediction_chunk(path, manifest, record_start=start, record_end=end, records=records[start:end])
    finish_prediction_manifest(path, manifest)
    print(json.dumps({"status": "COMPLETED", "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
