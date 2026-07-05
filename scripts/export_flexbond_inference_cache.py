#!/usr/bin/env python
"""Export a strictly label-free inference cache from validated training pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from etflow.data.flexbond_cache_schema import (
    INFERENCE_FORBIDDEN_FIELDS,
    validate_inference_record,
    x_init_sha256,
)
from etflow.data.flexbond_optimizer_dataset import validate_cache_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    source = (
        args.cache_dir / args.split
        if (args.cache_dir / args.split).is_dir()
        else args.cache_dir
    )
    destination = args.output_dir / args.split
    destination.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob("*.pt"))
    if not files:
        raise SystemExit(f"No cache files found in {source}")
    for path in files:
        record = torch.load(path, map_location="cpu", weights_only=False)
        validate_cache_record(record, require_persisted_pair=True)
        inference = {
            key: value
            for key, value in record.items()
            if key not in INFERENCE_FORBIDDEN_FIELDS
            and not key.startswith("x_ref")
            and not key.startswith("selected_ref")
            and key not in {"rmsd_before", "rmsd_after"}
        }
        inference["sample_id"] = str(record.get("sample_id", record["mol_id"]))
        inference["x_init_hash"] = x_init_sha256(
            inference["x_init"], inference["atomic_numbers"]
        )
        validate_inference_record(inference)
        output = destination / path.name
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite inference cache {output}")
        torch.save(inference, output)
    print(f"Exported {len(files)} label-free records to {destination}")


if __name__ == "__main__":
    main()
