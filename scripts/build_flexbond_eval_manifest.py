#!/usr/bin/env python
"""Freeze a fair evaluation cohort from a label-free inference cache."""

import argparse
from pathlib import Path

from etflow.data.flexbond_eval_manifest import (
    build_eval_manifest,
    limit_manifest_molecules,
    write_eval_manifest,
)
from etflow.data.flexbond_inference_dataset import FlexBondInferenceDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_molecules", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dataset = FlexBondInferenceDataset(args.cache_dir, args.split)
    manifest = build_eval_manifest(dataset)
    if args.max_molecules is not None:
        manifest = limit_manifest_molecules(manifest, args.max_molecules)
    write_eval_manifest(args.output, manifest)
    print(f"Wrote {len(manifest['records'])} manifest rows to {args.output}")


if __name__ == "__main__":
    main()
