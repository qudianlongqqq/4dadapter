#!/usr/bin/env python
"""Validate atom order and full directed-graph consistency for every cache record."""

import argparse
from pathlib import Path

import torch

from etflow.data.flexbond_optimizer_dataset import validate_cache_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    root = args.cache_dir / args.split if (args.cache_dir / args.split).is_dir() else args.cache_dir
    files = sorted(root.glob("*.pt"))
    if not files:
        raise SystemExit(f"No cache records found in {root}")
    for path in files:
        record = torch.load(path, map_location="cpu", weights_only=False)
        check = validate_cache_record(record, require_persisted_pair=True)
        if abs(check["rmsd_after"] - float(record["rmsd_after"])) > 1.0e-5:
            raise ValueError(f"Persisted Kabsch diagnostics differ for {path}")
    print(
        "PASS: validated atom order, topology, graph tensors, and Kabsch "
        f"for {len(files)} records"
    )


if __name__ == "__main__":
    main()
