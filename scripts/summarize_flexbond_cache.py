#!/usr/bin/env python
"""Report record and unique-molecule counts for FlexBond cache splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from etflow.commons.record_identity import source_record_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = {}
    for split in args.splits:
        root = args.cache_dir / split
        files = sorted(root.glob("*.pt"))
        molecule_ids = set()
        for path in files:
            record = torch.load(path, map_location="cpu", weights_only=False)
            molecule_ids.add(source_record_identity(record))
        summary[split] = {
            "cache_records": len(files),
            "unique_molecules": len(molecule_ids),
        }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
