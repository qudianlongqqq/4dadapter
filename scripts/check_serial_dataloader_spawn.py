#!/usr/bin/env python
"""Windows spawn-safety check for the completed Serial Stage 2 cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

from torch_geometric.loader import DataLoader

from etflow.serial_global4d.cache import SerialGlobal4DResidualDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    dataset = SerialGlobal4DResidualDataset(args.cache_root, "train")
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=False,
    )
    rows = []
    for index, batch in enumerate(loader):
        rows.append(
            {
                "graphs": int(batch.num_graphs),
                "atoms": int(batch.num_nodes),
                "joints": int(batch.rotatable_bond_index.size(1)),
            }
        )
        if index == 1:
            break
    if len(rows) != 2:
        raise RuntimeError("spawn DataLoader did not yield two batches")
    print(json.dumps({"status": "PASS", "workers": args.workers, "batches": rows}))


if __name__ == "__main__":
    main()
