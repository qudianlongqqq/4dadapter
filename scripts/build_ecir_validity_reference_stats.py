#!/usr/bin/env python
"""Fit MCVR chemical-validity thresholds from train references only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()

import pandas as pd
import torch

from etflow.ecir.chemical_validity import build_validity_reference_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_atlas", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_references_per_molecule", type=int, default=32)
    args = parser.parse_args()
    atlas = pd.read_parquet(args.train_atlas)
    if set(atlas.split.unique()) != {"train"}:
        raise ValueError("validity statistics must use train split only")
    seen = set(); records = []
    for row in atlas.sort_values(["molecule_id", "sample_id"]).itertuples(index=False):
        if row.molecule_id in seen:
            continue
        record = torch.load(Path(row.source_path), map_location="cpu", weights_only=False)
        references = torch.as_tensor(
            record.get("x_ref_candidates", record.get("x_ref_aligned")), dtype=torch.float32
        )
        if references.ndim == 2:
            references = references.unsqueeze(0)
        records.append((record, references[: args.max_references_per_molecule]))
        seen.add(row.molecule_id)
    stats = build_validity_reference_statistics(
        records,
        train_split_sha256=hashlib.sha256(args.train_atlas.read_bytes()).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "identity_sha256": stats["identity_sha256"], **stats["source"]}, indent=2))


if __name__ == "__main__":
    main()
