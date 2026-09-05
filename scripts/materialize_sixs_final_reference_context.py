#!/usr/bin/env python
"""Materialize the predeclared first Reference conformer as contextual rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import torch

try:
    from _bootstrap import bootstrap
except ModuleNotFoundError:
    from scripts._bootstrap import bootstrap

bootstrap()
from scripts.run_sixs_primary_final_coordinates import atomic_json, atomic_parquet, metric_row, sha256_file, write_sdf

EXPECTED = 5000


def main(args: argparse.Namespace) -> int:
    complete = json.loads((args.topology_cache / "COMPLETE.json").read_text(encoding="utf-8"))
    paths = [Path(value) for value in complete["chunks"]]
    if complete.get("protocol_sha256") != sha256_file(args.protocol) or not all(path.is_file() for path in paths):
        raise RuntimeError("reference context topology binding mismatch")
    topology = {}
    rows = []
    for path in paths:
        for item in torch.load(path, map_location="cpu", weights_only=False)["items"]:
            rank = int(item["final_molecule_index"])
            topology[rank] = {"mol_binary": item["mol_binary"]}
            reference0 = torch.as_tensor(item["references"][0], dtype=torch.float32)
            for source_row in item["source_records"]:
                source_payload = torch.load(source_row["record_asset"], map_location="cpu", weights_only=False)
                source = torch.as_tensor(source_payload["source_coordinates"], dtype=torch.float32)
                rows.append({
                    "record_id": source_row["record_id"], "final_molecule_index": rank,
                    "molecule_id": item["molecule_id"], "molecule_identity_sha256": item["molecule_identity_sha256"],
                    "etflow_record_index": int(source_row["etflow_record_index"]), "method": "reference_context",
                    "formulation": "Reference contextual comparator", "seed": None,
                    "proposal_coordinates": reference0, **metric_row(source, reference0, item["references"], item["graph"]),
                    "reference_context_definition": "first frozen Reference conformer; duplicated only for matched Source record identity",
                })
    rows.sort(key=lambda row: (row["final_molecule_index"], row["etflow_record_index"]))
    if len(rows) != EXPECTED or len({row["record_id"] for row in rows}) != EXPECTED:
        raise RuntimeError("reference context denominator mismatch")
    write_sdf(args.output_dir / "COORDINATES.sdf", rows, topology, "reference_context")
    atomic_parquet(args.output_dir / "PER_RECORD.parquet", pd.DataFrame([
        {key: value for key, value in row.items() if key != "proposal_coordinates"} for row in rows
    ]))
    atomic_json(args.output_dir / "COMPLETE.json", {
        "status": "PASS", "records": EXPECTED, "molecules": 2500,
        "contextual_only": True, "executable_method": False, "theoretical_upper_bound": False,
        "reference_self_cov_amr": "NOT_REPORTED", "reference_conformer_rule": "reference0",
        "sdf_sha256": sha256_file(args.output_dir / "COORDINATES.sdf"),
        "per_record_sha256": sha256_file(args.output_dir / "PER_RECORD.parquet"),
    })
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--topology-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("protocol", "topology_cache", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    return args


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
